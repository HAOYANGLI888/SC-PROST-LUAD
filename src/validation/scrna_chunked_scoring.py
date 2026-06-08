"""Chunked selected-gene expression and fixed-program scoring."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from features.scrna_signature_scores import KEY_GENES, RISK_PROGRAMS
from validation.scrna_lowmem_reader import (
    iter_sparse_row_blocks,
    load_obs_with_analysis_labels,
)


SELECTED_GENES = sorted(
    set(KEY_GENES)
    | {gene for genes in RISK_PROGRAMS.values() for gene in genes}
)


def gene_lookup(h5ad_path: str | Path) -> dict[str, int]:
    import anndata as ad

    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        return {
            str(gene).upper(): index
            for index, gene in enumerate(data.var_names.astype(str))
        }
    finally:
        data.file.close()


def _normalized_selected(
    block: sparse.csr_matrix,
    indices: list[int],
    *,
    target_sum: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = block[:, indices].toarray().astype(np.float32, copy=False)
    total = np.asarray(block.sum(axis=1)).ravel().astype(np.float64)
    scale = np.divide(
        target_sum,
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    normalized = np.log1p(selected * scale[:, None]).astype(np.float32)
    return selected, normalized


def summarize_selected_gene_expression(
    h5ad_path: str | Path,
    *,
    block_size: int = 2048,
    target_sum: float = 10_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize fixed genes by harmonized cell type using log1p(CPM)."""

    lookup = gene_lookup(h5ad_path)
    present = [gene for gene in SELECTED_GENES if gene in lookup]
    missing = [gene for gene in SELECTED_GENES if gene not in lookup]
    indices = [lookup[gene] for gene in present]
    obs = load_obs_with_analysis_labels(h5ad_path)
    labels = obs["analysis_cell_type"].astype(str).to_numpy()
    cell_types = sorted(pd.unique(labels))
    label_codes = pd.Categorical(labels, categories=cell_types).codes
    sums = np.zeros((len(cell_types), len(present)), dtype=np.float64)
    expressing = np.zeros((len(cell_types), len(present)), dtype=np.int64)
    totals = np.bincount(label_codes, minlength=len(cell_types)).astype(np.int64)

    for start, end, block in iter_sparse_row_blocks(
        h5ad_path, block_size=block_size
    ):
        raw, normalized = _normalized_selected(
            block, indices, target_sum=target_sum
        )
        codes = label_codes[start:end]
        for code in np.unique(codes):
            mask = codes == code
            sums[code] += normalized[mask].sum(axis=0, dtype=np.float64)
            expressing[code] += (raw[mask] > 0).sum(axis=0)

    rows = []
    for code, cell_type in enumerate(cell_types):
        for gene_index, gene in enumerate(present):
            rows.append(
                {
                    "analysis_cell_type": cell_type,
                    "gene_symbol": gene,
                    "average_log1p_cpm": sums[code, gene_index] / totals[code],
                    "detection_rate": expressing[code, gene_index] / totals[code],
                    "expressing_cell_count": int(expressing[code, gene_index]),
                    "total_cell_count": int(totals[code]),
                }
            )
    missingness = pd.DataFrame(
        [
            {
                "gene_symbol": gene,
                "status": "present" if gene in lookup else "missing",
                "matrix_index": lookup.get(gene, ""),
            }
            for gene in SELECTED_GENES
        ]
    )
    return pd.DataFrame(rows), missingness


class _GroupAccumulator:
    def __init__(self, programs: list[str]):
        self.programs = programs
        self.counts: dict[str, int] = defaultdict(int)
        self.sums: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(len(programs), dtype=np.float64)
        )
        self.sumsq: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(len(programs), dtype=np.float64)
        )

    def update(self, labels: np.ndarray, scores: np.ndarray) -> None:
        for label in pd.unique(labels):
            mask = labels == label
            values = scores[mask]
            key = str(label)
            self.counts[key] += int(mask.sum())
            self.sums[key] += values.sum(axis=0, dtype=np.float64)
            self.sumsq[key] += np.square(values).sum(axis=0, dtype=np.float64)

    def frame(self, group_column: str) -> pd.DataFrame:
        rows = []
        for label in sorted(self.counts):
            count = self.counts[label]
            for index, program in enumerate(self.programs):
                mean = self.sums[label][index] / count
                variance = max(self.sumsq[label][index] / count - mean**2, 0.0)
                rows.append(
                    {
                        group_column: label,
                        "program": program,
                        "mean_log1p_cpm_score": mean,
                        "sd_log1p_cpm_score": np.sqrt(variance),
                        "cell_count": count,
                    }
                )
        return pd.DataFrame(rows)


def score_fixed_programs(
    h5ad_path: str | Path,
    *,
    block_size: int = 2048,
    target_sum: float = 10_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score fixed programs and aggregate without retaining all per-cell scores."""

    lookup = gene_lookup(h5ad_path)
    obs = load_obs_with_analysis_labels(h5ad_path)
    programs = list(RISK_PROGRAMS)
    present_by_program = {
        program: [gene for gene in genes if gene in lookup]
        for program, genes in RISK_PROGRAMS.items()
    }
    union = sorted({gene for genes in present_by_program.values() for gene in genes})
    union_indices = [lookup[gene] for gene in union]
    union_lookup = {gene: index for index, gene in enumerate(union)}
    program_positions = {
        program: [union_lookup[gene] for gene in genes]
        for program, genes in present_by_program.items()
    }
    accumulators = {
        "analysis_cell_type": _GroupAccumulator(programs),
        "sample_id": _GroupAccumulator(programs),
        "patient_id": _GroupAccumulator(programs),
    }
    label_arrays = {
        column: obs[column].astype(str).to_numpy()
        for column in accumulators
    }
    for start, end, block in iter_sparse_row_blocks(
        h5ad_path, block_size=block_size
    ):
        _, normalized = _normalized_selected(
            block, union_indices, target_sum=target_sum
        )
        scores = np.column_stack(
            [
                normalized[:, program_positions[program]].mean(axis=1)
                if program_positions[program]
                else np.full(end - start, np.nan, dtype=np.float32)
                for program in programs
            ]
        )
        for column, accumulator in accumulators.items():
            accumulator.update(label_arrays[column][start:end], scores)

    missingness = pd.DataFrame(
        [
            {
                "program": program,
                "requested_genes": ";".join(genes),
                "present_genes": ";".join(present_by_program[program]),
                "missing_genes": ";".join(
                    gene for gene in genes if gene not in lookup
                ),
                "n_requested": len(genes),
                "n_present": len(present_by_program[program]),
                "missing_fraction": (
                    len(genes) - len(present_by_program[program])
                )
                / len(genes),
                "scoring_method": "mean_log1p_CPM10000",
            }
            for program, genes in RISK_PROGRAMS.items()
        ]
    )
    return (
        accumulators["analysis_cell_type"].frame("analysis_cell_type"),
        accumulators["sample_id"].frame("sample_id"),
        accumulators["patient_id"].frame("patient_id"),
        missingness,
    )


def scores_for_matrix(
    matrix: sparse.csr_matrix,
    var_names: pd.Index,
    *,
    target_sum: float = 10_000.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return key-gene expression and program scores for a sampled matrix."""

    lookup = {str(gene).upper(): index for index, gene in enumerate(var_names)}
    requested = sorted(
        set(KEY_GENES)
        | {gene for genes in RISK_PROGRAMS.values() for gene in genes}
    )
    present = [gene for gene in requested if gene in lookup]
    _, normalized = _normalized_selected(
        matrix, [lookup[gene] for gene in present], target_sum=target_sum
    )
    position = {gene: index for index, gene in enumerate(present)}
    key_values = {
        gene: (
            normalized[:, position[gene]]
            if gene in position
            else np.full(matrix.shape[0], np.nan)
        )
        for gene in KEY_GENES
    }
    program_values = {}
    for program, genes in RISK_PROGRAMS.items():
        available = [position[gene] for gene in genes if gene in position]
        program_values[program] = (
            normalized[:, available].mean(axis=1)
            if available
            else np.full(matrix.shape[0], np.nan)
        )
    return key_values, program_values
