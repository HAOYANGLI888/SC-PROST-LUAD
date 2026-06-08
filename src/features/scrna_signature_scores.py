"""Fixed Stage 4B program scores for raw single-cell expression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from data.scrna_raw_import import sanitize_anndata_strings


RISK_PROGRAMS = {
    "hypoxia": ["LDHA", "CA9", "SLC2A1", "VEGFA", "BNIP3", "EGLN3"],
    "proliferation": ["MKI67", "CDK1", "TOP2A", "PCNA", "MCM2", "MCM5", "CCNB1"],
    "emt_like": ["VIM", "FN1", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "CDH2", "ITGA5"],
    "caf_matrix": ["COL1A1", "COL1A2", "COL3A1", "ACTA2", "FAP", "PDGFRB", "TAGLN"],
    "dendritic_b_plasma_context": [
        "CD74", "HLA-DRA", "HLA-DRB1", "MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN", "XBP1"
    ],
}
KEY_GENES = ["LDHA", "MKI67", "CDK1"]


def _mean_expression(data, indices: list[int]) -> np.ndarray:
    matrix = data.X[:, indices]
    return np.asarray(matrix.mean(axis=1)).ravel()


def score_programs(
    input_h5ad: str | Path,
    scored_h5ad: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """Compute deterministic mean-expression z scores without survival outcomes."""

    import scanpy as sc

    data = sc.read_h5ad(input_h5ad)
    lookup = {str(gene).upper(): index for index, gene in enumerate(data.var_names)}
    per_cell = pd.DataFrame(
        {
            "cell_id": data.obs_names.astype(str),
            "analysis_cell_type": data.obs["analysis_cell_type"].astype(str).to_numpy(),
        }
    )
    for column in ("Sample", "Sample_Origin", "Cell_subtype"):
        if column in data.obs:
            per_cell[column] = data.obs[column].astype(str).to_numpy()
    missing_rows = []
    for program, genes in RISK_PROGRAMS.items():
        present = [gene for gene in genes if gene in lookup]
        missing = [gene for gene in genes if gene not in lookup]
        raw_score = _mean_expression(data, [lookup[gene] for gene in present]) if present else np.full(data.n_obs, np.nan)
        standard_deviation = float(np.nanstd(raw_score))
        score = (
            (raw_score - float(np.nanmean(raw_score))) / standard_deviation
            if standard_deviation > 0
            else np.zeros_like(raw_score)
        )
        data.obs[f"{program}_score"] = score
        per_cell[f"{program}_score"] = score
        missing_rows.append(
            {
                "program": program,
                "requested_genes": ";".join(genes),
                "present_genes": ";".join(present),
                "missing_genes": ";".join(missing),
                "n_requested": len(genes),
                "n_present": len(present),
                "missing_fraction": len(missing) / len(genes),
            }
        )

    score_columns = [f"{program}_score" for program in RISK_PROGRAMS]
    celltype_scores = (
        per_cell.groupby("analysis_cell_type", observed=True)[score_columns]
        .agg(["mean", "median", "count"])
    )
    celltype_scores.columns = [
        f"{score}_{stat}" for score, stat in celltype_scores.columns
    ]
    celltype_scores = celltype_scores.reset_index()

    genes_to_summarize = sorted(set(KEY_GENES) | {g for values in RISK_PROGRAMS.values() for g in values})
    labels = data.obs["analysis_cell_type"].astype(str)
    expression_rows = []
    for cell_type in sorted(labels.unique()):
        mask = labels.eq(cell_type).to_numpy()
        for gene in genes_to_summarize:
            if gene not in lookup:
                continue
            matrix = data.X[mask, lookup[gene]]
            values = matrix.toarray().ravel() if sparse.issparse(matrix) else np.asarray(matrix).ravel()
            expression_rows.append(
                {
                    "analysis_cell_type": cell_type,
                    "gene_symbol": gene,
                    "cell_count": int(mask.sum()),
                    "average_log_expression": float(np.mean(values)),
                    "detection_fraction": float(np.mean(values > 0)),
                }
            )
    gene_expression = pd.DataFrame(expression_rows)
    output = Path(scored_h5ad)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.uns["stage4b_program_definitions"] = {
        key: list(value) for key, value in RISK_PROGRAMS.items()
    }
    sanitize_anndata_strings(data).write_h5ad(output, compression="gzip")
    return (
        per_cell,
        celltype_scores,
        gene_expression,
        pd.DataFrame(missing_rows),
        data,
    )
