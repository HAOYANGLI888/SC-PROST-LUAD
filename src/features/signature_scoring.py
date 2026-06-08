"""Bulk expression scoring for curated or single-cell-derived signatures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.scrna_signature import CellStateSignature, normalize_gene_symbol


class SignatureScoringError(RuntimeError):
    """Raised when expression or signature scoring cannot proceed."""


@dataclass(frozen=True)
class ScoreResult:
    """Signature scores and per-signature missingness audit."""

    scores: pd.DataFrame
    missingness: pd.DataFrame
    method: str


def _normalize_gene_columns(columns: list[object]) -> list[str]:
    return [normalize_gene_symbol(column) for column in columns]


def _collapse_duplicate_gene_symbols(expression: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    """Collapse duplicated gene symbols by mean and keep the sample column."""

    sample = expression[[sample_col]].reset_index(drop=True)
    gene_frame = expression.loc[:, [column != sample_col for column in expression.columns]]
    genes = list(gene_frame.columns)
    if not genes:
        raise SignatureScoringError("Expression matrix contains no gene columns.")
    numeric = gene_frame.apply(pd.to_numeric, errors="coerce")
    numeric.columns = _normalize_gene_columns(genes)
    keep = [column != "" for column in numeric.columns]
    numeric = numeric.loc[:, keep]
    if numeric.empty:
        raise SignatureScoringError("Expression matrix has no usable gene-symbol columns.")
    collapsed = numeric.T.groupby(level=0, sort=True).mean().T
    return pd.concat(
        [sample, collapsed.reset_index(drop=True)],
        axis=1,
    )


def map_ensembl_columns_to_symbols(
    expression: pd.DataFrame,
    annotation: pd.DataFrame,
    *,
    sample_col: str,
) -> pd.DataFrame:
    """Map Ensembl gene columns to gene symbols and collapse duplicate symbols."""

    required = {"gene_id", "gene_symbol"}
    missing = sorted(required - set(annotation.columns))
    if missing:
        raise SignatureScoringError(f"Gene annotation is missing columns: {missing}")
    symbol_by_gene = {
        str(row.gene_id).split(".")[0]: normalize_gene_symbol(row.gene_symbol)
        for row in annotation.itertuples(index=False)
    }
    rename: dict[str, str] = {}
    for column in expression.columns:
        if column == sample_col:
            continue
        base = str(column).split(".")[0]
        symbol = symbol_by_gene.get(base, "")
        if symbol:
            rename[column] = symbol
    if not rename:
        raise SignatureScoringError("No expression columns overlapped the Ensembl-to-symbol annotation.")
    mapped = expression[[sample_col, *rename.keys()]].rename(columns=rename)
    return _collapse_duplicate_gene_symbols(mapped, sample_col)


def log2_transform_if_needed(expression: pd.DataFrame, *, sample_col: str) -> tuple[pd.DataFrame, str]:
    """Apply log2(x + 1) when expression appears to be raw TPM/count-like."""

    genes = [column for column in expression.columns if column != sample_col]
    numeric = expression[genes].apply(pd.to_numeric, errors="coerce")
    q99 = float(np.nanquantile(numeric.to_numpy(dtype=float), 0.99)) if numeric.size else 0.0
    if q99 > 50.0:
        if (numeric < 0).any().any():
            raise SignatureScoringError("Expression has negative values and cannot be log2(x + 1) transformed.")
        numeric = np.log2(numeric + 1.0)
        method = "auto_log2_plus_1"
    else:
        method = "as_provided"
    return pd.concat([expression[[sample_col]].reset_index(drop=True), numeric.reset_index(drop=True)], axis=1), method


def zscore_by_gene(expression: pd.DataFrame, *, sample_col: str) -> pd.DataFrame:
    """Z-score each gene within one dataset."""

    genes = [column for column in expression.columns if column != sample_col]
    numeric = expression[genes].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median(axis=0).fillna(0.0)
    numeric = numeric.fillna(medians)
    scales = numeric.std(axis=0, ddof=0).replace(0.0, 1.0)
    zscore = (numeric - numeric.mean(axis=0)) / scales
    return pd.concat([expression[[sample_col]].reset_index(drop=True), zscore.reset_index(drop=True)], axis=1)


def _available_signature_genes(
    expression_genes: set[str],
    signature: CellStateSignature,
) -> tuple[list[str], list[str]]:
    available = [gene for gene in signature.genes if gene in expression_genes]
    missing = [gene for gene in signature.genes if gene not in expression_genes]
    return available, missing


def _missingness_rows(
    expression_genes: set[str],
    signatures: list[CellStateSignature],
    *,
    method: str,
    dataset_label: str,
) -> list[dict[str, object]]:
    rows = []
    for signature in signatures:
        available, missing = _available_signature_genes(expression_genes, signature)
        rows.append(
            {
                "dataset": dataset_label,
                "method": method,
                "signature_name": signature.signature_name,
                "cell_state": signature.cell_state,
                "category": signature.category,
                "expected_risk_direction": signature.expected_risk_direction,
                "n_signature_genes": len(signature.genes),
                "n_available_genes": len(available),
                "n_missing_genes": len(missing),
                "missing_fraction": len(missing) / max(len(signature.genes), 1),
                "available_genes": ";".join(available),
                "missing_genes": ";".join(missing),
            }
        )
    return rows


def score_signatures(
    expression: pd.DataFrame,
    signatures: list[CellStateSignature],
    *,
    sample_col: str,
    method: str = "mean_zscore",
    dataset_label: str = "dataset",
    min_genes: int = 2,
) -> ScoreResult:
    """Score each signature in a sample-by-gene expression matrix."""

    if sample_col not in expression.columns:
        raise SignatureScoringError(f"Expression matrix is missing sample column {sample_col!r}.")
    if method not in {"mean_zscore", "rank", "ssgsea_like"}:
        raise ValueError("method must be mean_zscore, rank, or ssgsea_like.")
    if not signatures:
        raise SignatureScoringError("At least one signature is required.")
    prepared = _collapse_duplicate_gene_symbols(expression, sample_col)
    gene_columns = [column for column in prepared.columns if column != sample_col]
    gene_set = set(gene_columns)
    missingness = pd.DataFrame(
        _missingness_rows(gene_set, signatures, method=method, dataset_label=dataset_label)
    )
    output = pd.DataFrame({sample_col: prepared[sample_col].astype(str).to_numpy()})

    if method == "mean_zscore":
        union = sorted(
            {
                gene
                for signature in signatures
                for gene in signature.genes
                if gene in gene_set
            }
        )
        if not union:
            raise SignatureScoringError("No signature genes were present in expression data.")
        zscore = zscore_by_gene(prepared[[sample_col, *union]], sample_col=sample_col)
        zmat = zscore.set_index(sample_col)
        for signature in signatures:
            available, _ = _available_signature_genes(gene_set, signature)
            output[signature.signature_name] = (
                zmat[available].mean(axis=1).to_numpy(dtype=float)
                if len(available) >= min_genes
                else np.nan
            )
    else:
        numeric = prepared[gene_columns].apply(pd.to_numeric, errors="coerce")
        medians = numeric.median(axis=0).fillna(0.0)
        numeric = numeric.fillna(medians)
        ranks = numeric.rank(axis=1, method="average", pct=True)
        for signature in signatures:
            available, _ = _available_signature_genes(gene_set, signature)
            if len(available) < min_genes:
                output[signature.signature_name] = np.nan
                continue
            if method == "rank":
                output[signature.signature_name] = ranks[available].mean(axis=1).to_numpy(dtype=float)
            else:
                signature_mean = ranks[available].mean(axis=1)
                background_mean = ranks.drop(columns=available).mean(axis=1)
                output[signature.signature_name] = (signature_mean - background_mean).to_numpy(dtype=float)
    return ScoreResult(scores=output, missingness=missingness, method=method)


def load_tcga_symbol_expression(
    root: str | Path,
    *,
    small_test: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load TCGA TPM, map Ensembl IDs to symbols, and return log-scale expression."""

    project_root = Path(root).resolve()
    matrix_path = project_root / "data" / "raw" / "tcga_luad" / "rnaseq" / "tcga_luad_tpm_matrix.csv"
    annotation_path = project_root / "data" / "metadata" / "stage2c_tcga_gene_annotation.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"TCGA RNA matrix not found: {matrix_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"TCGA gene annotation not found: {annotation_path}")
    nrows = 40 if small_test else None
    expression = pd.read_csv(matrix_path, nrows=nrows)
    if "patient_id" not in expression.columns:
        raise SignatureScoringError("TCGA expression matrix must contain patient_id.")
    annotation = pd.read_csv(annotation_path, dtype=str)
    mapped = map_ensembl_columns_to_symbols(expression, annotation, sample_col="patient_id")
    mapped, scale_method = log2_transform_if_needed(mapped, sample_col="patient_id")
    summary = {
        "source_matrix": str(matrix_path),
        "annotation": str(annotation_path),
        "samples": len(mapped),
        "genes": len(mapped.columns) - 1,
        "expression_scale_method": scale_method,
        "small_test": small_test,
    }
    return mapped, summary
