"""RNA-seq loading and train-only preprocessing for Stage 2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TCGA_PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


class RNASeqPreprocessError(RuntimeError):
    """Raised when an RNA-seq matrix cannot be used safely."""


def normalize_tcga_patient_id(value: object) -> str | None:
    """Return the first 12 characters of a valid TCGA barcode."""

    match = TCGA_PATIENT_RE.match(str(value or "").strip().upper())
    return match.group(1) if match else None


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"RNA-seq matrix not found: {path}. "
            "See docs/stage2_rnaseq_manual_download_guide.md."
        )
    if path.stat().st_size == 0:
        raise RNASeqPreprocessError(f"RNA-seq matrix is empty: {path}")
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    try:
        return pd.read_csv(path, sep=separator)
    except Exception as exc:
        raise RNASeqPreprocessError(f"Failed to read RNA-seq matrix: {path}") from exc


def load_rnaseq_matrix(
    path: str | Path,
    *,
    input_scale: str = "tpm",
) -> pd.DataFrame:
    """Load an RNA-seq matrix and return patient rows with log2(TPM + 1).

    Accepted layouts:
    - Wide: first column contains TCGA sample/patient IDs and genes are columns.
    - Transposed: first column contains gene IDs and TCGA samples are columns.
    """

    matrix_path = Path(path)
    raw = _read_table(matrix_path)
    if raw.empty or raw.shape[1] < 2:
        raise RNASeqPreprocessError("RNA-seq matrix must contain IDs and gene values.")

    first_column = raw.columns[0]
    first_values = raw[first_column].map(normalize_tcga_patient_id)
    column_patient_ids = [normalize_tcga_patient_id(column) for column in raw.columns[1:]]
    row_layout = first_values.notna().mean() >= 0.5
    column_layout = np.mean([item is not None for item in column_patient_ids]) >= 0.5

    if row_layout:
        expression = raw.copy()
        if first_column == "patient_id":
            expression["patient_id"] = first_values
        else:
            expression.insert(0, "patient_id", first_values)
            expression = expression.drop(columns=[first_column])
    elif column_layout:
        transposed = raw.set_index(first_column).T
        transposed.index.name = "sample_id"
        expression = transposed.reset_index()
        expression.insert(0, "patient_id", expression["sample_id"].map(normalize_tcga_patient_id))
        expression = expression.drop(columns=["sample_id"])
    else:
        raise RNASeqPreprocessError(
            "Could not find TCGA barcodes in RNA-seq rows or columns. "
            "Expected values such as TCGA-05-4244 or TCGA-05-4244-01A."
        )

    expression = expression.dropna(subset=["patient_id"]).copy()
    gene_columns = [column for column in expression.columns if column != "patient_id"]
    if not gene_columns:
        raise RNASeqPreprocessError("RNA-seq matrix has no gene columns after parsing.")
    numeric_expression = expression[gene_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    expression = pd.concat(
        [expression[["patient_id"]].reset_index(drop=True), numeric_expression.reset_index(drop=True)],
        axis=1,
    )
    expression = expression.groupby("patient_id", as_index=False)[gene_columns].mean()

    if input_scale == "tpm":
        numeric = expression[gene_columns]
        if (numeric.dropna() < 0).any().any():
            raise RNASeqPreprocessError("TPM input contains negative values.")
        expression[gene_columns] = np.log2(numeric + 1.0)
    elif input_scale != "log2_tpm":
        raise ValueError("input_scale must be 'tpm' or 'log2_tpm'.")

    if expression.empty:
        raise RNASeqPreprocessError("RNA-seq parsing returned no valid TCGA patients.")
    return expression


@dataclass
class RNATrainPreprocessor:
    """Filter, impute, and scale RNA features using training patients only."""

    missing_fraction_max: float = 0.20
    expressed_fraction_min: float = 0.10
    log2_expression_min: float = 0.1
    top_variable_genes: int = 3000

    def fit(self, frame: pd.DataFrame, patient_ids: Iterable[str] | None = None) -> "RNATrainPreprocessor":
        if frame.empty:
            raise RNASeqPreprocessError("Cannot fit RNA preprocessor on empty training data.")
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        missing_ok = numeric.isna().mean() <= self.missing_fraction_max
        expressed_ok = (numeric.fillna(0.0) > self.log2_expression_min).mean() >= self.expressed_fraction_min
        selected = numeric.columns[missing_ok & expressed_ok].tolist()
        if not selected:
            raise RNASeqPreprocessError("All RNA features were removed by train-only filters.")
        variances = numeric[selected].var(axis=0, skipna=True).fillna(0.0)
        selected = variances.sort_values(ascending=False).head(self.top_variable_genes).index.tolist()
        medians = numeric[selected].median(axis=0).fillna(0.0)
        imputed = numeric[selected].fillna(medians)
        means = imputed.mean(axis=0)
        scales = imputed.std(axis=0, ddof=0).replace(0.0, 1.0)

        self.selected_genes_ = selected
        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        resolved_patient_ids = frame.index.astype(str).tolist() if patient_ids is None else patient_ids
        self.fit_patient_ids_ = tuple(resolved_patient_ids)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        missing_columns = sorted(set(self.selected_genes_) - set(frame.columns))
        if missing_columns:
            raise RNASeqPreprocessError(
                f"RNA matrix is missing {len(missing_columns)} selected genes, "
                f"including: {missing_columns[:5]}"
            )
        numeric = frame[self.selected_genes_].apply(pd.to_numeric, errors="coerce")
        imputed = numeric.fillna(self.medians_)
        scaled = (imputed - self.means_) / self.scales_
        return scaled.astype("float32")

    def fit_transform(
        self,
        frame: pd.DataFrame,
        patient_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return self.fit(frame, patient_ids=patient_ids).transform(frame)

    def _check_fitted(self) -> None:
        if not hasattr(self, "selected_genes_"):
            raise RNASeqPreprocessError("RNA preprocessor must be fit on training data first.")


def candidate_feature_table(expression: pd.DataFrame) -> pd.DataFrame:
    """Describe unfiltered candidate genes without fitting selectors globally."""

    genes = [column for column in expression.columns if column != "patient_id"]
    return pd.DataFrame(
        {
            "feature": genes,
            "feature_type": "rnaseq_candidate",
            "selection_scope": "candidate_only_train_fit_required",
        }
    )
