"""Clinical OS parsing and train-only covariate preprocessing for Stage 2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data.rnaseq_preprocess import normalize_tcga_patient_id


class ClinicalPreprocessError(RuntimeError):
    """Raised when clinical survival fields are absent or inconsistent."""


CLINICAL_COLUMNS = ["age", "male", "stage_numeric"]
STAGE_RE = re.compile(r"(?:stage\s*)?(iv|iii|ii|i|[1-4])", re.IGNORECASE)
ROMAN_STAGE = {"i": 1.0, "ii": 2.0, "iii": 3.0, "iv": 4.0}


def _first_present(values: pd.Series) -> object:
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() not in {"nan", "na", "n/a", "not reported", "unknown"}:
            return value
    return np.nan


def stage_to_numeric(value: object) -> float:
    """Map AJCC stage strings to a coarse ordinal stage I-IV covariate."""

    match = STAGE_RE.search(str(value or "").strip())
    if not match:
        return np.nan
    token = match.group(1).lower()
    return ROMAN_STAGE.get(token, float(token) if token.isdigit() else np.nan)


def _status_to_int(value: object) -> float:
    text = str(value or "").strip().lower()
    if text in {"1", "1.0", "dead", "deceased", "true"}:
        return 1.0
    if text in {"0", "0.0", "alive", "living", "false"}:
        return 0.0
    return np.nan


def load_tcga_cdr_os(path: str | Path) -> pd.DataFrame:
    """Load LUAD OS and core clinical covariates from the cached TCGA-CDR table."""

    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"TCGA-CDR survival table not found: {table_path}")
    if table_path.stat().st_size == 0:
        raise ClinicalPreprocessError(f"TCGA-CDR survival table is empty: {table_path}")
    frame = pd.read_csv(table_path, sep="\t", dtype=str)
    aliases = {
        "patient_id": ["_PATIENT", "patient_id", "bcr_patient_barcode"],
        "cancer": ["cancer type abbreviation", "cancer_type"],
        "os_event": ["OS", "os_event"],
        "os_time_days": ["OS.time", "os_time_days", "os_time"],
        "age": ["age_at_initial_pathologic_diagnosis", "age"],
        "gender": ["gender", "sex"],
        "stage": ["ajcc_pathologic_tumor_stage", "clinical_stage", "stage"],
    }
    resolved: dict[str, str] = {}
    for key, options in aliases.items():
        found = next((item for item in options if item in frame.columns), None)
        if found is None:
            raise ClinicalPreprocessError(
                f"Required clinical column missing or renamed: {key}. "
                f"Observed columns: {frame.columns.tolist()}"
            )
        resolved[key] = found

    luad = frame.loc[frame[resolved["cancer"]].str.upper() == "LUAD"].copy()
    luad["patient_id"] = luad[resolved["patient_id"]].map(normalize_tcga_patient_id)
    luad = luad.dropna(subset=["patient_id"])
    if luad.empty:
        raise ClinicalPreprocessError("No TCGA-LUAD rows found in TCGA-CDR table.")
    aggregated = (
        luad.groupby("patient_id", as_index=False)
        .agg(
            {
                resolved["os_event"]: _first_present,
                resolved["os_time_days"]: _first_present,
                resolved["age"]: _first_present,
                resolved["gender"]: _first_present,
                resolved["stage"]: _first_present,
            }
        )
        .rename(
            columns={
                resolved["os_event"]: "os_event_raw",
                resolved["os_time_days"]: "os_time_days",
                resolved["age"]: "age",
                resolved["gender"]: "gender",
                resolved["stage"]: "stage_raw",
            }
        )
    )
    aggregated["os_event"] = aggregated["os_event_raw"].map(_status_to_int)
    aggregated["os_time_days"] = pd.to_numeric(aggregated["os_time_days"], errors="coerce")
    aggregated["age"] = pd.to_numeric(aggregated["age"], errors="coerce")
    aggregated["male"] = aggregated["gender"].str.strip().str.upper().map({"MALE": 1.0, "FEMALE": 0.0})
    aggregated["stage_numeric"] = aggregated["stage_raw"].map(stage_to_numeric)
    aggregated = aggregated.drop(columns=["os_event_raw"])
    return aggregated


@dataclass
class ClinicalTrainPreprocessor:
    """Median-impute and scale clinical covariates using training patients only."""

    columns: tuple[str, ...] = tuple(CLINICAL_COLUMNS)

    def fit(self, frame: pd.DataFrame, patient_ids: Iterable[str] | None = None) -> "ClinicalTrainPreprocessor":
        if frame.empty:
            raise ClinicalPreprocessError("Cannot fit clinical preprocessor on empty data.")
        numeric = frame[list(self.columns)].apply(pd.to_numeric, errors="coerce")
        medians = numeric.median(axis=0).fillna(0.0)
        imputed = numeric.fillna(medians)
        means = imputed.mean(axis=0)
        scales = imputed.std(axis=0, ddof=0).replace(0.0, 1.0)
        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        resolved_patient_ids = frame.index.astype(str).tolist() if patient_ids is None else patient_ids
        self.fit_patient_ids_ = tuple(resolved_patient_ids)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "medians_"):
            raise ClinicalPreprocessError("Clinical preprocessor must be fit first.")
        numeric = frame[list(self.columns)].apply(pd.to_numeric, errors="coerce")
        return ((numeric.fillna(self.medians_) - self.means_) / self.scales_).astype("float32")

    def fit_transform(
        self,
        frame: pd.DataFrame,
        patient_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return self.fit(frame, patient_ids=patient_ids).transform(frame)
