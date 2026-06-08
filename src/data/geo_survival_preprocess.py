"""External GEO overall-survival preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.clinical_preprocess import stage_to_numeric
from data.geo_download import GEOCohortSpec


class GEOSurvivalPreprocessError(RuntimeError):
    """Raised when required external OS fields cannot be resolved."""


def _event_to_int(value: object) -> float:
    text = str(value or "").strip().lower()
    if text in {"1", "1.0", "dead", "deceased", "death", "true", "yes"}:
        return 1.0
    if text in {"0", "0.0", "alive", "living", "censored", "false", "no"}:
        return 0.0
    return np.nan


def _numeric(value: object) -> float:
    text = str(value or "").strip()
    if not text or text.casefold() in {"na", "n/a", "nan", "unknown", "--"}:
        return np.nan
    extracted = pd.Series([text]).str.extract(r"([-+]?\d*\.?\d+)")[0].iloc[0]
    return float(extracted) if pd.notna(extracted) else np.nan


def _male_to_int(value: object) -> float:
    text = str(value or "").strip().casefold()
    if text in {"male", "m", "1", "1.0"}:
        return 1.0
    if text in {"female", "f", "0", "0.0"}:
        return 0.0
    return np.nan


def prepare_geo_os_from_series_metadata(
    metadata: pd.DataFrame,
    spec: GEOCohortSpec,
) -> pd.DataFrame:
    """Derive OS and compatible clinical covariates from official GEO metadata."""

    required = {"sample_id", spec.os_time_field, spec.os_status_field}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise GEOSurvivalPreprocessError(
            f"{spec.accession} metadata is missing required OS fields: {missing}. "
            f"Observed columns: {metadata.columns.tolist()}"
        )
    result = metadata[["sample_id"]].copy()
    result["os_time_days"] = metadata[spec.os_time_field].map(_numeric)
    result["os_event"] = metadata[spec.os_status_field].map(_event_to_int)
    if spec.os_time_unit == "months":
        result["os_time_days"] *= 30.4375
    elif spec.os_time_unit == "years":
        result["os_time_days"] *= 365.25
    elif spec.os_time_unit != "days":
        raise ValueError(f"Unsupported GEO time unit: {spec.os_time_unit}")
    result["age"] = (
        metadata[spec.age_field].map(_numeric)
        if spec.age_field and spec.age_field in metadata
        else np.nan
    )
    result["male"] = (
        metadata[spec.sex_field].map(_male_to_int)
        if spec.sex_field and spec.sex_field in metadata
        else np.nan
    )
    result["stage_numeric"] = (
        metadata[spec.stage_field].map(stage_to_numeric)
        if spec.stage_field and spec.stage_field in metadata
        else np.nan
    )
    result = result.dropna(subset=["sample_id", "os_time_days", "os_event"])
    result = result.loc[result["os_time_days"] > 0].copy()
    result["os_event"] = result["os_event"].astype(int)
    if result.empty:
        raise GEOSurvivalPreprocessError(
            f"{spec.accession} has no usable OS rows after schema parsing."
        )
    if result["sample_id"].duplicated().any():
        raise GEOSurvivalPreprocessError(
            f"{spec.accession} survival metadata contains duplicate sample IDs."
        )
    return result.reset_index(drop=True)


def load_geo_os(path: str | Path, *, time_unit: str = "days") -> pd.DataFrame:
    """Load sample-level OS with explicit unit conversion and schema checks."""

    survival_path = Path(path)
    if not survival_path.exists():
        raise FileNotFoundError(f"GEO survival file not found: {survival_path}")
    separator = "\t" if survival_path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(survival_path, sep=separator, dtype=str)
    aliases = {
        "sample_id": ("sample_id", "geo_accession", "GSM", "patient_id"),
        "OS_time": ("OS_time", "os_time", "survival_time", "time", "OS.time"),
        "OS_status": ("OS_status", "os_status", "event", "status", "OS"),
    }
    resolved = {}
    for field, options in aliases.items():
        resolved[field] = next((column for column in options if column in frame.columns), None)
        if resolved[field] is None:
            raise GEOSurvivalPreprocessError(
                f"External survival table is missing {field}. Observed columns: {frame.columns.tolist()}"
            )
    result = frame[[resolved["sample_id"], resolved["OS_time"], resolved["OS_status"]]].rename(
        columns={
            resolved["sample_id"]: "sample_id",
            resolved["OS_time"]: "OS_time",
            resolved["OS_status"]: "OS_status",
        }
    )
    result["sample_id"] = result["sample_id"].astype(str).str.strip().str.strip('"')
    result["OS_time"] = pd.to_numeric(result["OS_time"], errors="coerce")
    result["OS_status"] = result["OS_status"].map(_event_to_int)
    if time_unit == "months":
        result["OS_time"] *= 30.4375
    elif time_unit == "years":
        result["OS_time"] *= 365.25
    elif time_unit != "days":
        raise ValueError("External OS time_unit must be days, months, or years.")
    result = result.dropna(subset=["sample_id", "OS_time", "OS_status"])
    result = result.loc[result["OS_time"] > 0].copy()
    result["OS_status"] = result["OS_status"].astype(int)
    if result.empty:
        raise GEOSurvivalPreprocessError("External cohort has no usable OS rows.")
    if result["sample_id"].duplicated().any():
        raise GEOSurvivalPreprocessError("External survival table contains duplicate sample IDs.")
    return result
