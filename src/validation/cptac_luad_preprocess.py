"""Stage 5B CPTAC-LUAD protein matrix preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from validation.cptac_data_import import CPTACPaths, select_inventory_file
from validation.cptac_validation import parse_protein_matrix


class CPTACPreprocessError(RuntimeError):
    """Raised when CPTAC-LUAD preprocessing cannot proceed."""


def candidate_table_path(paths: CPTACPaths, *, small_test: bool = False) -> Path:
    small = paths.root / "outputs" / "stage5_small_test" / "tables" / "candidate_genes.csv"
    formal = paths.root / "outputs" / "tables" / "stage5_candidate_genes.csv"
    if small_test and small.exists():
        return small
    return formal


def read_stage5_candidates(paths: CPTACPaths, *, small_test: bool = False) -> pd.DataFrame:
    path = candidate_table_path(paths, small_test=small_test)
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 5 candidate table missing: {path}. "
            "Run scripts/stage5_select_candidate_genes.py first."
        )
    frame = pd.read_csv(path)
    required = {"gene_symbol", "mechanism_layer", "expected_direction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CPTACPreprocessError(f"Stage 5 candidate table is missing columns: {missing}")
    frame["gene_symbol"] = frame["gene_symbol"].astype(str).str.upper()
    return frame


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def normalize_clinical_metadata(path: Path | None) -> pd.DataFrame:
    """Normalize a local CPTAC/PDC clinical table."""

    if path is None:
        return pd.DataFrame(columns=["sample_id", "risk_score", "risk_group", "os_time_days", "os_event", "age", "male", "stage_numeric"])
    frame = _read_table(path)
    lower = {str(column).strip().lower(): column for column in frame.columns}
    aliases = {
        "sample_id": ("sample_id", "patient_id", "case_id", "aliquot_id", "sample"),
        "risk_score": ("risk_score", "stage2d_risk_score", "rna_risk_score"),
        "risk_group": ("risk_group", "risk_strata", "group"),
        "os_time_days": ("os_time_days", "os_time", "survival_time", "days_to_death", "days_to_last_follow_up"),
        "os_event": ("os_event", "os_status", "vital_status", "death", "event"),
        "age": ("age", "age_at_diagnosis"),
        "male": ("male", "sex", "gender"),
        "stage_numeric": ("stage_numeric", "stage", "tumor_stage", "pathologic_stage"),
    }
    resolved = {
        field: next((lower[alias] for alias in options if alias in lower), None)
        for field, options in aliases.items()
    }
    if resolved["sample_id"] is None:
        raise CPTACPreprocessError(f"Clinical metadata lacks a sample ID column: {path}")
    result = pd.DataFrame({"sample_id": frame[resolved["sample_id"]].astype(str)})
    for field in ("risk_score", "risk_group", "os_time_days", "os_event", "age", "male", "stage_numeric"):
        source = resolved[field]
        result[field] = frame[source] if source is not None else np.nan
    result["risk_score"] = pd.to_numeric(result["risk_score"], errors="coerce")
    if result["risk_group"].notna().any():
        result["risk_group"] = result["risk_group"].astype(str).str.lower()
    elif result["risk_score"].notna().sum() > 2:
        cutoff = float(result["risk_score"].median())
        result["risk_group"] = np.where(result["risk_score"] >= cutoff, "high", "low")
    result["os_time_days"] = pd.to_numeric(result["os_time_days"], errors="coerce")
    result["os_event"] = result["os_event"].map(_event_to_int)
    result["age"] = pd.to_numeric(result["age"], errors="coerce")
    result["male"] = result["male"].map(_male_to_int)
    result["stage_numeric"] = result["stage_numeric"].map(_stage_to_numeric)
    return result.drop_duplicates("sample_id").reset_index(drop=True)


def _event_to_int(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "1.0", "dead", "deceased", "death", "true", "yes"}:
        return 1.0
    if text in {"0", "0.0", "alive", "living", "censored", "false", "no"}:
        return 0.0
    return np.nan


def _male_to_int(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "1.0", "male", "m"}:
        return 1.0
    if text in {"0", "0.0", "female", "f"}:
        return 0.0
    return np.nan


def _stage_to_numeric(value: object) -> float:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return np.nan
    for roman, number in (("iv", 4), ("iii", 3), ("ii", 2), ("i", 1)):
        if roman in text:
            return float(number)
    extracted = pd.Series([text]).str.extract(r"([1-4])")[0].iloc[0]
    return float(extracted) if pd.notna(extracted) else np.nan


def standardize_protein_matrix(protein: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate symbols and z-score proteins."""

    if "sample_id" not in protein.columns:
        raise CPTACPreprocessError("Protein matrix must contain sample_id after parsing.")
    sample = protein[["sample_id"]].astype(str).reset_index(drop=True)
    numeric = protein.drop(columns="sample_id").apply(pd.to_numeric, errors="coerce")
    numeric.columns = [str(column).upper().split(";")[0].strip() for column in numeric.columns]
    numeric = numeric.loc[:, [column != "" for column in numeric.columns]]
    numeric = numeric.T.groupby(level=0, sort=True).mean().T
    medians = numeric.median(axis=0).fillna(0.0)
    filled = numeric.fillna(medians)
    scale = filled.std(axis=0, ddof=0).replace(0.0, 1.0)
    zscore = (filled - filled.mean(axis=0)) / scale
    return pd.concat([sample, zscore.reset_index(drop=True)], axis=1)


def _empty_outputs(paths: CPTACPaths, candidates: pd.DataFrame) -> dict[str, object]:
    protein_out = paths.processed_dir / "cptac_luad_protein_matrix_processed.csv"
    clinical_out = paths.processed_dir / "cptac_luad_clinical_processed.csv"
    availability_out = paths.tables_dir / "stage5b_cptac_candidate_availability.csv"
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["sample_id"]).to_csv(protein_out, index=False)
    pd.DataFrame(columns=["sample_id", "risk_score", "risk_group", "os_time_days", "os_event", "age", "male", "stage_numeric"]).to_csv(clinical_out, index=False)
    availability = candidates[["gene_symbol", "mechanism_layer", "expected_direction"]].drop_duplicates().copy()
    availability["cptac_available"] = False
    availability["missing_fraction"] = np.nan
    availability["status"] = "unavailable_no_cptac_protein_matrix"
    availability.to_csv(availability_out, index=False)
    return {
        "status": "unavailable_no_cptac_protein_matrix",
        "protein_output": protein_out,
        "clinical_output": clinical_out,
        "availability_output": availability_out,
        "sample_count": 0,
        "protein_count": 0,
        "candidate_available": 0,
    }


def preprocess_cptac_luad(
    paths: CPTACPaths,
    inventory: pd.DataFrame,
    *,
    small_test: bool = False,
) -> dict[str, object]:
    """Preprocess local CPTAC-LUAD protein and clinical data."""

    candidates = read_stage5_candidates(paths, small_test=small_test)
    protein_path = select_inventory_file(inventory, "protein_abundance_matrix")
    if protein_path is None:
        return _empty_outputs(paths, candidates)
    clinical_path = select_inventory_file(inventory, "clinical_metadata")
    raw_protein = parse_protein_matrix(protein_path)
    protein = standardize_protein_matrix(raw_protein)
    clinical = normalize_clinical_metadata(clinical_path)
    if not clinical.empty:
        overlap = set(protein["sample_id"]) & set(clinical["sample_id"])
        if overlap:
            protein = protein.loc[protein["sample_id"].isin(overlap)].reset_index(drop=True)
            clinical = clinical.loc[clinical["sample_id"].isin(overlap)].reset_index(drop=True)
    protein_out = paths.processed_dir / "cptac_luad_protein_matrix_processed.csv"
    clinical_out = paths.processed_dir / "cptac_luad_clinical_processed.csv"
    availability_out = paths.tables_dir / "stage5b_cptac_candidate_availability.csv"
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    protein.to_csv(protein_out, index=False)
    clinical.to_csv(clinical_out, index=False)
    protein_genes = set(column for column in protein.columns if column != "sample_id")
    availability_rows = []
    for row in candidates.drop_duplicates("gene_symbol").itertuples(index=False):
        gene = str(row.gene_symbol).upper()
        if gene in protein_genes:
            missing_fraction = float(protein[gene].isna().mean())
            status = "available"
        else:
            missing_fraction = np.nan
            status = "unavailable_gene_not_in_cptac_matrix"
        availability_rows.append(
            {
                "gene_symbol": gene,
                "mechanism_layer": row.mechanism_layer,
                "expected_direction": row.expected_direction,
                "cptac_available": gene in protein_genes,
                "missing_fraction": missing_fraction,
                "status": status,
            }
        )
    availability = pd.DataFrame(availability_rows)
    availability.to_csv(availability_out, index=False)
    return {
        "status": "processed",
        "protein_output": protein_out,
        "clinical_output": clinical_out,
        "availability_output": availability_out,
        "sample_count": len(protein),
        "protein_count": len(protein.columns) - 1,
        "candidate_available": int(availability["cptac_available"].sum()),
    }
