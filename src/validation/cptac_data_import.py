"""Stage 5B CPTAC/PDC LUAD data inventory and local import helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


class CPTACDataImportError(RuntimeError):
    """Raised when CPTAC/PDC input files cannot be inventoried."""


PDC_GRAPHQL_URL = "https://pdc.cancer.gov/graphql"


@dataclass(frozen=True)
class CPTACPaths:
    """Stage 5B path bundle."""

    root: Path
    raw_dir: Path
    processed_dir: Path
    tables_dir: Path
    reports_dir: Path
    figures_dir: Path


def stage5b_paths(root: str | Path = ".", *, small_test: bool = False) -> CPTACPaths:
    """Return Stage 5B paths, isolating small-test outputs."""

    project_root = Path(root).resolve()
    if small_test:
        base = project_root / "outputs" / "stage5b_small_test"
        return CPTACPaths(
            root=project_root,
            raw_dir=base / "raw" / "cptac_luad",
            processed_dir=base / "processed" / "cptac_luad",
            tables_dir=base / "tables",
            reports_dir=base / "reports",
            figures_dir=base / "figures",
        )
    return CPTACPaths(
        root=project_root,
        raw_dir=project_root / "data" / "raw" / "cptac_luad",
        processed_dir=project_root / "data" / "processed" / "cptac_luad",
        tables_dir=project_root / "outputs" / "tables",
        reports_dir=project_root / "outputs" / "reports",
        figures_dir=project_root / "outputs" / "figures",
    )


def ensure_stage5b_dirs(paths: CPTACPaths) -> None:
    for path in (
        paths.raw_dir,
        paths.processed_dir,
        paths.tables_dir,
        paths.reports_dir,
        paths.figures_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def md5sum(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file MD5 checksum."""

    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_cptac_file(path: str | Path) -> str:
    """Infer a CPTAC/PDC file role from its name."""

    name = Path(path).name.lower()
    if any(token in name for token in ("clinical", "survival", "followup", "follow_up")):
        return "clinical_metadata"
    if any(token in name for token in ("sample", "annotation", "aliquot", "mapping", "biospecimen")):
        return "sample_annotation"
    if any(token in name for token in ("protein", "proteome", "proteomic", "abundance", "itraq", "tmt")):
        return "protein_abundance_matrix"
    return "unclassified"


def _safe_table_shape(path: Path) -> tuple[int | None, int | None, str]:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv", ".txt"}:
        return None, None, "not_tabular_or_unsupported_extension"
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    try:
        frame = pd.read_csv(path, sep=separator, nrows=50)
    except Exception as exc:
        return None, None, f"read_failed: {exc}"
    rows = sum(1 for _ in path.open("rb")) - 1 if suffix in {".csv", ".tsv", ".txt"} else None
    return rows if rows is not None and rows >= 0 else None, int(frame.shape[1]), "readable_tabular"


def _file_luad_hint(path: Path) -> bool:
    text = path.name.lower()
    if any(token in text for token in ("luad", "lung", "adenocarcinoma")):
        return True
    if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        return False
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:20000].lower()
    except OSError:
        return False
    return any(token in sample for token in ("luad", "lung adenocarcinoma", "lung"))


def import_local_files(
    paths: CPTACPaths,
    *,
    protein_matrix: str | Path | None = None,
    clinical_metadata: str | Path | None = None,
    sample_annotation: str | Path | None = None,
) -> list[Path]:
    """Copy user-supplied local files into the Stage 5B raw directory."""

    ensure_stage5b_dirs(paths)
    copied: list[Path] = []
    for role, source in (
        ("protein_abundance_matrix", protein_matrix),
        ("clinical_metadata", clinical_metadata),
        ("sample_annotation", sample_annotation),
    ):
        if source is None:
            continue
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Stage 5B {role} file not found: {source_path}")
        destination = paths.raw_dir / source_path.name
        if destination.exists() and md5sum(destination) == md5sum(source_path):
            copied.append(destination)
            continue
        shutil.copy2(source_path, destination)
        copied.append(destination)
    return copied


def create_small_test_raw_files(paths: CPTACPaths) -> None:
    """Create isolated toy CPTAC-like files for smoke tests only."""

    ensure_stage5b_dirs(paths)
    rng = np.random.default_rng(20260603)
    genes = [
        "MKI67", "TOP2A", "PCNA", "MCM2", "CA9", "VEGFA", "LDHA", "SLC2A1",
        "VIM", "FN1", "COL1A1", "FAP", "CD74", "MS4A1", "MZB1", "JCHAIN",
    ]
    n = 48
    risk = rng.normal(size=n)
    protein = pd.DataFrame({"sample_id": [f"CPTAC_TOY_{i:03d}" for i in range(n)]})
    for gene in genes:
        direction = 0.55 if gene in {"MKI67", "TOP2A", "CA9", "VEGFA", "VIM", "FN1", "COL1A1", "FAP"} else -0.35
        protein[gene] = rng.normal(size=n) + direction * risk
    clinical = pd.DataFrame(
        {
            "sample_id": protein["sample_id"],
            "risk_score": risk,
            "risk_group": np.where(risk >= np.median(risk), "high", "low"),
            "os_time_days": np.maximum(30, 1400 - 180 * risk + rng.normal(0, 300, n)),
            "os_event": rng.binomial(1, 0.45, n),
            "age": rng.normal(66, 8, n),
            "male": rng.binomial(1, 0.5, n),
            "stage_numeric": rng.integers(1, 4, n),
        }
    )
    annotation = pd.DataFrame(
        {
            "sample_id": protein["sample_id"],
            "case_id": [f"CASE_TOY_{i:03d}" for i in range(n)],
            "tumor_type": "LUAD",
            "data_source": "toy_small_test_not_scientific",
        }
    )
    protein.to_csv(paths.raw_dir / "toy_cptac_luad_protein_abundance.csv", index=False)
    clinical.to_csv(paths.raw_dir / "toy_cptac_luad_clinical.csv", index=False)
    annotation.to_csv(paths.raw_dir / "toy_cptac_luad_sample_annotation.csv", index=False)


def build_cptac_inventory(paths: CPTACPaths) -> pd.DataFrame:
    """Inventory local CPTAC/PDC files with MD5 and role classification."""

    ensure_stage5b_dirs(paths)
    rows: list[dict[str, object]] = []
    for path in sorted(paths.raw_dir.glob("*")):
        if not path.is_file():
            continue
        role = classify_cptac_file(path)
        row_count, column_count, read_status = _safe_table_shape(path)
        rows.append(
            {
                "file_path": str(path),
                "file_name": path.name,
                "role": role,
                "bytes": path.stat().st_size,
                "md5": md5sum(path),
                "row_count": row_count,
                "column_count": column_count,
                "read_status": read_status,
                "luad_hint": _file_luad_hint(path),
            }
        )
    if not rows:
        rows.append(
            {
                "file_path": "",
                "file_name": "",
                "role": "missing_all_required_files",
                "bytes": 0,
                "md5": "",
                "row_count": None,
                "column_count": None,
                "read_status": "no_local_cptac_files_detected",
                "luad_hint": False,
            }
        )
    frame = pd.DataFrame(rows)
    required_roles = {"protein_abundance_matrix", "clinical_metadata", "sample_annotation"}
    present_roles = set(frame["role"])
    frame["required_roles_present"] = required_roles.issubset(present_roles)
    return frame


def query_pdc_luad_proteome_file_candidates(
    *,
    study_ids: list[str] | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Query PDC for LUAD proteome report files without downloading data."""

    studies = study_ids or _query_pdc_luad_proteome_studies(timeout=timeout)
    rows: list[dict[str, object]] = []
    for study_id in studies:
        query = (
            '{ filesPerStudy(pdc_study_id:"%s") { '
            "file_id file_name file_type data_category file_location md5sum "
            "} }"
        ) % study_id
        try:
            response = requests.post(PDC_GRAPHQL_URL, json={"query": query}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                raise CPTACDataImportError(str(payload["errors"][:2]))
            files = payload.get("data", {}).get("filesPerStudy", [])
        except Exception as exc:
            rows.append(
                {
                    "pdc_study_id": study_id,
                    "file_id": "",
                    "file_name": "",
                    "file_type": "",
                    "data_category": "",
                    "file_location": "",
                    "md5sum": "",
                    "stage5b_remote_role": "query_failed",
                    "is_stage5b_candidate": False,
                    "status": f"pdc_query_failed: {exc}",
                }
            )
            continue
        for file_record in files:
            role = _remote_file_role(file_record)
            if role == "not_stage5b_candidate":
                continue
            rows.append(
                {
                    "pdc_study_id": study_id,
                    "file_id": file_record.get("file_id", ""),
                    "file_name": file_record.get("file_name", ""),
                    "file_type": file_record.get("file_type", ""),
                    "data_category": file_record.get("data_category", ""),
                    "file_location": file_record.get("file_location", ""),
                    "md5sum": file_record.get("md5sum", ""),
                    "stage5b_remote_role": role,
                    "is_stage5b_candidate": role in {"protein_quantitation_candidate", "sample_annotation_candidate"},
                    "status": "remote_candidate_listed_not_downloaded",
                }
            )
    if not rows:
        rows.append(
            {
                "pdc_study_id": "",
                "file_id": "",
                "file_name": "",
                "file_type": "",
                "data_category": "",
                "file_location": "",
                "md5sum": "",
                "stage5b_remote_role": "no_remote_candidates_found",
                "is_stage5b_candidate": False,
                "status": "no_pdc_luad_proteome_candidates_found",
            }
        )
    return pd.DataFrame(rows)


def download_pdc_luad_study(
    paths: CPTACPaths,
    *,
    pdc_study_id: str = "PDC000153",
    data_type: str = "log2_ratio",
    timeout: int = 240,
    force: bool = False,
) -> dict[str, object]:
    """Download a PDC LUAD quantification matrix and linked metadata via GraphQL."""

    ensure_stage5b_dirs(paths)
    study_dir = paths.raw_dir / "pdc_downloads" / pdc_study_id
    manifest_dir = paths.raw_dir / "manifest"
    study_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw_quant_json": study_dir / f"{pdc_study_id}_quant_{data_type}.json",
        "raw_quant_tsv": study_dir / f"{pdc_study_id}_quant_{data_type}_gene_by_aliquot.tsv",
        "raw_clinical_json": study_dir / f"{pdc_study_id}_clinical_per_study.json",
        "raw_diagnoses_json": study_dir / f"{pdc_study_id}_case_diagnoses.json",
        "raw_mapping_json": study_dir / f"{pdc_study_id}_case_sample_aliquot_map.json",
        "protein": paths.raw_dir / f"{pdc_study_id}_cptac_luad_primary_tumor_protein_abundance.csv",
        "clinical": paths.raw_dir / f"{pdc_study_id}_cptac_luad_primary_tumor_clinical.csv",
        "annotation": paths.raw_dir / f"{pdc_study_id}_cptac_luad_sample_annotation.csv",
        "duplicates": manifest_dir / f"{pdc_study_id}_duplicate_aliquot_resolution.csv",
        "manifest": manifest_dir / f"{pdc_study_id}_download_manifest.csv",
    }
    required = ("protein", "clinical", "annotation")
    if not force and all(outputs[key].exists() and outputs[key].stat().st_size > 0 for key in required):
        return {
            "status": "already_downloaded",
            **outputs,
            "sample_count": _safe_csv_rows(outputs["protein"]),
            "gene_count": max(_safe_csv_columns(outputs["protein"]) - 1, 0),
        }

    quant_query = (
        '{ quantDataMatrix(pdc_study_id:"%s" data_type:"%s" acceptDUA:true) }'
        % (pdc_study_id, data_type)
    )
    clinical_query = (
        '{ clinicalPerStudy(pdc_study_id:"%s" acceptDUA:true) { '
        "case_id case_submitter_id status disease_type primary_site gender race ethnicity "
        "days_to_death vital_status age_at_index tumor_stage primary_diagnosis morphology "
        "follow_ups { days_to_follow_up days_to_progression days_to_progression_free "
        "days_to_recurrence progression_or_recurrence disease_response } } }"
        % pdc_study_id
    )
    diagnoses_query = (
        '{ paginatedCaseDiagnosesPerStudy(pdc_study_id:"%s" offset:0 limit:1000) { '
        "total caseDiagnosesPerStudy { case_id case_submitter_id disease_type primary_site "
        "diagnoses { age_at_diagnosis primary_diagnosis tumor_grade tumor_stage "
        "classification_of_tumor days_to_last_follow_up days_to_last_known_disease_status "
        "days_to_recurrence progression_or_recurrence morphology ajcc_pathologic_stage "
        "ajcc_pathologic_t ajcc_pathologic_n ajcc_pathologic_m } } } }"
        % pdc_study_id
    )
    mapping_query = (
        '{ paginatedCasesSamplesAliquots(pdc_study_id:"%s" offset:0 limit:1000) { '
        "total casesSamplesAliquots { case_id case_submitter_id disease_type primary_site "
        "samples { sample_id sample_submitter_id sample_type gdc_sample_id gdc_project_id "
        "tissue_type tumor_descriptor aliquots { aliquot_id aliquot_submitter_id analyte_type } "
        "} } } }"
        % pdc_study_id
    )
    quant_payload = _post_pdc_graphql(quant_query, timeout=timeout)
    clinical_payload = _post_pdc_graphql(clinical_query, timeout=timeout)
    diagnoses_payload = _post_pdc_graphql(diagnoses_query, timeout=timeout)
    mapping_payload = _post_pdc_graphql(mapping_query, timeout=timeout)
    _write_json(outputs["raw_quant_json"], quant_payload)
    _write_json(outputs["raw_clinical_json"], clinical_payload)
    _write_json(outputs["raw_diagnoses_json"], diagnoses_payload)
    _write_json(outputs["raw_mapping_json"], mapping_payload)

    matrix = quant_payload.get("data", {}).get("quantDataMatrix")
    if not matrix or len(matrix) < 2 or len(matrix[0]) < 2:
        raise CPTACDataImportError(f"PDC {pdc_study_id} quantDataMatrix returned no usable matrix.")
    mapping_cases = (
        mapping_payload.get("data", {})
        .get("paginatedCasesSamplesAliquots", {})
        .get("casesSamplesAliquots", [])
    )
    clinical_cases = clinical_payload.get("data", {}).get("clinicalPerStudy", [])
    diagnosis_cases = (
        diagnoses_payload.get("data", {})
        .get("paginatedCaseDiagnosesPerStudy", {})
        .get("caseDiagnosesPerStudy", [])
    )
    mapping = _flatten_pdc_mapping(mapping_cases)
    raw_header = [str(value) for value in matrix[0]]
    raw_aliquots = [value.split(":", 1)[1] if ":" in value else value for value in raw_header[1:]]
    raw_aliquot_set = set(raw_aliquots)
    mapping["in_quant_matrix"] = mapping["aliquot_submitter_id"].astype(str).isin(raw_aliquot_set)
    mapping.to_csv(outputs["annotation"], index=False)
    primary_mapping = mapping.loc[
        mapping["sample_type"].astype(str).str.lower().eq("primary tumor")
        & mapping["analyte_type"].astype(str).str.lower().eq("protein")
        & mapping["in_quant_matrix"]
    ].drop_duplicates("aliquot_submitter_id")

    primary_aliquots = set(primary_mapping["aliquot_submitter_id"].astype(str))
    selected_positions = [
        index for index, aliquot in enumerate(raw_aliquots, start=1)
        if aliquot in primary_aliquots
    ]
    if not selected_positions:
        raise CPTACDataImportError(
            f"PDC {pdc_study_id} matrix contains no Primary Tumor protein aliquots."
        )
    selected_header = ["gene_symbol", *[raw_aliquots[index - 1] for index in selected_positions]]
    gene_by_aliquot = pd.DataFrame(
        [[row[0], *[row[index] for index in selected_positions]] for row in matrix[1:]],
        columns=selected_header,
    )
    gene_by_aliquot["gene_symbol"] = gene_by_aliquot["gene_symbol"].astype(str).str.upper()
    values = gene_by_aliquot.drop(columns="gene_symbol").apply(pd.to_numeric, errors="coerce")
    gene_by_aliquot = pd.concat([gene_by_aliquot[["gene_symbol"]], values], axis=1)
    gene_by_aliquot = gene_by_aliquot.groupby("gene_symbol", sort=True, as_index=False).mean()
    gene_by_aliquot.to_csv(outputs["raw_quant_tsv"], sep="\t", index=False)
    protein = gene_by_aliquot.set_index("gene_symbol").T.reset_index(names="sample_id")
    duplicate_counts = protein["sample_id"].value_counts()
    duplicate_rows = [
        {
            "sample_id": sample_id,
            "matrix_column_count": int(count),
            "resolution": "mean_across_duplicate_quantitation_columns",
        }
        for sample_id, count in duplicate_counts.items()
        if count > 1
    ]
    pd.DataFrame(
        duplicate_rows,
        columns=["sample_id", "matrix_column_count", "resolution"],
    ).to_csv(outputs["duplicates"], index=False)
    if protein["sample_id"].duplicated().any():
        protein = protein.groupby("sample_id", sort=True, as_index=False).mean(numeric_only=True)
    protein.to_csv(outputs["protein"], index=False)

    clinical = _build_pdc_primary_tumor_clinical(
        primary_mapping,
        clinical_cases,
        diagnosis_cases,
    )
    clinical.to_csv(outputs["clinical"], index=False)
    manifest = _build_download_manifest(outputs, pdc_study_id=pdc_study_id, data_type=data_type)
    manifest.to_csv(outputs["manifest"], index=False)
    return {
        "status": "downloaded",
        **outputs,
        "sample_count": len(protein),
        "gene_count": len(protein.columns) - 1,
        "clinical_count": len(clinical),
        "usable_os_count": int(clinical[["os_time_days", "os_event"]].notna().all(axis=1).sum()),
    }


def _post_pdc_graphql(query: str, *, timeout: int) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(PDC_GRAPHQL_URL, json={"query": query}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                raise CPTACDataImportError(str(payload["errors"][:2]))
            return payload
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
    raise CPTACDataImportError(f"PDC GraphQL request failed after 3 attempts: {last_error}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _flatten_pdc_mapping(cases: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in cases:
        for sample in case.get("samples") or []:
            for aliquot in sample.get("aliquots") or []:
                rows.append(
                    {
                        "case_id": case.get("case_id"),
                        "case_submitter_id": case.get("case_submitter_id"),
                        "disease_type": case.get("disease_type"),
                        "primary_site": case.get("primary_site"),
                        "sample_id": sample.get("sample_id"),
                        "sample_submitter_id": sample.get("sample_submitter_id"),
                        "sample_type": sample.get("sample_type"),
                        "gdc_sample_id": sample.get("gdc_sample_id"),
                        "gdc_project_id": sample.get("gdc_project_id"),
                        "tissue_type": sample.get("tissue_type"),
                        "tumor_descriptor": sample.get("tumor_descriptor"),
                        "aliquot_id": aliquot.get("aliquot_id"),
                        "aliquot_submitter_id": aliquot.get("aliquot_submitter_id"),
                        "analyte_type": aliquot.get("analyte_type"),
                    }
                )
    if not rows:
        raise CPTACDataImportError("PDC case/sample/aliquot mapping returned no rows.")
    return pd.DataFrame(rows)


def _build_pdc_primary_tumor_clinical(
    primary_mapping: pd.DataFrame,
    clinical_cases: list[dict[str, object]],
    diagnosis_cases: list[dict[str, object]],
) -> pd.DataFrame:
    clinical_by_case = {str(row.get("case_id")): row for row in clinical_cases}
    diagnoses_by_case = {
        str(row.get("case_id")): (row.get("diagnoses") or [{}])[0]
        for row in diagnosis_cases
    }
    rows: list[dict[str, object]] = []
    for mapping_row in primary_mapping.itertuples(index=False):
        clinical = clinical_by_case.get(str(mapping_row.case_id), {})
        diagnosis = diagnoses_by_case.get(str(mapping_row.case_id), {})
        vital = str(clinical.get("vital_status") or "").strip().lower()
        os_event = 1.0 if vital in {"dead", "deceased"} else (0.0 if vital in {"alive", "living"} else np.nan)
        days_to_death = pd.to_numeric(pd.Series([clinical.get("days_to_death")]), errors="coerce").iloc[0]
        days_to_follow_up = pd.to_numeric(
            pd.Series([diagnosis.get("days_to_last_follow_up")]), errors="coerce"
        ).iloc[0]
        os_time = days_to_death if os_event == 1.0 else days_to_follow_up
        age_days = pd.to_numeric(
            pd.Series([diagnosis.get("age_at_diagnosis")]), errors="coerce"
        ).iloc[0]
        gender = str(clinical.get("gender") or "").strip().lower()
        stage = diagnosis.get("ajcc_pathologic_stage") or diagnosis.get("tumor_stage") or clinical.get("tumor_stage")
        rows.append(
            {
                "sample_id": mapping_row.aliquot_submitter_id,
                "case_id": mapping_row.case_id,
                "patient_id": mapping_row.case_submitter_id,
                "sample_submitter_id": mapping_row.sample_submitter_id,
                "sample_type": mapping_row.sample_type,
                "risk_score": np.nan,
                "risk_group": np.nan,
                "os_time_days": os_time,
                "os_event": os_event,
                "age": float(age_days / 365.25) if pd.notna(age_days) else np.nan,
                "male": 1.0 if gender == "male" else (0.0 if gender == "female" else np.nan),
                "stage_numeric": _stage_numeric(stage),
                "pathologic_stage": stage,
                "vital_status": clinical.get("vital_status"),
                "days_to_death": days_to_death,
                "days_to_last_follow_up": days_to_follow_up,
                "primary_diagnosis": diagnosis.get("primary_diagnosis") or clinical.get("primary_diagnosis"),
                "tumor_grade": diagnosis.get("tumor_grade"),
                "race": clinical.get("race"),
                "ethnicity": clinical.get("ethnicity"),
                "pdc_source": "PDC GraphQL API",
            }
        )
    return pd.DataFrame(rows).drop_duplicates("sample_id").reset_index(drop=True)


def _stage_numeric(value: object) -> float:
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "not reported"}:
        return np.nan
    for roman, number in (("iv", 4), ("iii", 3), ("ii", 2), ("i", 1)):
        if roman in text:
            return float(number)
    return np.nan


def _build_download_manifest(
    outputs: dict[str, Path],
    *,
    pdc_study_id: str,
    data_type: str,
) -> pd.DataFrame:
    rows = []
    for role, path in outputs.items():
        if role == "manifest" or not path.exists():
            continue
        rows.append(
            {
                "pdc_study_id": pdc_study_id,
                "data_type": data_type,
                "role": role,
                "file_path": str(path),
                "bytes": path.stat().st_size,
                "md5": md5sum(path),
                "source": PDC_GRAPHQL_URL,
                "status": "downloaded_and_verified_local_file",
            }
        )
    return pd.DataFrame(rows)


def _safe_csv_rows(path: Path) -> int:
    try:
        return max(sum(1 for _ in path.open("rb")) - 1, 0)
    except OSError:
        return 0


def _safe_csv_columns(path: Path) -> int:
    try:
        return int(pd.read_csv(path, nrows=1).shape[1])
    except Exception:
        return 0


def _query_pdc_luad_proteome_studies(*, timeout: int) -> list[str]:
    query = (
        "{ studyCatalog { pdc_study_id versions { "
        "study_submitter_id submitter_id_name study_shortname is_latest_version "
        "} } }"
    )
    try:
        response = requests.post(PDC_GRAPHQL_URL, json={"query": query}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise CPTACDataImportError(str(payload["errors"][:2]))
    except Exception as exc:
        raise CPTACDataImportError(f"PDC LUAD study catalog query failed: {exc}") from exc
    studies: list[str] = []
    for entry in payload.get("data", {}).get("studyCatalog", []):
        versions = entry.get("versions") or []
        latest = next((version for version in versions if version.get("is_latest_version") == "yes"), versions[0] if versions else {})
        text = " ".join(str(latest.get(field, "")) for field in ("study_submitter_id", "submitter_id_name", "study_shortname")).lower()
        is_luad = "luad" in text or "lung adeno" in text
        is_proteome = "proteome" in text
        is_excluded = any(token in text for token in ("phosphoproteome", "acetylome", "ubiquitylome", "compre"))
        if is_luad and is_proteome and not is_excluded:
            studies.append(str(entry.get("pdc_study_id")))
    return sorted(set(study for study in studies if study))


def _remote_file_role(file_record: dict[str, object]) -> str:
    name = str(file_record.get("file_name", "")).lower()
    location = str(file_record.get("file_location", "")).lower()
    category = str(file_record.get("data_category", "")).lower()
    file_type = str(file_record.get("file_type", "")).lower()
    text = " ".join((name, location, category, file_type))
    if "protein assembly" not in category:
        return "not_stage5b_candidate"
    if name.endswith(".sample.txt") or "sample.txt" in name:
        return "sample_annotation_candidate"
    if name.endswith(".label.txt") or "label.txt" in name:
        return "label_reagent_metadata"
    if "peptides" in name or "peptide." in name:
        return "peptide_level_not_primary_stage5b"
    if name.endswith(".summary.tsv") or "summary.tsv" in name:
        return "protein_summary_candidate_not_abundance"
    if any(token in text for token in ("tmt10.tsv", "tmt11.tsv", "abundance", "quant")):
        return "protein_quantitation_candidate"
    return "not_stage5b_candidate"


def select_inventory_file(inventory: pd.DataFrame, role: str) -> Path | None:
    """Select the first readable file matching a role."""

    subset = inventory.loc[
        (inventory["role"] == role)
        & inventory["file_path"].astype(str).ne("")
        & inventory["read_status"].astype(str).str.startswith("readable")
    ].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["luad_hint", "bytes"], ascending=[False, False])
    return Path(str(subset.iloc[0]["file_path"]))


def write_inventory_report(
    inventory: pd.DataFrame,
    output_path: str | Path,
    *,
    small_test: bool,
    remote_candidates: pd.DataFrame | None = None,
) -> None:
    """Write a Stage 5B inventory report."""

    required = {"protein_abundance_matrix", "clinical_metadata", "sample_annotation"}
    present = set(inventory["role"])
    missing = sorted(required - present)
    remote_count = 0 if remote_candidates is None else int(remote_candidates["is_stage5b_candidate"].fillna(False).sum())
    remote_studies = [] if remote_candidates is None else sorted(set(remote_candidates["pdc_study_id"].dropna().astype(str)) - {""})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Stage 5B CPTAC/PDC Data Inventory Report\n\n"
        f"- Mode: {'toy small-test' if small_test else 'formal'}.\n"
        f"- Local files inventoried: {int((inventory['file_path'].astype(str) != '').sum())}.\n"
        f"- Required roles present: {required.issubset(present)}.\n"
        f"- Missing roles: {', '.join(missing) if missing else 'none'}.\n"
        f"- PDC remote candidate files listed: {remote_count}.\n"
        f"- PDC LUAD proteome studies queried: {', '.join(remote_studies) if remote_studies else 'none or skipped'}.\n"
        "- If required files are missing, downstream formal Stage 5B outputs must remain unavailable/manual-download-required.\n\n"
        "## Integrity Boundary\n\n"
        "- No toy CPTAC data are used for formal analysis.\n"
        "- Missing CPTAC data are not interpreted as negative evidence.\n",
        encoding="utf-8",
    )
