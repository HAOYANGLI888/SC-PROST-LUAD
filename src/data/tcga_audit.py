"""TCGA-LUAD public metadata audit for Stage 1.

This module intentionally downloads metadata only. Large biomedical data files
are deferred to later stages and documented in the manual download guide.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import io
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

GDC_API = "https://api.gdc.cancer.gov"
GDC_PROJECT = "TCGA-LUAD"
TCGA_CDR_URL = (
    "https://pancanatlas.xenahubs.net/download/"
    "Survival_SupplementalTable_S1_20171025_xena_sp"
)
USER_AGENT = "SC-PROST-LUAD-stage1/0.1"
TCGA_PATIENT_PATTERN = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)
PRIMARY_TUMOR_TYPES = {
    "primary tumor",
    "recurrent tumor",
    "metastatic",
    "additional - new primary",
}
ENDPOINTS = ("OS", "DSS", "PFI")

MODALITY_SPECS: dict[str, dict[str, Any]] = {
    "rnaseq": {
        "label": "RNA-seq gene expression",
        "filters": {
            "data_category": "Transcriptome Profiling",
            "data_type": "Gene Expression Quantification",
            "experimental_strategy": "RNA-Seq",
        },
        "tumor_sample_required": True,
    },
    "mutation_maf": {
        "label": "Mutation MAF",
        "filters": {
            "data_category": "Simple Nucleotide Variation",
            "data_type": "Masked Somatic Mutation",
            "data_format": "MAF",
        },
        "tumor_sample_required": True,
    },
    "cnv": {
        "label": "Gene-level CNV",
        "filters": {
            "data_category": "Copy Number Variation",
            "data_type": "Gene Level Copy Number",
        },
        "tumor_sample_required": True,
    },
    "methylation": {
        "label": "Methylation beta value",
        "filters": {
            "data_category": "DNA Methylation",
            "data_type": "Methylation Beta Value",
        },
        "tumor_sample_required": True,
    },
    "diagnostic_wsi": {
        "label": "Diagnostic slide / WSI",
        "filters": {
            "data_type": "Slide Image",
            "experimental_strategy": "Diagnostic Slide",
        },
        "tumor_sample_required": True,
    },
}

PATIENT_MATRIX_COLUMNS = [
    "patient_id",
    "case_id",
    "clinical_available",
    "os_available",
    "dss_available",
    "pfi_available",
    "rnaseq_available",
    "mutation_maf_available",
    "cnv_available",
    "methylation_available",
    "diagnostic_wsi_available",
    "rnaseq_file_count",
    "mutation_maf_file_count",
    "cnv_file_count",
    "methylation_file_count",
    "diagnostic_wsi_file_count",
    "eligible_clinical_rnaseq",
    "eligible_clinical_rnaseq_wsi",
    "eligible_clinical_rnaseq_mutation_methylation",
    "eligible_complete_multimodal",
    "eligible_model_sets",
]

SURVIVAL_ALIASES: dict[str, tuple[str, ...]] = {
    "patient_id": ("_PATIENT", "patient", "patient_id", "bcr_patient_barcode"),
    "cancer": ("cancer type abbreviation", "cancer_type", "type"),
    "OS": ("OS", "os"),
    "OS.time": ("OS.time", "os_time", "OS_time"),
    "DSS": ("DSS", "dss"),
    "DSS.time": ("DSS.time", "dss_time", "DSS_time"),
    "PFI": ("PFI", "pfi"),
    "PFI.time": ("PFI.time", "pfi_time", "PFI_time"),
}


class AuditError(RuntimeError):
    """Raised when metadata auditing cannot continue safely."""


class NetworkAuditError(AuditError):
    """Raised after a public metadata endpoint repeatedly fails."""


class SchemaAuditError(AuditError):
    """Raised when an upstream table changes incompatibly."""


@dataclass(frozen=True)
class AuditPaths:
    """Stage 1 output and cache paths."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "AuditPaths":
        return cls(Path(root).resolve())

    @property
    def metadata(self) -> Path:
        return self.root / "data" / "metadata"

    @property
    def reports(self) -> Path:
        return self.root / "outputs" / "reports"

    @property
    def audit(self) -> Path:
        return self.root / "outputs" / "audit" / "stage1"

    @property
    def logs(self) -> Path:
        return self.root / "outputs" / "logs"

    @property
    def survival_cache(self) -> Path:
        return (
            self.root
            / "data"
            / "raw"
            / "tcga_luad"
            / "clinical"
            / "Survival_SupplementalTable_S1_20171025_xena_sp.tsv"
        )

    @property
    def patient_matrix(self) -> Path:
        return self.metadata / "tcga_luad_patient_modality_matrix.csv"

    @property
    def survival_summary(self) -> Path:
        return self.metadata / "tcga_luad_survival_summary.csv"

    @property
    def modality_summary(self) -> Path:
        return self.metadata / "tcga_luad_available_modalities_summary.csv"

    @property
    def api_snapshot(self) -> Path:
        return self.metadata / "tcga_luad_stage1_api_snapshot.json"

    @property
    def report(self) -> Path:
        return self.reports / "stage1_tcga_luad_audit_report.md"

    @property
    def audit_mirror(self) -> Path:
        return self.audit / "audit_report.md"

    @property
    def log(self) -> Path:
        return self.logs / "stage1_tcga_luad_audit.json"

    @property
    def dry_run_log(self) -> Path:
        return self.logs / "stage1_tcga_luad_dry_run.json"

    @property
    def small_test_log(self) -> Path:
        return self.logs / "stage1_tcga_luad_small_test.json"

    @property
    def pytest_log(self) -> Path:
        return self.logs / "stage1_pytest.txt"

    @property
    def dry_run_report(self) -> Path:
        return self.reports / "stage1_tcga_luad_dry_run_report.md"

    def ensure_dirs(self) -> None:
        for directory in (
            self.metadata,
            self.reports,
            self.audit,
            self.logs,
            self.survival_cache.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def extract_patient_id(value: Any) -> str | None:
    """Extract a TCGA patient barcode from a case or sample identifier."""

    text = str(value or "").strip().upper()
    match = TCGA_PATIENT_PATTERN.match(text)
    return match.group(0) if match else None


def _is_present(value: Any) -> bool:
    return str(value or "").strip().lower() not in {
        "",
        "nan",
        "na",
        "n/a",
        "not reported",
        "unknown",
        "[not available]",
    }


def _patient_has_tumor_sample(hit: dict[str, Any]) -> bool:
    samples = [
        sample
        for case in hit.get("cases", [])
        for sample in case.get("samples", [])
    ]
    if not samples:
        return False
    return any(
        str(sample.get("sample_type", "")).strip().lower() in PRIMARY_TUMOR_TYPES
        for sample in samples
    )


def _request_bytes(
    url: str,
    *,
    timeout: int = 60,
    retries: int = 3,
) -> bytes:
    """GET a public URL with retry handling and an actionable error."""

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(attempt)
    raise NetworkAuditError(
        f"Network request failed after {retries} attempts: {url}. "
        f"Last error: {last_error}"
    )


def _request_json(
    endpoint: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: int = 60,
    retries: int = 3,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(params or {})
    url = f"{endpoint}?{query}" if query else endpoint
    payload = _request_bytes(url, timeout=timeout, retries=retries)
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaAuditError(f"Endpoint returned invalid JSON: {url}") from exc
    if not isinstance(data, dict):
        raise SchemaAuditError(f"Expected a JSON object from endpoint: {url}")
    return data


def _build_filter(clauses: Iterable[tuple[str, str]]) -> dict[str, Any]:
    return {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": field, "value": value}}
            for field, value in clauses
        ],
    }


def _fetch_paginated_hits(
    endpoint: str,
    *,
    filters: dict[str, Any],
    fields: str,
    page_size: int = 2000,
    timeout: int = 60,
    retries: int = 3,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        response = _request_json(
            endpoint,
            {
                "filters": json.dumps(filters, separators=(",", ":")),
                "fields": fields,
                "format": "JSON",
                "from": offset,
                "size": page_size,
            },
            timeout=timeout,
            retries=retries,
        )
        try:
            data = response["data"]
            page_hits = data["hits"]
            total = int(data["pagination"]["total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaAuditError(
                f"Unexpected paginated response schema from {endpoint}"
            ) from exc
        if not isinstance(page_hits, list):
            raise SchemaAuditError(f"Expected list of hits from {endpoint}")
        hits.extend(page_hits)
        if not page_hits:
            break
        offset += len(page_hits)
    return hits


def fetch_gdc_status(*, timeout: int, retries: int) -> dict[str, Any]:
    """Fetch GDC release metadata for reproducibility."""

    return _request_json(f"{GDC_API}/status", timeout=timeout, retries=retries)


def fetch_gdc_cases(*, timeout: int, retries: int) -> list[dict[str, Any]]:
    """Fetch TCGA-LUAD case-level clinical metadata."""

    return _fetch_paginated_hits(
        f"{GDC_API}/cases",
        filters=_build_filter([("project.project_id", GDC_PROJECT)]),
        fields=(
            "case_id,submitter_id,project.project_id,"
            "demographic.vital_status,demographic.days_to_death,"
            "diagnoses.days_to_last_follow_up,diagnoses.primary_diagnosis"
        ),
        page_size=1000,
        timeout=timeout,
        retries=retries,
    )


def fetch_gdc_modality_files(
    modality: str,
    *,
    timeout: int,
    retries: int,
) -> list[dict[str, Any]]:
    """Fetch open GDC file metadata for one modality."""

    if modality not in MODALITY_SPECS:
        raise KeyError(f"Unsupported modality: {modality}")
    spec = MODALITY_SPECS[modality]
    clauses = [
        ("cases.project.project_id", GDC_PROJECT),
        ("access", "open"),
        *spec["filters"].items(),
    ]
    return _fetch_paginated_hits(
        f"{GDC_API}/files",
        filters=_build_filter(clauses),
        fields=(
            "file_id,file_name,data_category,data_type,data_format,"
            "experimental_strategy,access,cases.case_id,cases.submitter_id,"
            "cases.samples.sample_type,cases.samples.submitter_id"
        ),
        page_size=2000,
        timeout=timeout,
        retries=retries,
    )


def fetch_survival_table(
    cache_path: Path,
    *,
    refresh: bool,
    timeout: int,
    retries: int,
) -> tuple[str, str]:
    """Read a cached TCGA-CDR table or download it once from UCSC Xena."""

    if cache_path.exists() and not refresh:
        text = cache_path.read_text(encoding="utf-8")
        if text.strip():
            return text, "cached"
        raise AuditError(f"Cached survival table is empty: {cache_path}")

    payload = _request_bytes(TCGA_CDR_URL, timeout=timeout, retries=retries)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaAuditError("TCGA-CDR survival table is not UTF-8 text.") from exc
    if not text.strip():
        raise AuditError("Downloaded TCGA-CDR survival table is empty.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text, "downloaded"


def _resolve_survival_columns(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise SchemaAuditError("TCGA-CDR survival table has no header.")
    resolved: dict[str, str] = {}
    for canonical, aliases in SURVIVAL_ALIASES.items():
        match = next((alias for alias in aliases if alias in fieldnames), None)
        if match is None:
            raise SchemaAuditError(
                f"TCGA-CDR column missing or renamed: {canonical}. "
                f"Observed columns: {fieldnames}"
            )
        resolved[canonical] = match
    return resolved


def parse_survival_table(text: str) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Parse LUAD rows from TCGA-CDR while detecting schema and ID issues."""

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    columns = _resolve_survival_columns(reader.fieldnames)
    patients: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    invalid_ids = 0
    duplicate_ids = 0
    for row in reader:
        if str(row.get(columns["cancer"], "")).strip().upper() != "LUAD":
            continue
        patient_id = extract_patient_id(row.get(columns["patient_id"]))
        if patient_id is None:
            invalid_ids += 1
            continue
        parsed = {
            endpoint: str(row.get(columns[endpoint], "") or "").strip()
            for endpoint in ENDPOINTS
        }
        parsed.update(
            {
                f"{endpoint}.time": str(
                    row.get(columns[f"{endpoint}.time"], "") or ""
                ).strip()
                for endpoint in ENDPOINTS
            }
        )
        if patient_id in patients:
            duplicate_ids += 1
            for key, value in parsed.items():
                if not _is_present(patients[patient_id].get(key)) and _is_present(value):
                    patients[patient_id][key] = value
        else:
            patients[patient_id] = parsed
    if not patients:
        raise AuditError("TCGA-CDR parsing returned no LUAD patients.")
    if invalid_ids:
        warnings.append(f"Skipped {invalid_ids} TCGA-CDR LUAD rows with invalid patient IDs.")
    if duplicate_ids:
        warnings.append(
            f"Merged {duplicate_ids} duplicate TCGA-CDR LUAD rows at patient level."
        )
    return patients, warnings


def _case_index(
    cases: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not cases:
        raise AuditError("GDC case query returned no TCGA-LUAD cases.")
    indexed: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    invalid_ids = 0
    for case in cases:
        patient_id = extract_patient_id(case.get("submitter_id"))
        if patient_id is None:
            invalid_ids += 1
            continue
        indexed[patient_id] = case
    if not indexed:
        raise AuditError("No GDC case IDs could be mapped to TCGA patient barcodes.")
    if invalid_ids:
        warnings.append(f"Skipped {invalid_ids} GDC cases with invalid patient IDs.")
    return indexed, warnings


def _modality_patient_counts(
    files: list[dict[str, Any]],
    *,
    tumor_sample_required: bool,
) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = defaultdict(int)
    warnings: list[str] = []
    invalid_case_ids = 0
    files_without_tumor_sample = 0
    for hit in files:
        if tumor_sample_required and not _patient_has_tumor_sample(hit):
            files_without_tumor_sample += 1
            continue
        patient_ids = {
            patient_id
            for case in hit.get("cases", [])
            if (patient_id := extract_patient_id(case.get("submitter_id"))) is not None
        }
        if not patient_ids:
            invalid_case_ids += 1
            continue
        for patient_id in patient_ids:
            counts[patient_id] += 1
    if invalid_case_ids:
        warnings.append(f"Skipped {invalid_case_ids} files with unmappable case IDs.")
    if files_without_tumor_sample:
        warnings.append(
            f"Excluded {files_without_tumor_sample} files without a linked tumor sample."
        )
    return dict(counts), warnings


def _endpoint_available(survival: dict[str, str], endpoint: str) -> bool:
    return _is_present(survival.get(endpoint)) and _is_present(
        survival.get(f"{endpoint}.time")
    )


def build_patient_matrix(
    cases: list[dict[str, Any]],
    survival: dict[str, dict[str, str]],
    modality_files: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], list[str]]:
    """Build a patient-level matrix and retain file-level audit statistics."""

    case_by_patient, warnings = _case_index(cases)
    modality_counts: dict[str, dict[str, int]] = {}
    for modality, spec in MODALITY_SPECS.items():
        counts, modality_warnings = _modality_patient_counts(
            modality_files.get(modality, []),
            tumor_sample_required=bool(spec["tumor_sample_required"]),
        )
        modality_counts[modality] = counts
        warnings.extend(f"{modality}: {item}" for item in modality_warnings)

    unmatched_survival = sorted(set(survival) - set(case_by_patient))
    if unmatched_survival:
        warnings.append(
            f"{len(unmatched_survival)} TCGA-CDR LUAD patient IDs did not match GDC cases."
        )

    rows: list[dict[str, Any]] = []
    for patient_id in sorted(case_by_patient):
        patient_survival = survival.get(patient_id, {})
        row: dict[str, Any] = {
            "patient_id": patient_id,
            "case_id": case_by_patient[patient_id].get("case_id", ""),
            "clinical_available": True,
            "os_available": _endpoint_available(patient_survival, "OS"),
            "dss_available": _endpoint_available(patient_survival, "DSS"),
            "pfi_available": _endpoint_available(patient_survival, "PFI"),
        }
        for modality in MODALITY_SPECS:
            count = modality_counts[modality].get(patient_id, 0)
            row[f"{modality}_available"] = count > 0
            row[f"{modality}_file_count"] = count

        row["eligible_clinical_rnaseq"] = (
            row["clinical_available"] and row["os_available"] and row["rnaseq_available"]
        )
        row["eligible_clinical_rnaseq_wsi"] = (
            row["eligible_clinical_rnaseq"] and row["diagnostic_wsi_available"]
        )
        row["eligible_clinical_rnaseq_mutation_methylation"] = (
            row["eligible_clinical_rnaseq"]
            and row["mutation_maf_available"]
            and row["methylation_available"]
        )
        row["eligible_complete_multimodal"] = (
            row["eligible_clinical_rnaseq_mutation_methylation"]
            and row["cnv_available"]
            and row["diagnostic_wsi_available"]
        )
        eligible_sets = [
            name
            for name in (
                "clinical_rnaseq",
                "clinical_rnaseq_wsi",
                "clinical_rnaseq_mutation_methylation",
                "complete_multimodal",
            )
            if row[f"eligible_{name}"]
        ]
        row["eligible_model_sets"] = ";".join(eligible_sets)
        rows.append(row)
    if not rows:
        raise AuditError("Patient modality matrix is empty.")
    return rows, modality_counts, warnings


def build_survival_summary(
    patient_rows: list[dict[str, Any]],
    survival: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Summarize missing status/time and usable patient counts per endpoint."""

    patient_ids = [row["patient_id"] for row in patient_rows]
    rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        status_present = sum(
            _is_present(survival.get(patient_id, {}).get(endpoint))
            for patient_id in patient_ids
        )
        time_present = sum(
            _is_present(survival.get(patient_id, {}).get(f"{endpoint}.time"))
            for patient_id in patient_ids
        )
        usable = sum(
            _endpoint_available(survival.get(patient_id, {}), endpoint)
            for patient_id in patient_ids
        )
        rows.append(
            {
                "endpoint": endpoint,
                "gdc_patient_count": len(patient_ids),
                "status_present_count": status_present,
                "time_present_count": time_present,
                "usable_count": usable,
                "missing_status_count": len(patient_ids) - status_present,
                "missing_time_count": len(patient_ids) - time_present,
                "notes": "Usable requires both endpoint status and endpoint time.",
            }
        )
    return rows


def _count_true(rows: list[dict[str, Any]], column: str) -> int:
    return sum(_as_bool(row.get(column)) for row in rows)


def build_modality_summary(
    patient_rows: list[dict[str, Any]],
    modality_files: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Summarize individual modalities and model-ready patient intersections."""

    summary = [
        {
            "record_type": "modality",
            "item": "clinical",
            "patient_count": _count_true(patient_rows, "clinical_available"),
            "open_file_metadata_count": "",
            "definition": "GDC TCGA-LUAD case-level clinical metadata",
        },
        {
            "record_type": "survival_endpoint",
            "item": "OS",
            "patient_count": _count_true(patient_rows, "os_available"),
            "open_file_metadata_count": "",
            "definition": "TCGA-CDR endpoint status and time both present",
        },
        {
            "record_type": "survival_endpoint",
            "item": "DSS",
            "patient_count": _count_true(patient_rows, "dss_available"),
            "open_file_metadata_count": "",
            "definition": "TCGA-CDR endpoint status and time both present",
        },
        {
            "record_type": "survival_endpoint",
            "item": "PFI",
            "patient_count": _count_true(patient_rows, "pfi_available"),
            "open_file_metadata_count": "",
            "definition": "TCGA-CDR endpoint status and time both present",
        },
    ]
    for modality, spec in MODALITY_SPECS.items():
        summary.append(
            {
                "record_type": "modality",
                "item": modality,
                "patient_count": _count_true(patient_rows, f"{modality}_available"),
                "open_file_metadata_count": len(modality_files.get(modality, [])),
                "definition": f"{spec['label']}; linked tumor samples only",
            }
        )
    combinations = {
        "clinical_rnaseq": "Clinical + OS + RNA-seq",
        "clinical_rnaseq_wsi": "Clinical + OS + RNA-seq + diagnostic WSI",
        "clinical_rnaseq_mutation_methylation": (
            "Clinical + OS + RNA-seq + mutation MAF + methylation"
        ),
        "complete_multimodal": (
            "Clinical + OS + RNA-seq + mutation MAF + CNV + methylation + diagnostic WSI"
        ),
    }
    for item, definition in combinations.items():
        summary.append(
            {
                "record_type": "model_cohort",
                "item": item,
                "patient_count": _count_true(patient_rows, f"eligible_{item}"),
                "open_file_metadata_count": "",
                "definition": definition,
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        raise AuditError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_lookup(summary_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {str(row["item"]): int(row["patient_count"]) for row in summary_rows}


def _recommendation_lines(counts: dict[str, int]) -> list[str]:
    complete = counts.get("complete_multimodal", 0)
    rnaseq = counts.get("clinical_rnaseq", 0)
    pathology = counts.get("clinical_rnaseq_wsi", 0)
    omics = counts.get("clinical_rnaseq_mutation_methylation", 0)
    if complete >= 200:
        sufficiency = (
            f"Conditionally yes for a retrospective proof-of-concept: {complete} patients "
            "have all audited modalities plus usable OS. A high-capacity final model still "
            "needs strict cross-validation, modality dropout, and simpler baselines because "
            "the cohort is not large for deep multimodal learning."
        )
    else:
        sufficiency = (
            f"Not as the first robust deep-learning analysis: only {complete} patients "
            "have all audited modalities plus usable OS. Use the complete intersection as "
            "an exploratory sensitivity cohort."
        )
    return [
        f"**Is the current public data sufficient for a complete multimodal deep model?** {sufficiency}",
        (
            "**Recommended simplified first version:** use clinical + RNA-seq with OS "
            f"({rnaseq} patients) as the primary Stage 2 cohort. Add regularized multi-omics "
            f"({omics} patients) and clinical + RNA-seq + WSI ({pathology} patients) as "
            "separate extensions before attempting the final fusion model."
        ),
        (
            "**Modalities suitable for the primary model:** curated clinical covariates, OS, "
            "and tumor RNA-seq. Mutation MAF, CNV, and methylation are suitable structured "
            "extensions after single-modality and RNA-seq baselines are stable."
        ),
        (
            "**Modalities suitable for external or mechanism validation:** DSS and PFI are "
            "secondary internal endpoints; CPTAC proteomics and HPA IHC are protein-level "
            "mechanism validation sources; public GEO/CPTAC/ICGC transcriptomic cohorts are "
            "external validation sources. WSI should be modeled as a separate pathology "
            "branch before cross-modal fusion."
        ),
    ]


def write_report(
    paths: AuditPaths,
    *,
    mode: str,
    gdc_status: dict[str, Any],
    patient_rows: list[dict[str, Any]],
    survival_summary: list[dict[str, Any]],
    modality_summary: list[dict[str, Any]],
    warnings: list[str],
    survival_source: str,
) -> None:
    """Write the requested human-readable Stage 1 audit report and mirror."""

    counts = _summary_lookup(modality_summary)
    release = gdc_status.get("data_release", "fixture/offline")
    pytest_text = (
        paths.pytest_log.read_bytes()
        .replace(b"\x00", b"")
        .decode("utf-8", errors="replace")
        if paths.pytest_log.exists()
        else ""
    )
    pytest_lower = pytest_text.lower()
    pytest_passed = (
        ("passed" in pytest_lower or "[100%]" in pytest_text)
        and "failed" not in pytest_lower
    )
    verification = [
        (
            "`python scripts/stage1_audit_tcga_luad.py --dry-run`",
            "passed" if paths.dry_run_log.exists() else "not recorded",
            "Metadata query plan only; network disabled.",
        ),
        (
            "`python scripts/stage1_audit_tcga_luad.py --small-test`",
            "passed" if paths.small_test_log.exists() else "not recorded",
            "Offline miniature patient-intersection smoke test.",
        ),
        (
            "`python -m pytest tests -q`",
            "passed" if pytest_passed else "not recorded",
            "See `outputs/logs/stage1_pytest.txt`.",
        ),
        (
            "`python scripts/stage1_audit_tcga_luad.py`",
            "passed" if mode == "live" else "not run in this mode",
            "Live public-metadata audit.",
        ),
    ]
    lines = [
        "# Stage 1 TCGA-LUAD Data Availability Audit Report",
        "",
        f"- Generated: {_timestamp()}",
        f"- Mode: `{mode}`",
        f"- GDC release: `{release}`",
        f"- GDC project: `{GDC_PROJECT}`",
        f"- Patient-level matrix rows: `{len(patient_rows)}`",
        f"- TCGA-CDR survival table source: `{survival_source}`",
        "",
        "## Scope",
        "",
        "This stage audits public metadata and patient-level overlap. It does not download "
        "large expression, mutation, CNV, methylation, or SVS files.",
        "",
        "## Availability Summary",
        "",
        "| Type | Item | Patients | Open file metadata rows | Definition |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in modality_summary:
        lines.append(
            f"| {row['record_type']} | {row['item']} | {row['patient_count']} | "
            f"{row['open_file_metadata_count']} | {row['definition']} |"
        )
    lines.extend(
        [
            "",
            "## Survival Endpoint Summary",
            "",
            "| Endpoint | GDC patients | Status present | Time present | Usable | Missing status | Missing time |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in survival_summary:
        lines.append(
            f"| {row['endpoint']} | {row['gdc_patient_count']} | "
            f"{row['status_present_count']} | {row['time_present_count']} | "
            f"{row['usable_count']} | {row['missing_status_count']} | "
            f"{row['missing_time_count']} |"
        )
    lines.extend(
        [
            "",
            "## Verification Results",
            "",
            "| Command | Result | Notes |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| {command} | {status} | {notes} |"
        for command, status, notes in verification
    )
    lines.extend(["", "## Feasibility Answers", ""])
    lines.extend(f"- {line}" for line in _recommendation_lines(counts))
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `data/metadata/tcga_luad_patient_modality_matrix.csv`",
            "- `data/metadata/tcga_luad_survival_summary.csv`",
            "- `data/metadata/tcga_luad_available_modalities_summary.csv`",
            "- `data/metadata/tcga_luad_stage1_api_snapshot.json`",
            "- `outputs/reports/stage1_tcga_luad_audit_report.md`",
            "- `outputs/audit/stage1/audit_report.md`",
            "- `outputs/logs/stage1_tcga_luad_audit.json`",
            "- `outputs/logs/stage1_tcga_luad_dry_run.json`",
            "- `outputs/logs/stage1_tcga_luad_small_test.json`",
            "- `outputs/logs/stage1_pytest.txt`",
            "",
            "## Windows Run Commands",
            "",
            "```powershell",
            "cd SC-PROST-LUAD",
            "conda activate gpu_py310",
            "python scripts/stage1_audit_tcga_luad.py --help",
            "python scripts/stage1_audit_tcga_luad.py --dry-run",
            "python scripts/stage1_audit_tcga_luad.py --small-test",
            "python scripts/stage1_audit_tcga_luad.py",
            "python -m pytest tests -q",
            "```",
            "",
            "## Potential Issues",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {item}" for item in sorted(set(warnings)))
    else:
        lines.append("- No warnings recorded.")
    lines.extend(
        [
            "- The audit proves public metadata availability, not successful bulk-file download.",
            "- Public GDC responses can end early or time out; metadata requests retry automatically and fail with an actionable error after the configured retry count.",
            "- Raw WSI files are large and are intentionally deferred to Stage 6.",
            "- Public data availability does not remove the need for leakage control, feature selection inside folds, and external validation.",
            "",
            "## Next Step Suggestions",
            "",
            "- Freeze the patient-level matrix as the Stage 1 cohort manifest.",
            "- In Stage 2, download and preprocess clinical, OS, and tumor RNA-seq data first.",
            "- Build Cox, LASSO-Cox, Random Survival Forest, and DeepSurv baselines before adding modalities.",
            "- Use `docs/tcga_luad_manual_download_guide.md` when large files are needed.",
        ]
    )
    text = "\n".join(lines) + "\n"
    paths.report.write_text(text, encoding="utf-8")
    paths.audit_mirror.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.report, paths.audit_mirror)


def _small_test_fixture() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, str]],
    dict[str, list[dict[str, Any]]],
]:
    patient_ids = [f"TCGA-ZZ-{index:04d}" for index in range(1, 7)]
    cases = [
        {"case_id": f"fixture-case-{index}", "submitter_id": patient_id}
        for index, patient_id in enumerate(patient_ids, start=1)
    ]
    survival = {
        patient_ids[0]: {"OS": "1", "OS.time": "100", "DSS": "1", "DSS.time": "100", "PFI": "1", "PFI.time": "80"},
        patient_ids[1]: {"OS": "0", "OS.time": "800", "DSS": "0", "DSS.time": "800", "PFI": "1", "PFI.time": "300"},
        patient_ids[2]: {"OS": "0", "OS.time": "500", "DSS": "", "DSS.time": "", "PFI": "0", "PFI.time": "500"},
        patient_ids[3]: {"OS": "1", "OS.time": "", "DSS": "1", "DSS.time": "", "PFI": "", "PFI.time": ""},
        patient_ids[4]: {"OS": "", "OS.time": "", "DSS": "", "DSS.time": "", "PFI": "", "PFI.time": ""},
    }

    def file_hit(patient_id: str, suffix: str) -> dict[str, Any]:
        return {
            "file_id": f"fixture-{patient_id}-{suffix}",
            "cases": [
                {
                    "submitter_id": patient_id,
                    "samples": [
                        {
                            "submitter_id": f"{patient_id}-01A",
                            "sample_type": "Primary Tumor",
                        }
                    ],
                }
            ],
        }

    modality_files = {
        "rnaseq": [file_hit(item, "rna") for item in patient_ids[:5]],
        "mutation_maf": [file_hit(item, "maf") for item in patient_ids[:4]],
        "cnv": [file_hit(item, "cnv") for item in patient_ids[:3]],
        "methylation": [file_hit(item, "meth") for item in patient_ids[:4]],
        "diagnostic_wsi": [file_hit(item, "wsi") for item in (patient_ids[0], patient_ids[2], patient_ids[4])],
    }
    return {"data_release": "fixture/offline"}, cases, survival, modality_files


def _dry_run(paths: AuditPaths) -> dict[str, Any]:
    paths.ensure_dirs()
    lines = [
        "# Stage 1 TCGA-LUAD Dry Run",
        "",
        f"- Generated: {_timestamp()}",
        "- Network access: disabled",
        "- Large-file download: disabled",
        "",
        "## Planned Metadata Queries",
        "",
        f"- `{GDC_API}/status`",
        f"- `{GDC_API}/cases` filtered to `{GDC_PROJECT}`",
    ]
    lines.extend(
        f"- `{GDC_API}/files` for `{name}`: {spec['filters']}"
        for name, spec in MODALITY_SPECS.items()
    )
    lines.extend(
        [
            f"- `{TCGA_CDR_URL}` for curated OS, DSS, and PFI endpoints",
            "",
            "## Planned Outputs",
            "",
            "- `data/metadata/tcga_luad_patient_modality_matrix.csv`",
            "- `data/metadata/tcga_luad_survival_summary.csv`",
            "- `data/metadata/tcga_luad_available_modalities_summary.csv`",
            "- `outputs/reports/stage1_tcga_luad_audit_report.md`",
        ]
    )
    paths.dry_run_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "stage": "stage1",
        "mode": "dry-run",
        "status": "passed",
        "network_access": False,
        "report": str(paths.dry_run_report),
    }
    paths.dry_run_log.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_audit(
    root: str | Path = ".",
    *,
    dry_run: bool = False,
    small_test: bool = False,
    refresh_survival_cache: bool = False,
    timeout: int = 60,
    retries: int = 3,
) -> dict[str, Any]:
    """Run the Stage 1 TCGA-LUAD public metadata audit."""

    if dry_run and small_test:
        raise AuditError("Choose either --dry-run or --small-test, not both.")
    paths = AuditPaths.from_root(root)
    paths.ensure_dirs()
    if dry_run:
        return _dry_run(paths)

    warnings: list[str] = []
    if small_test:
        gdc_status, cases, survival, modality_files = _small_test_fixture()
        survival_source = "built-in offline fixture"
        mode = "small-test"
    else:
        gdc_status = fetch_gdc_status(timeout=timeout, retries=retries)
        cases = fetch_gdc_cases(timeout=timeout, retries=retries)
        survival_text, survival_source = fetch_survival_table(
            paths.survival_cache,
            refresh=refresh_survival_cache,
            timeout=timeout,
            retries=retries,
        )
        survival, survival_warnings = parse_survival_table(survival_text)
        warnings.extend(survival_warnings)
        modality_files = {
            modality: fetch_gdc_modality_files(
                modality,
                timeout=timeout,
                retries=retries,
            )
            for modality in MODALITY_SPECS
        }
        mode = "live"

    patient_rows, modality_counts, matrix_warnings = build_patient_matrix(
        cases,
        survival,
        modality_files,
    )
    warnings.extend(matrix_warnings)
    survival_summary = build_survival_summary(patient_rows, survival)
    modality_summary = build_modality_summary(patient_rows, modality_files)

    _write_csv(paths.patient_matrix, patient_rows, PATIENT_MATRIX_COLUMNS)
    _write_csv(
        paths.survival_summary,
        survival_summary,
        list(survival_summary[0]),
    )
    _write_csv(
        paths.modality_summary,
        modality_summary,
        list(modality_summary[0]),
    )

    snapshot = {
        "generated_at": _timestamp(),
        "mode": mode,
        "gdc_project": GDC_PROJECT,
        "gdc_status": gdc_status,
        "gdc_case_count": len(cases),
        "survival_patient_count": len(survival),
        "survival_source": survival_source,
        "open_file_metadata_count": {
            key: len(value) for key, value in modality_files.items()
        },
        "tumor_linked_patient_count": {
            key: len(value) for key, value in modality_counts.items()
        },
        "warnings": sorted(set(warnings)),
    }
    paths.api_snapshot.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    write_report(
        paths,
        mode=mode,
        gdc_status=gdc_status,
        patient_rows=patient_rows,
        survival_summary=survival_summary,
        modality_summary=modality_summary,
        warnings=warnings,
        survival_source=survival_source,
    )
    result = {
        "stage": "stage1",
        "mode": mode,
        "status": "passed",
        "gdc_release": gdc_status.get("data_release", "fixture/offline"),
        "patient_count": len(patient_rows),
        "summary": _summary_lookup(modality_summary),
        "warnings": sorted(set(warnings)),
        "report": str(paths.report),
    }
    log_path = paths.small_test_log if small_test else paths.log
    log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit TCGA-LUAD public metadata availability without downloading "
            "large biomedical files."
        ),
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and save the metadata query plan without network access.",
    )
    parser.add_argument(
        "--small-test",
        action="store_true",
        help="Run an offline smoke test using built-in miniature metadata.",
    )
    parser.add_argument(
        "--refresh-survival-cache",
        action="store_true",
        help="Redownload the small public TCGA-CDR survival table even if cached.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request network timeout in seconds. Default: 60.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Public endpoint retry count. Default: 3.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_audit(
            root=args.root,
            dry_run=args.dry_run,
            small_test=args.small_test,
            refresh_survival_cache=args.refresh_survival_cache,
            timeout=args.timeout,
            retries=args.retries,
        )
    except AuditError as exc:
        print(f"Stage 1 audit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
