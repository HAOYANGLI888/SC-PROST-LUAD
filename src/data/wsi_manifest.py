"""TCGA-LUAD diagnostic WSI metadata audit and manifest construction."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data.rnaseq_preprocess import normalize_tcga_patient_id


GDC_API = "https://api.gdc.cancer.gov"
GDC_PROJECT = "TCGA-LUAD"
USER_AGENT = "SC-PROST-LUAD-stage6a/0.1"


class WSIManifestError(RuntimeError):
    """Raised when WSI metadata cannot be audited safely."""


@dataclass(frozen=True)
class WSIPaths:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "WSIPaths":
        return cls(Path(root).resolve())

    @property
    def metadata(self) -> Path:
        return self.root / "data" / "metadata"

    @property
    def tables(self) -> Path:
        return self.root / "outputs" / "tables"

    @property
    def reports(self) -> Path:
        return self.root / "outputs" / "reports"

    @property
    def manifest(self) -> Path:
        return self.metadata / "stage6a_tcga_luad_wsi_manifest.tsv"

    @property
    def patient_slide_map(self) -> Path:
        return self.metadata / "stage6a_tcga_luad_wsi_patient_slide_map.csv"

    @property
    def overlap(self) -> Path:
        return self.metadata / "stage6a_wsi_modality_overlap_summary.csv"

    @property
    def query_summary(self) -> Path:
        return self.tables / "stage6a_wsi_query_summary.csv"

    @property
    def overlap_table(self) -> Path:
        return self.tables / "stage6a_wsi_overlap_summary.csv"

    @property
    def audit_report(self) -> Path:
        return self.reports / "stage6a_wsi_audit_report.md"

    def ensure_dirs(self) -> None:
        for path in (self.metadata, self.tables, self.reports):
            path.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _filters() -> dict[str, Any]:
    return {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": GDC_PROJECT}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
            {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
        ],
    }


def query_diagnostic_wsi(*, timeout: int = 120, retries: int = 4) -> list[dict[str, Any]]:
    """Query open TCGA-LUAD diagnostic SVS files from the public GDC API."""

    fields = (
        "file_id,file_name,file_size,md5sum,state,access,data_category,data_type,"
        "data_format,experimental_strategy,cases.case_id,cases.submitter_id,"
        "cases.samples.submitter_id,cases.samples.sample_type"
    )
    params = {
        "filters": json.dumps(_filters(), separators=(",", ":")),
        "fields": fields,
        "format": "JSON",
        "size": 2000,
    }
    url = f"{GDC_API}/files?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            hits = payload["data"]["hits"]
            total = int(payload["data"]["pagination"]["total"])
            if not isinstance(hits, list) or len(hits) != total or not hits:
                raise WSIManifestError(f"GDC returned {len(hits)} of {total} WSI records.")
            return hits
        except (KeyError, TypeError, ValueError, OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(attempt)
    raise WSIManifestError(f"GDC WSI query failed after {retries} attempts: {last_error}")


def flatten_patient_slides(hits: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten GDC records and deterministically mark one preferred slide per patient."""

    rows = []
    for hit in hits:
        for case in hit.get("cases", []):
            patient_id = normalize_tcga_patient_id(case.get("submitter_id"))
            if patient_id is None:
                continue
            samples = case.get("samples", []) or [{}]
            for sample in samples:
                rows.append(
                    {
                        "file_id": str(hit.get("file_id") or hit.get("id") or ""),
                        "file_name": str(hit.get("file_name") or ""),
                        "md5sum": str(hit.get("md5sum") or ""),
                        "file_size": int(hit.get("file_size") or 0),
                        "state": str(hit.get("state") or "released"),
                        "access": str(hit.get("access") or ""),
                        "data_category": str(hit.get("data_category") or ""),
                        "data_type": str(hit.get("data_type") or ""),
                        "data_format": str(hit.get("data_format") or ""),
                        "experimental_strategy": str(hit.get("experimental_strategy") or ""),
                        "case_id": str(case.get("case_id") or ""),
                        "patient_id": patient_id,
                        "sample_id": str(sample.get("submitter_id") or ""),
                        "sample_type": str(sample.get("sample_type") or ""),
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates("file_id")
    if frame.empty:
        raise WSIManifestError("No patient-linked diagnostic WSI files remained.")
    frame["_sample_priority"] = frame["sample_type"].map({"Primary Tumor": 0, "Recurrent Tumor": 1}).fillna(9)
    frame = frame.sort_values(["patient_id", "_sample_priority", "file_name", "file_id"]).reset_index(drop=True)
    frame["slide_rank_for_patient"] = frame.groupby("patient_id").cumcount() + 1
    frame["patient_slide_count"] = frame.groupby("patient_id")["file_id"].transform("count")
    frame["preferred_slide_for_smallset"] = frame["slide_rank_for_patient"] == 1
    return frame.drop(columns="_sample_priority")


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold().isin({"true", "1", "yes"})


def _overlap_summary(paths: WSIPaths, slide_map: pd.DataFrame) -> pd.DataFrame:
    matrix_path = paths.metadata / "tcga_luad_patient_modality_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Stage 1 patient modality matrix is missing: {matrix_path}")
    matrix = pd.read_csv(matrix_path)
    has_wsi = set(slide_map["patient_id"])
    matrix["stage6a_diagnostic_wsi"] = matrix["patient_id"].isin(has_wsi)
    clinical = _bool(matrix["clinical_available"])
    os_available = _bool(matrix["os_available"])
    rna = _bool(matrix["rnaseq_available"])
    wsi = matrix["stage6a_diagnostic_wsi"]
    return pd.DataFrame(
        [
            {"cohort": "diagnostic_wsi", "patient_count": int(wsi.sum()), "definition": "Open GDC diagnostic SVS"},
            {"cohort": "clinical_os_wsi", "patient_count": int((clinical & os_available & wsi).sum()), "definition": "Clinical + usable OS + diagnostic WSI"},
            {"cohort": "clinical_os_wsi_rna", "patient_count": int((clinical & os_available & wsi & rna).sum()), "definition": "Clinical + usable OS + diagnostic WSI + RNA-seq"},
        ]
    )


def _write_audit(paths: WSIPaths, slide_map: pd.DataFrame, overlap: pd.DataFrame) -> None:
    total_bytes = int(slide_map["file_size"].sum())
    patients = int(slide_map["patient_id"].nunique())
    multi = int((slide_map.groupby("patient_id").size() > 1).sum())
    clinical_os_wsi = int(overlap.loc[overlap["cohort"] == "clinical_os_wsi", "patient_count"].iloc[0])
    clinical_os_wsi_rna = int(overlap.loc[overlap["cohort"] == "clinical_os_wsi_rna", "patient_count"].iloc[0])
    report = (
        "# Stage 6A TCGA-LUAD WSI Audit Report\n\n"
        f"Generated: {_timestamp()}\n\n"
        "## Scope\n\n"
        "This report audits public TCGA-LUAD diagnostic SVS metadata only. It does not "
        "claim formal pathology-model training and does not trigger full WSI download.\n\n"
        "## Availability\n\n"
        f"- Diagnostic SVS files: {len(slide_map)}.\n"
        f"- Patients with diagnostic WSI: {patients}.\n"
        f"- Clinical + usable OS + WSI patients: {clinical_os_wsi}.\n"
        f"- Clinical + usable OS + WSI + RNA patients: {clinical_os_wsi_rna}.\n"
        f"- Mean slides per WSI patient: {len(slide_map) / patients:.3f}.\n"
        f"- Patients with multiple diagnostic slides: {multi}.\n"
        f"- Total diagnostic SVS size: {total_bytes / 1e9:.1f} GB "
        f"({total_bytes / 1024**4:.3f} TiB).\n\n"
        "## Deterministic Slide Rule\n\n"
        "All diagnostic slides remain in the patient map. For Stage 6A smallset download, "
        "one slide per patient is selected deterministically: Primary Tumor before "
        "Recurrent Tumor, then lexical `file_name`, then `file_id`. Later full training "
        "should aggregate all slide patch bags at patient level.\n\n"
        "## Recommendation\n\n"
        "Do not download all slides by default. The full diagnostic set is feasible only "
        "after storage and compute planning. Start with the balanced smallset and synthetic "
        "pipeline smoke test.\n"
    )
    paths.audit_report.write_text(report, encoding="utf-8")


def build_wsi_manifest(root: str | Path = ".", *, small_test: bool = False) -> dict[str, Any]:
    """Build real GDC WSI outputs, or an isolated miniature manifest."""

    paths = WSIPaths.from_root(root)
    paths.ensure_dirs()
    if small_test:
        target = paths.metadata / "stage6a_small_test"
        target.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            [
                {
                    "file_id": f"TOY-{index:03d}",
                    "file_name": f"TOY-{index:03d}.tif",
                    "md5sum": "",
                    "file_size": 0,
                    "state": "synthetic",
                    "access": "local",
                    "data_category": "synthetic",
                    "data_type": "Slide Image",
                    "data_format": "TIFF",
                    "experimental_strategy": "Synthetic Diagnostic Slide",
                    "case_id": f"TOY-CASE-{index:03d}",
                    "patient_id": f"TOY-PATIENT-{index:03d}",
                    "sample_id": f"TOY-SAMPLE-{index:03d}",
                    "sample_type": "Primary Tumor",
                    "slide_rank_for_patient": 1,
                    "patient_slide_count": 1,
                    "preferred_slide_for_smallset": True,
                }
                for index in range(20)
            ]
        )
        frame.to_csv(target / "wsi_patient_slide_map.csv", index=False)
        return {"status": "passed", "dataset_mode": "toy_small_test", "slides": len(frame)}
    hits = query_diagnostic_wsi()
    slide_map = flatten_patient_slides(hits)
    manifest = slide_map[["file_id", "file_name", "md5sum", "file_size", "state"]].rename(
        columns={"file_id": "id", "file_name": "filename", "md5sum": "md5", "file_size": "size"}
    )
    manifest.to_csv(paths.manifest, sep="\t", index=False)
    slide_map.to_csv(paths.patient_slide_map, index=False)
    overlap = _overlap_summary(paths, slide_map)
    overlap.to_csv(paths.overlap, index=False)
    overlap.to_csv(paths.overlap_table, index=False)
    query = pd.DataFrame(
        [
            {"metric": "diagnostic_svs_files", "value": len(slide_map), "notes": "Open GDC Diagnostic Slide SVS"},
            {"metric": "diagnostic_wsi_patients", "value": slide_map["patient_id"].nunique(), "notes": "Unique TCGA barcode"},
            {"metric": "patients_with_multiple_slides", "value": int((slide_map.groupby("patient_id").size() > 1).sum()), "notes": "All diagnostic slides retained"},
            {"metric": "mean_slides_per_patient", "value": len(slide_map) / slide_map["patient_id"].nunique(), "notes": "Diagnostic SVS files / patients"},
            {"metric": "total_bytes", "value": int(slide_map["file_size"].sum()), "notes": "Expected full diagnostic SVS download bytes"},
            {"metric": "total_gb_decimal", "value": slide_map["file_size"].sum() / 1e9, "notes": "Storage estimate before derived patches/features"},
        ]
    )
    query.to_csv(paths.query_summary, index=False)
    _write_audit(paths, slide_map, overlap)
    return {
        "status": "passed",
        "dataset_mode": "real_gdc_metadata",
        "diagnostic_svs_files": len(slide_map),
        "diagnostic_wsi_patients": int(slide_map["patient_id"].nunique()),
        "total_gb_decimal": float(slide_map["file_size"].sum() / 1e9),
    }

