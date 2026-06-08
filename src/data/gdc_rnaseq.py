"""GDC TCGA-LUAD STAR-counts acquisition and TPM matrix construction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from data.rnaseq_preprocess import normalize_tcga_patient_id


GDC_API = "https://api.gdc.cancer.gov"
GDC_PROJECT = "TCGA-LUAD"
USER_AGENT = "SC-PROST-LUAD-stage2b/0.1"
PRIMARY_TUMOR = "Primary Tumor"
DEFAULT_WORKERS = 8


class GDCRNASeqError(RuntimeError):
    """Raised when Stage 2B GDC acquisition cannot continue safely."""


@dataclass(frozen=True)
class GDCRNASeqPaths:
    """Resolved Stage 2B inputs and outputs."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "GDCRNASeqPaths":
        return cls(Path(root).resolve())

    @property
    def manifest(self) -> Path:
        return self.root / "data" / "metadata" / "gdc_tcga_luad_rnaseq_star_counts_manifest.tsv"

    @property
    def file_patient_map(self) -> Path:
        return self.root / "data" / "metadata" / "gdc_tcga_luad_rnaseq_file_patient_map.csv"

    @property
    def query_summary(self) -> Path:
        return self.root / "outputs" / "tables" / "stage2_gdc_rnaseq_query_summary.csv"

    @property
    def download_dir(self) -> Path:
        return self.root / "data" / "raw" / "tcga_luad" / "rnaseq" / "gdc_star_counts"

    @property
    def download_summary(self) -> Path:
        return self.root / "data" / "metadata" / "stage2_rnaseq_download_summary.csv"

    @property
    def matrix(self) -> Path:
        return self.root / "data" / "raw" / "tcga_luad" / "rnaseq" / "tcga_luad_tpm_matrix.csv"

    @property
    def matrix_summary(self) -> Path:
        return self.root / "data" / "metadata" / "stage2_rnaseq_matrix_build_summary.csv"

    @property
    def duplicate_resolution(self) -> Path:
        return self.root / "data" / "metadata" / "stage2_rnaseq_duplicate_patient_resolution.csv"

    @property
    def audit_report(self) -> Path:
        return self.root / "outputs" / "audit" / "stage2b" / "audit_report.md"

    def ensure_dirs(self) -> None:
        for directory in (
            self.manifest.parent,
            self.query_summary.parent,
            self.download_dir,
            self.audit_report.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _request_json(url: str, *, retries: int = 4, timeout: int = 90) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(attempt)
    raise GDCRNASeqError(f"GDC request failed after {retries} attempts: {url}. Last error: {last_error}")


def _gdc_filters() -> dict[str, Any]:
    return {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": GDC_PROJECT}},
            {"op": "=", "content": {"field": "data_category", "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "RNA-Seq"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ],
    }


def query_gdc_star_counts(*, retries: int = 4, timeout: int = 90) -> list[dict[str, Any]]:
    """Query open TCGA-LUAD STAR-counts file metadata from GDC."""

    fields = (
        "file_id,file_name,file_size,md5sum,updated_datetime,data_category,data_type,"
        "data_format,experimental_strategy,analysis.workflow_type,access,"
        "cases.case_id,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type"
    )
    params = {
        "filters": json.dumps(_gdc_filters(), separators=(",", ":")),
        "fields": fields,
        "format": "JSON",
        "size": 2000,
    }
    url = f"{GDC_API}/files?{urllib.parse.urlencode(params)}"
    response = _request_json(url, retries=retries, timeout=timeout)
    try:
        hits = response["data"]["hits"]
        total = int(response["data"]["pagination"]["total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GDCRNASeqError("Unexpected GDC files response schema.") from exc
    if not isinstance(hits, list) or not hits:
        raise GDCRNASeqError("GDC STAR-counts query returned no files.")
    if len(hits) != total:
        raise GDCRNASeqError(f"GDC query returned {len(hits)} of {total} files; increase page size.")
    return hits


def _flatten_primary_tumor_rows(hits: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in hits:
        analysis = hit.get("analysis") or {}
        for case in hit.get("cases", []):
            patient_id = normalize_tcga_patient_id(case.get("submitter_id"))
            for sample in case.get("samples", []):
                if sample.get("sample_type") != PRIMARY_TUMOR:
                    continue
                if patient_id is None:
                    raise GDCRNASeqError(f"Could not normalize TCGA patient ID for file {hit.get('file_id')}.")
                rows.append(
                    {
                        "file_id": hit.get("file_id", ""),
                        "file_name": hit.get("file_name", ""),
                        "md5sum": hit.get("md5sum", ""),
                        "file_size": int(hit.get("file_size") or 0),
                        "updated_datetime": hit.get("updated_datetime", ""),
                        "case_id": case.get("case_id", ""),
                        "patient_id": patient_id,
                        "sample_id": sample.get("submitter_id", ""),
                        "sample_type": sample.get("sample_type", ""),
                        "workflow_type": analysis.get("workflow_type", ""),
                        "access": hit.get("access", ""),
                    }
                )
    unique = {row["file_id"]: row for row in rows}
    flattened = sorted(unique.values(), key=lambda row: (row["patient_id"], row["sample_id"], row["file_id"]))
    if not flattened:
        raise GDCRNASeqError("No Primary Tumor STAR-counts files remained after local filtering.")
    return flattened


def build_manifest(root: str | Path = ".", *, retries: int = 4, timeout: int = 90) -> dict[str, Any]:
    """Build a gdc-client manifest and patient mapping table."""

    paths = GDCRNASeqPaths.from_root(root)
    paths.ensure_dirs()
    hits = query_gdc_star_counts(retries=retries, timeout=timeout)
    rows = _flatten_primary_tumor_rows(hits)
    manifest_rows = [
        {
            "id": row["file_id"],
            "filename": row["file_name"],
            "md5": row["md5sum"],
            "size": row["file_size"],
            "state": "released",
        }
        for row in rows
    ]
    pd.DataFrame(manifest_rows).to_csv(paths.manifest, sep="\t", index=False)
    pd.DataFrame(rows).to_csv(paths.file_patient_map, index=False)
    summary_rows = [
        {"metric": "gdc_open_star_counts_files_all_sample_types", "value": len(hits), "notes": "GDC API query before local Primary Tumor filter"},
        {"metric": "primary_tumor_star_counts_files", "value": len(rows), "notes": "Manifest rows"},
        {"metric": "primary_tumor_unique_patients", "value": len({row["patient_id"] for row in rows}), "notes": "Before deterministic duplicate resolution"},
        {"metric": "primary_tumor_total_bytes", "value": sum(row["file_size"] for row in rows), "notes": "Expected direct-download bytes"},
        {"metric": "workflow_type", "value": "STAR - Counts", "notes": "GDC analysis.workflow_type"},
        {"metric": "expression_value_for_matrix", "value": "tpm_unstranded", "notes": "Extracted during matrix build"},
    ]
    pd.DataFrame(summary_rows).to_csv(paths.query_summary, index=False)
    result = {
        "status": "passed",
        "generated_at": _timestamp(),
        "gdc_project": GDC_PROJECT,
        "workflow_type": "STAR - Counts",
        "all_sample_type_file_count": len(hits),
        "primary_tumor_file_count": len(rows),
        "primary_tumor_unique_patient_count": len({row["patient_id"] for row in rows}),
        "primary_tumor_total_bytes": sum(row["file_size"] for row in rows),
        "manifest": str(paths.manifest),
        "file_patient_map": str(paths.file_patient_map),
    }
    _write_stage2b_audit(paths, "manifest_built", result)
    return result


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"GDC manifest not found: {path}")
    frame = pd.read_csv(path, sep="\t", dtype={"id": str, "filename": str, "md5": str})
    required = {"id", "filename", "md5", "size"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise GDCRNASeqError(f"GDC manifest is missing columns: {missing}")
    if frame.empty:
        raise GDCRNASeqError(f"GDC manifest is empty: {path}")
    frame["size"] = pd.to_numeric(frame["size"], errors="raise").astype(int)
    return frame


def _md5(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def locate_downloaded_file(download_dir: Path, file_id: str, filename: str) -> Path:
    """Return the expected nested gdc-client-compatible local file path."""

    return download_dir / file_id / filename


def _is_complete(path: Path, *, expected_size: int, expected_md5: str, verify_md5: bool) -> bool:
    if not path.exists() or path.stat().st_size != expected_size:
        return False
    return not verify_md5 or _md5(path).lower() == str(expected_md5).lower()


def _download_one_direct(
    row: dict[str, Any],
    *,
    download_dir: Path,
    retries: int,
    timeout: int,
    verify_md5: bool,
) -> dict[str, Any]:
    file_id = str(row["id"])
    filename = str(row["filename"])
    expected_size = int(row["size"])
    expected_md5 = str(row["md5"])
    target = locate_downloaded_file(download_dir, file_id, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_complete(target, expected_size=expected_size, expected_md5=expected_md5, verify_md5=verify_md5):
        return {"file_id": file_id, "filename": filename, "status": "skipped_complete", "bytes": expected_size}

    partial = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            existing = partial.stat().st_size if partial.exists() else 0
            if existing > expected_size:
                partial.unlink()
                existing = 0
            headers = {"User-Agent": USER_AGENT}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            request = urllib.request.Request(f"{GDC_API}/data/{file_id}", headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if existing and status != 206:
                    partial.unlink(missing_ok=True)
                    existing = 0
                mode = "ab" if existing and status == 206 else "wb"
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if partial.stat().st_size != expected_size:
                raise GDCRNASeqError(
                    f"Size mismatch for {file_id}: got {partial.stat().st_size}, expected {expected_size}."
                )
            if verify_md5 and _md5(partial).lower() != expected_md5.lower():
                raise GDCRNASeqError(f"MD5 mismatch for {file_id}.")
            partial.replace(target)
            return {"file_id": file_id, "filename": filename, "status": "downloaded", "bytes": expected_size}
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
            GDCRNASeqError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(attempt)
    return {"file_id": file_id, "filename": filename, "status": "failed", "bytes": 0, "error": str(last_error)}


def _run_gdc_client(gdc_client: Path, manifest: Path, download_dir: Path) -> None:
    if not gdc_client.exists():
        raise FileNotFoundError(
            f"gdc-client.exe not found: {gdc_client}. Download the Windows client from "
            "https://gdc.cancer.gov/access-data/gdc-data-transfer-tool"
        )
    command = [str(gdc_client), "download", "-m", str(manifest), "-d", str(download_dir)]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise GDCRNASeqError(f"gdc-client exited with code {completed.returncode}.")


def download_manifest(
    root: str | Path = ".",
    *,
    manifest: str | Path | None = None,
    gdc_client: str | Path | None = None,
    method: str = "auto",
    workers: int = DEFAULT_WORKERS,
    retries: int = 4,
    timeout: int = 120,
    verify_md5: bool = True,
) -> dict[str, Any]:
    """Download manifest files with gdc-client or resumable public GDC API GETs."""

    paths = GDCRNASeqPaths.from_root(root)
    paths.ensure_dirs()
    manifest_path = Path(manifest) if manifest else paths.manifest
    if not manifest_path.is_absolute():
        manifest_path = paths.root / manifest_path
    frame = _load_manifest(manifest_path)
    client_path = Path(gdc_client) if gdc_client else None
    resolved_method = method
    if method == "auto":
        resolved_method = "gdc-client" if client_path and client_path.exists() else "direct-api"
    if resolved_method == "gdc-client":
        if client_path is None:
            raise GDCRNASeqError(
                "--method gdc-client requires --gdc-client C:\\path\\to\\gdc-client.exe. "
                "Install it from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool"
            )
        _run_gdc_client(client_path, manifest_path, paths.download_dir)
        results = []
        for row in frame.to_dict(orient="records"):
            target = locate_downloaded_file(paths.download_dir, str(row["id"]), str(row["filename"]))
            status = "downloaded_or_existing" if _is_complete(
                target,
                expected_size=int(row["size"]),
                expected_md5=str(row["md5"]),
                verify_md5=verify_md5,
            ) else "failed"
            results.append({"file_id": row["id"], "filename": row["filename"], "status": status, "bytes": target.stat().st_size if target.exists() else 0})
    elif resolved_method == "direct-api":
        if client_path is None or not client_path.exists():
            print(
                "gdc-client.exe was not found; using resumable public GDC /data/{file_id} downloads. "
                "For gdc-client, install https://gdc.cancer.gov/access-data/gdc-data-transfer-tool",
                file=sys.stderr,
            )
        results = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(
                    _download_one_direct,
                    row,
                    download_dir=paths.download_dir,
                    retries=retries,
                    timeout=timeout,
                    verify_md5=verify_md5,
                )
                for row in frame.to_dict(orient="records")
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if index % 25 == 0 or result["status"] == "failed":
                    print(f"[{index}/{len(futures)}] {result['status']}: {result['file_id']}", file=sys.stderr)
    else:
        raise ValueError("method must be one of: auto, gdc-client, direct-api")

    result_frame = pd.DataFrame(results).sort_values(["status", "file_id"])
    result_frame.to_csv(paths.download_summary, index=False)
    failed = int((result_frame["status"] == "failed").sum())
    complete = len(result_frame) - failed
    summary = {
        "status": "passed" if failed == 0 else "failed",
        "generated_at": _timestamp(),
        "method": resolved_method,
        "manifest_file_count": len(frame),
        "complete_file_count": complete,
        "failed_file_count": failed,
        "download_dir": str(paths.download_dir),
        "download_summary": str(paths.download_summary),
    }
    _write_stage2b_audit(paths, "download_completed" if failed == 0 else "download_incomplete", summary)
    if failed:
        raise GDCRNASeqError(
            f"{failed} GDC RNA-seq files failed to download. Rerun the same command to resume. "
            f"See {paths.download_summary}."
        )
    return summary


def _select_patient_files(file_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"file_id", "file_name", "patient_id", "sample_id", "sample_type", "updated_datetime"}
    missing = sorted(required - set(file_map.columns))
    if missing:
        raise GDCRNASeqError(f"File-patient map is missing columns: {missing}")
    primary = file_map.loc[file_map["sample_type"] == PRIMARY_TUMOR].copy()
    if primary.empty:
        raise GDCRNASeqError("File-patient map contains no Primary Tumor rows.")
    primary["updated_sort"] = pd.to_datetime(primary["updated_datetime"], errors="coerce", utc=True)
    primary = primary.sort_values(
        ["patient_id", "updated_sort", "file_id"],
        ascending=[True, False, True],
        na_position="last",
    )
    primary["selection_rank"] = primary.groupby("patient_id").cumcount() + 1
    primary["selected"] = primary["selection_rank"] == 1
    resolution = primary.loc[
        primary.groupby("patient_id")["patient_id"].transform("size") > 1,
        ["patient_id", "sample_id", "file_id", "file_name", "updated_datetime", "selection_rank", "selected"],
    ].copy()
    if not resolution.empty:
        resolution["resolution_rule"] = "Primary Tumor; newest updated_datetime; then lexicographically first file_id"
    return primary.loc[primary["selected"]].copy(), resolution


def _read_tpm(path: Path, *, protein_coding_only: bool) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Downloaded GDC STAR-counts file not found: {path}")
    try:
        frame = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            usecols=["gene_id", "gene_name", "gene_type", "tpm_unstranded"],
            dtype={"gene_id": str, "gene_name": str, "gene_type": str},
        )
    except ValueError as exc:
        raise GDCRNASeqError(
            f"STAR-counts schema changed or tpm_unstranded is missing: {path}"
        ) from exc
    if protein_coding_only:
        frame = frame.loc[frame["gene_type"] == "protein_coding"].copy()
    frame["tpm_unstranded"] = pd.to_numeric(frame["tpm_unstranded"], errors="coerce")
    frame = frame.dropna(subset=["gene_id", "tpm_unstranded"])
    frame["gene_id"] = frame["gene_id"].str.replace(r"\.\d+$", "", regex=True)
    if frame["gene_id"].duplicated().any():
        frame = frame.groupby("gene_id", as_index=False)["tpm_unstranded"].mean()
    if frame.empty:
        raise GDCRNASeqError(f"No usable TPM rows found in {path}")
    return frame.set_index("gene_id")["tpm_unstranded"]


def build_tpm_matrix(
    root: str | Path = ".",
    *,
    protein_coding_only: bool = True,
) -> dict[str, Any]:
    """Build a patients x genes TPM matrix from downloaded GDC STAR-counts TSVs."""

    paths = GDCRNASeqPaths.from_root(root)
    paths.ensure_dirs()
    if not paths.file_patient_map.exists():
        raise FileNotFoundError(f"File-patient map not found: {paths.file_patient_map}")
    file_map = pd.read_csv(paths.file_patient_map, dtype=str)
    selected, resolution = _select_patient_files(file_map)
    series: list[pd.Series] = []
    missing_files: list[str] = []
    for row in selected.to_dict(orient="records"):
        path = locate_downloaded_file(paths.download_dir, row["file_id"], row["file_name"])
        if not path.exists():
            missing_files.append(str(path))
            continue
        values = _read_tpm(path, protein_coding_only=protein_coding_only)
        values.name = row["patient_id"]
        series.append(values)
    if missing_files:
        raise GDCRNASeqError(
            f"{len(missing_files)} selected GDC files are missing. "
            f"Rerun scripts/stage2_download_gdc_rnaseq.py. First missing file: {missing_files[0]}"
        )
    if not series:
        raise GDCRNASeqError("No downloaded TPM files were available for matrix construction.")
    matrix = pd.concat(series, axis=1, join="inner").T
    matrix.index.name = "patient_id"
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    matrix.reset_index().to_csv(paths.matrix, index=False)
    resolution.to_csv(paths.duplicate_resolution, index=False)
    summary_rows = [
        {"metric": "source", "value": "GDC_STAR_COUNTS", "notes": "Primary analysis source"},
        {"metric": "matrix_layout", "value": "patients_x_genes", "notes": "First column patient_id; remaining columns Ensembl gene IDs without version suffix"},
        {"metric": "expression_value", "value": "tpm_unstranded", "notes": "Raw TPM; Stage 2 preparation applies log2(TPM+1)"},
        {"metric": "protein_coding_only", "value": protein_coding_only, "notes": "Filtered using gene_type"},
        {"metric": "manifest_primary_tumor_file_count", "value": len(file_map), "notes": "Before deterministic patient duplicate handling"},
        {"metric": "duplicate_patient_count", "value": int(resolution["patient_id"].nunique()) if not resolution.empty else 0, "notes": "Patients with more than one Primary Tumor STAR-counts file"},
        {"metric": "matrix_patient_count", "value": matrix.shape[0], "notes": "After deterministic duplicate handling"},
        {"metric": "matrix_gene_count", "value": matrix.shape[1], "notes": "Shared genes retained across selected files"},
    ]
    pd.DataFrame(summary_rows).to_csv(paths.matrix_summary, index=False)
    result = {
        "status": "passed",
        "generated_at": _timestamp(),
        "source": "GDC_STAR_COUNTS",
        "matrix_layout": "patients_x_genes",
        "expression_value": "tpm_unstranded",
        "protein_coding_only": protein_coding_only,
        "manifest_primary_tumor_file_count": len(file_map),
        "duplicate_patient_count": int(resolution["patient_id"].nunique()) if not resolution.empty else 0,
        "matrix_patient_count": int(matrix.shape[0]),
        "matrix_gene_count": int(matrix.shape[1]),
        "matrix": str(paths.matrix),
    }
    _write_stage2b_audit(paths, "matrix_built", result)
    return result


def _write_stage2b_audit(paths: GDCRNASeqPaths, status: str, details: dict[str, Any]) -> None:
    paths.audit_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 2B GDC RNA-seq Acquisition Audit Report",
        "",
        f"- Updated: {_timestamp()}",
        f"- Status: `{status}`",
        "",
        "## Current Details",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in details.items())
    lines.extend(
        [
            "",
            "## Scope Guard",
            "",
            "- Stage 2B is restricted to TCGA-LUAD clinical + OS + RNA-seq.",
            "- No mutation, CNV, methylation, WSI, single-cell, or protein data are used.",
            "",
            "## Windows Commands",
            "",
            "```powershell",
            "cd SC-PROST-LUAD",
            "python scripts/stage2_build_gdc_rnaseq_manifest.py",
            "python scripts/stage2_download_gdc_rnaseq.py --method direct-api",
            "python scripts/stage2_build_rnaseq_tpm_matrix.py",
            "```",
        ]
    )
    paths.audit_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Primary Tumor TCGA-LUAD GDC STAR-counts manifest.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=90)
    return parser


def manifest_main(argv: list[str] | None = None) -> int:
    args = build_manifest_parser().parse_args(argv)
    try:
        result = build_manifest(args.root, retries=args.retries, timeout=args.timeout)
    except (FileNotFoundError, GDCRNASeqError) as exc:
        print(f"Stage 2B manifest build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def build_download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download public TCGA-LUAD GDC STAR-counts files with resume checks.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--manifest", default="data/metadata/gdc_tcga_luad_rnaseq_star_counts_manifest.tsv")
    parser.add_argument("--gdc-client", help=r"Optional path such as C:\path\to\gdc-client.exe")
    parser.add_argument("--method", choices=["auto", "gdc-client", "direct-api"], default="auto")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers for direct API mode.")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--skip-md5", action="store_true", help="Skip MD5 validation; size checks still run.")
    return parser


def download_main(argv: list[str] | None = None) -> int:
    args = build_download_parser().parse_args(argv)
    try:
        result = download_manifest(
            args.root,
            manifest=args.manifest,
            gdc_client=args.gdc_client,
            method=args.method,
            workers=args.workers,
            retries=args.retries,
            timeout=args.timeout,
            verify_md5=not args.skip_md5,
        )
    except (FileNotFoundError, GDCRNASeqError, ValueError) as exc:
        print(f"Stage 2B download failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def build_matrix_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a TCGA-LUAD patients x genes TPM matrix from GDC STAR-counts.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--all-genes", action="store_true", help="Keep all gene types instead of protein_coding only.")
    return parser


def matrix_main(argv: list[str] | None = None) -> int:
    args = build_matrix_parser().parse_args(argv)
    try:
        result = build_tpm_matrix(args.root, protein_coding_only=not args.all_genes)
    except (FileNotFoundError, GDCRNASeqError) as exc:
        print(f"Stage 2B matrix build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
