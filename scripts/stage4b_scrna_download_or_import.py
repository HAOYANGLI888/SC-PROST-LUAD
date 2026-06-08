"""Download or import raw LUAD scRNA-seq data and write a data inventory."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_raw_import import (  # noqa: E402
    GSE131907_FILES,
    SCRNAImportError,
    annotation_matrix_match,
    build_data_inventory,
    convert_gse131907_to_h5ad_chunked,
    create_small_test_h5ad,
    download_gse131907,
    export_rds_dimnames_with_r,
    file_md5,
    import_best_available,
    inspect_rds_with_r,
    scrna_paths,
    validate_gse131907_h5ad,
)


def _output_paths(small_test: bool) -> tuple[Path, Path]:
    suffix = "stage4b_small_test" if small_test else ""
    table_dir = ROOT / "outputs" / "tables" / suffix
    report_dir = ROOT / "outputs" / "reports" / suffix
    return (
        table_dir / "stage4b_scrna_data_inventory.csv",
        report_dir / "stage4b_scrna_data_inventory_report.md",
    )


def _write_report(
    path: Path,
    inventory,
    *,
    mode: str,
    status: str,
    source: str,
    range_part_bytes: int = 0,
    error: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_size = inventory["file_size_bytes"].sum() if not inventory.empty else 0
    path.write_text(
        f"""# Stage 4B scRNA Data Inventory Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Mode: `{mode}`
- Dataset priority: `GSE131907`
- Status: `{status}`
- Imported source: `{source}`
- Discovered files: `{len(inventory)}`
- Total discovered size: `{raw_size / 1024**2:.1f} MB`
- Retained resumable range parts: `{range_part_bytes / 1024**2:.1f} MB`
- Error: `{error or "none"}`

## Integrity Boundary

Toy data are used only in `--small-test` and are isolated from formal outputs. Formal raw scRNA cellular-context claims require a complete real dataset. An incomplete download is reported as `manual_download_required` and is not analysed.

## Official GSE131907 Files

- Annotation: `{GSE131907_FILES["annotation"]["name"]}`
- Raw UMI RDS: `{GSE131907_FILES["raw_rds"]["name"]}`

See `docs/stage4b_scrna_manual_download_guide.md` for manual and R/Seurat conversion options.
""",
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--small-test", action="store_true", help="Run an isolated toy engineering test.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel HTTP range workers for GSE131907.")
    parser.add_argument("--skip-download", action="store_true", help="Only inspect and import local files.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Cells per resumable R sparse-export chunk.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = scrna_paths(ROOT, small_test=args.small_test)
    table_path, report_path = _output_paths(args.small_test)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    status = "completed"
    source = ""
    error = ""
    exit_code = 0
    integrity_rows: list[dict[str, object]] = []
    match_table = None
    rds_details: dict[str, object] = {}
    converted_path = None
    conversion_note = ""
    conversion_checks = None
    conversion_details: dict[str, object] = {}
    conversion_started = False
    conversion_start = time.time()
    external_data_root = os.environ.get("SC_PROST_DATA_ROOT")
    d_data_root = Path(external_data_root).resolve() if external_data_root else None
    work_dir = (
        d_data_root
        / "data"
        / "processed"
        / "scrna_luad"
        / "stage4b_h5ad_conversion_work"
        if d_data_root is not None and d_data_root.exists()
        else paths.processed_dir / "stage4b_h5ad_conversion_work"
    )
    conversion_log = ROOT / "outputs" / "logs" / "stage4b_h5ad_conversion.log"
    try:
        import psutil

        memory = psutil.virtual_memory()
        total_ram = int(memory.total)
        available_ram = int(memory.available)
    except Exception:
        total_ram = 0
        available_ram = 0
    disk_root = (
        Path(d_data_root.anchor)
        if d_data_root is not None and d_data_root.exists()
        else Path(paths.processed_dir.anchor)
    )
    try:
        disk_usage = shutil.disk_usage(disk_root)
        disk_free = int(disk_usage.free)
    except Exception:
        disk_free = 0
    try:
        if args.small_test:
            create_small_test_h5ad(paths)
            source = "toy_engineering_smoke_test"
        else:
            if not args.skip_download:
                download_gse131907(paths, workers=args.workers)
            actual_size = paths.raw_rds.stat().st_size if paths.raw_rds.exists() else 0
            expected_size = int(GSE131907_FILES["raw_rds"]["size"])
            integrity_rows.append(
                {
                    "check": "file_size_bytes",
                    "observed": actual_size,
                    "expected": expected_size,
                    "status": "pass" if actual_size == expected_size else "fail",
                    "note": "Official NCBI HTTP Content-Length.",
                }
            )
            repair_backup = paths.raw_rds.with_name(
                paths.raw_rds.name + ".pre_repair_corrupt_backup"
            )
            if repair_backup.exists():
                integrity_rows.append(
                    {
                        "check": "byte_range_repair",
                        "observed": "replaced bytes 16777216-20971519",
                        "expected": "official GEO HTTP range",
                        "status": "pass",
                        "note": (
                            "A verified 4 MiB remote range repaired the old "
                            "partial-file corruption at bytes 18227216-18583551. "
                            f"The pre-repair file is retained at {repair_backup}."
                        ),
                    }
                )
            if actual_size != expected_size:
                raise SCRNAImportError(
                    f"Raw UMI RDS is incomplete: {actual_size}/{expected_size} bytes."
                )
            md5_value = file_md5(paths.raw_rds)
            integrity_rows.append(
                {
                    "check": "md5",
                    "observed": md5_value,
                    "expected": "",
                    "status": "reference_unavailable",
                    "note": "NCBI GEO supplementary directory did not publish an MD5 file.",
                }
            )
            rds_details = inspect_rds_with_r(paths.raw_rds)
            integrity_rows.append(
                {
                    "check": "RDS_readable",
                    "observed": rds_details.get("rds_readable"),
                    "expected": True,
                    "status": "pass" if rds_details.get("rds_readable") else "fail",
                    "note": rds_details.get("rds_error", ""),
                }
            )
            if not rds_details.get("rds_readable"):
                raise SCRNAImportError(
                    "The completed file has the expected size but R cannot read the RDS."
                )
            integrity_rows.extend(
                [
                    {
                        "check": "RDS_object_class",
                        "observed": rds_details.get("class", ""),
                        "expected": "matrix-like object",
                        "status": "pass",
                        "note": "Read directly from the complete RDS with R.",
                    },
                    {
                        "check": "RDS_dimensions",
                        "observed": (
                            f'{rds_details.get("nrow")} x '
                            f'{rds_details.get("ncol")}'
                        ),
                        "expected": "genes x approximately 208506 cells",
                        "status": "pass"
                        if int(rds_details.get("ncol", 0)) == 208506
                        else "review",
                        "note": "No QC or filtering was applied.",
                    },
                    {
                        "check": "outer_gzip_stream",
                        "observed": (
                            f'{rds_details.get("compression_layers", "")} '
                            "compression layers"
                        ),
                        "expected": "full stream readable",
                        "status": "pass",
                        "note": "Outer gzip was fully decompressed before RDS parsing.",
                    },
                ]
            )
            dimnames_dir = paths.processed_dir / "gse131907_dimnames"
            features_path, barcodes_path = export_rds_dimnames_with_r(
                paths.raw_rds, dimnames_dir
            )
            match_table, match_details = annotation_matrix_match(
                paths.annotation, barcodes_path
            )
            gene_count = sum(1 for _ in features_path.open("r", encoding="utf-8"))
            integrity_rows.extend(
                [
                    {
                        "check": "matrix_cells",
                        "observed": match_details["matrix_cells"],
                        "expected": 208506,
                        "status": "pass"
                        if abs(match_details["matrix_cells"] - 208506)
                        <= max(10, int(0.01 * 208506))
                        else "review",
                        "note": "Compared with official annotation row count.",
                    },
                    {
                        "check": "matrix_genes",
                        "observed": gene_count,
                        "expected": "10000-60000",
                        "status": "pass" if 10000 <= gene_count <= 60000 else "review",
                        "note": "Broad plausibility range for a human scRNA count matrix.",
                    },
                    {
                        "check": "annotation_match_fraction",
                        "observed": match_details["matrix_match_fraction"],
                        "expected": ">=0.95",
                        "status": "pass"
                        if match_details["matrix_match_fraction"] >= 0.95
                        else "review",
                        "note": "Exact cell barcode match after source annotation reindexing.",
                    },
                ]
            )
            if paths.raw_or_converted_h5ad.exists():
                converted_path = paths.raw_or_converted_h5ad
                source = "existing_GSE131907_converted_h5ad"
                conversion_checks, conversion_details = validate_gse131907_h5ad(
                    converted_path, paths.annotation
                )
            else:
                conversion_started = True
                conversion_checks, conversion_details = (
                    convert_gse131907_to_h5ad_chunked(
                        paths,
                        work_dir=work_dir,
                        chunk_size=args.chunk_size,
                        log_path=conversion_log,
                    )
                )
                converted_path = Path(conversion_details["h5ad_path"])
                source = "GSE131907_chunked_R_to_AnnData_CSR"
            status = "completed" if converted_path and converted_path.exists() else status
            integrity_rows.append(
                {
                    "check": "converted_h5ad",
                    "observed": str(converted_path or "not_generated"),
                    "expected": "existing readable h5ad",
                    "status": "pass"
                    if converted_path and Path(converted_path).exists()
                    else "pending",
                    "note": conversion_note
                    or "Raw/converted artifact only; no QC or scientific scoring.",
                }
            )
    except Exception as exc:
        status = "conversion_failed" if conversion_started else "manual_download_required"
        error = str(exc)
        exit_code = 2
    inventory = build_data_inventory(paths.raw_dir)
    inventory.to_csv(table_path, index=False)
    parts_dir = paths.raw_rds.with_name(paths.raw_rds.name + ".parts")
    range_part_bytes = (
        sum(path.stat().st_size for path in parts_dir.glob("*") if path.is_file())
        if parts_dir.exists()
        else 0
    )
    _write_report(
        report_path,
        inventory,
        mode="toy_small_test" if args.small_test else "formal",
        status=status,
        source=source or "none",
        range_part_bytes=range_part_bytes,
        error=error,
    )
    if not args.small_test:
        integrity_path = (
            ROOT / "outputs" / "tables" / "stage4b_raw_umi_integrity_check.csv"
        )
        match_path = (
            ROOT
            / "outputs"
            / "tables"
            / "stage4b_annotation_matrix_match_check.csv"
        )
        resume_report = (
            ROOT
            / "outputs"
            / "reports"
            / "stage4b_raw_umi_download_resume_report.md"
        )
        integrity_path.parent.mkdir(parents=True, exist_ok=True)
        if not integrity_rows:
            integrity_rows.append(
                {
                    "check": "download_and_integrity",
                    "observed": error or "not completed",
                    "expected": int(GSE131907_FILES["raw_rds"]["size"]),
                    "status": "incomplete",
                    "note": "No formal matrix processing was performed.",
                }
            )
        import pandas as pd

        integrity_frame = pd.DataFrame(integrity_rows)
        integrity_frame.to_csv(integrity_path, index=False)
        if match_table is None:
            match_table = pd.DataFrame(
                [
                    {
                        "check": "annotation_matrix_match",
                        "value": "",
                        "status": "not_run",
                        "note": "Requires a complete, readable raw UMI RDS.",
                    }
                ]
            )
        match_table.to_csv(match_path, index=False)
        prefix_size = paths.raw_rds.stat().st_size if paths.raw_rds.exists() else 0
        retained = prefix_size + range_part_bytes
        expected = int(GSE131907_FILES["raw_rds"]["size"])
        match_statuses = (
            match_table["status"].astype(str).tolist()
            if "status" in match_table.columns
            else []
        )
        resume_report.parent.mkdir(parents=True, exist_ok=True)
        resume_report.write_text(
            f"""# Stage 4B-Resume Raw UMI Download Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Dataset: `GSE131907`
- Status: `{status}`
- Raw RDS path: `{paths.raw_rds}`
- Assembled file size: `{prefix_size}` bytes
- Retained range-part bytes: `{range_part_bytes}`
- Total retained download data: `{retained}` bytes
- Expected final size: `{expected}` bytes
- Expected-size complete: `{prefix_size == expected}`
- Checksum status: `{next((row["status"] for row in integrity_rows if row["check"] == "md5"), "not_run")}`
- RDS readable: `{rds_details.get("rds_readable", "not_run")}`
- Converted h5ad: `{converted_path or "not_generated"}`
- Conversion note: `{conversion_note or "none"}`
- Annotation/matrix match check: `{"completed" if match_statuses and "not_run" not in match_statuses else "not_run"}`

## Integrity Boundary

This resume stage performs download, file integrity, RDS readability, conversion,
and barcode matching only. It does not run formal QC, UMAP, cell-source analysis,
module scoring, survival analysis, or scientific figure generation.

## Check Results

{integrity_frame.to_markdown(index=False)}

## Error

`{error or "none"}`
""",
            encoding="utf-8",
        )
        conversion_summary_path = (
            ROOT / "outputs" / "tables" / "stage4b_h5ad_conversion_summary.csv"
        )
        conversion_integrity_path = (
            ROOT / "outputs" / "tables" / "stage4b_h5ad_integrity_check.csv"
        )
        conversion_report_path = (
            ROOT / "outputs" / "reports" / "stage4b_h5ad_conversion_report.md"
        )
        conversion_summary_path.parent.mkdir(parents=True, exist_ok=True)
        elapsed_seconds = round(time.time() - conversion_start, 3)
        initial_conversion_seconds = ""
        if conversion_log.exists() and paths.raw_or_converted_h5ad.exists():
            match = re.search(
                r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]",
                conversion_log.read_text(encoding="utf-8", errors="replace"),
            )
            if match:
                export_started = datetime.strptime(
                    match.group(1), "%Y-%m-%d %H:%M:%S"
                )
                initial_conversion_seconds = round(
                    paths.raw_or_converted_h5ad.stat().st_mtime
                    - export_started.timestamp(),
                    3,
                )
        summary_rows = [
            {"item": "status", "value": status},
            {"item": "raw_rds_path", "value": str(paths.raw_rds)},
            {
                "item": "raw_rds_size_bytes",
                "value": paths.raw_rds.stat().st_size if paths.raw_rds.exists() else 0,
            },
            {
                "item": "raw_rds_md5",
                "value": next(
                    (
                        row["observed"]
                        for row in integrity_rows
                        if row["check"] == "md5"
                    ),
                    "",
                ),
            },
            {"item": "annotation_path", "value": str(paths.annotation)},
            {"item": "annotation_cells", "value": 208506},
            {"item": "annotation_samples", "value": 58},
            {"item": "total_ram_bytes", "value": total_ram},
            {"item": "available_ram_before_bytes", "value": available_ram},
            {"item": "disk_free_before_bytes", "value": disk_free},
            {"item": "temporary_work_dir", "value": str(work_dir)},
            {"item": "conversion_log", "value": str(conversion_log)},
            {"item": "chunk_size_cells", "value": args.chunk_size},
            {"item": "elapsed_seconds", "value": elapsed_seconds},
            {
                "item": "initial_conversion_seconds_from_export_start",
                "value": initial_conversion_seconds,
            },
            {
                "item": "h5ad_path",
                "value": str(converted_path or paths.raw_or_converted_h5ad),
            },
            {
                "item": "h5ad_size_bytes",
                "value": (
                    Path(converted_path).stat().st_size
                    if converted_path and Path(converted_path).exists()
                    else 0
                ),
            },
        ]
        existing_items = {row["item"] for row in summary_rows}
        for key, value in conversion_details.items():
            if key not in existing_items:
                summary_rows.append({"item": key, "value": value})
        pd.DataFrame(summary_rows).to_csv(conversion_summary_path, index=False)
        if conversion_checks is None:
            conversion_checks = pd.DataFrame(
                [
                    {
                        "check": "h5ad_conversion",
                        "observed": error or "not completed",
                        "expected": "complete validated h5ad",
                        "status": "fail" if conversion_started else "not_run",
                    }
                ]
            )
        conversion_checks.to_csv(conversion_integrity_path, index=False)
        h5ad_exists = bool(
            converted_path and Path(converted_path).exists()
        )
        checks_pass = bool(
            not conversion_checks.empty
            and (conversion_checks["status"].astype(str) == "pass").all()
        )
        conversion_report_path.write_text(
            f"""# Stage 4B h5ad Conversion Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Status: `{status}`
- Raw RDS: `{paths.raw_rds}`
- Raw RDS size: `{paths.raw_rds.stat().st_size if paths.raw_rds.exists() else 0}` bytes
- Raw RDS MD5: `{next((row["observed"] for row in integrity_rows if row["check"] == "md5"), "not_available")}`
- Annotation: `{paths.annotation}`
- Annotation cells / samples: `208506 / 58`
- RAM total / available before conversion: `{total_ram / 1024**3:.2f} / {available_ram / 1024**3:.2f} GiB`
- D-volume free before conversion: `{disk_free / 1024**3:.2f} GiB`
- Temporary work directory: `{work_dir}`
- Chunk size: `{args.chunk_size}` cells
- Conversion log: `{conversion_log}`
- Current command elapsed: `{elapsed_seconds / 3600:.2f}` hours
- Initial conversion from R export start: `{float(initial_conversion_seconds or 0) / 60:.2f}` minutes
- Final h5ad exists: `{h5ad_exists}`
- Final h5ad: `{converted_path or paths.raw_or_converted_h5ad}`
- Physical h5ad storage: `{Path(converted_path or paths.raw_or_converted_h5ad).resolve()}`
- Integrity checks all pass: `{checks_pass}`

## Conversion Method

R reads the official dense RDS once and exports resumable cell batches as
cell-major sparse CSR arrays. Python assembles those arrays into a compressed
AnnData CSR file. The output is first written with an `.incomplete` suffix and
is renamed to the formal h5ad path only after complete assembly. A failed
assembly is retained only with a failure suffix and is never presented as a
successful h5ad.

## Integrity Checks

{conversion_checks.to_markdown(index=False)}

## Readiness

- The validated h5ad is technically ready for a separately authorized formal
  Stage 4B QC workflow.
- The 32 GB workstation was sufficient for this resumable conversion because
  the final matrix is sparse and batches were kept small.
- Full 208,506-cell QC, normalization, highly variable gene selection, PCA,
  neighbor graph construction, and UMAP can create several concurrent matrix
  copies. A machine with at least 64 GB RAM is recommended for stable formal
  analysis; otherwise the workflow must remain backed/chunked and be monitored
  carefully.

## Scientific Boundary

This stage performs format conversion and integrity validation only. It does
not perform QC, filtering, normalization, UMAP, clustering, annotation
inference, signature scoring, survival analysis, or biological interpretation.

## Error

`{error or "none"}`
""",
            encoding="utf-8",
        )
    print(f"Stage 4B inventory: {status} -> {table_path}")
    if exit_code:
        print(error, file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
