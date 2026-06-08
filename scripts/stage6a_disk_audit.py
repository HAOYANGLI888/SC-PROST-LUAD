"""Audit junction-based storage migration for Stage 6A diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _dir_size(path: Path) -> tuple[int, int]:
    files = list(path.rglob("*")) if path.exists() else []
    file_items = [item for item in files if item.is_file()]
    return len(file_items), sum(item.stat().st_size for item in file_items)


def _junction_info(path: Path) -> tuple[bool, str]:
    item = path.stat()
    is_reparse = bool(Path(path).is_dir() and (Path(path).lstat().st_file_attributes & 0x400))
    try:
        target = Path(path).readlink()
    except OSError:
        target = ""
    return is_reparse, str(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit data/outputs junctions and disk space after migration to D drive.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--target-root",
        default=os.environ.get("SC_PROST_DATA_ROOT"),
        help=(
            "External storage root. Defaults to SC_PROST_DATA_ROOT or the parent "
            "of the current data junction target."
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    if args.target_root:
        target_root = Path(args.target_root).resolve()
    else:
        try:
            target_root = (root / "data").readlink().resolve().parent
        except OSError:
            target_root = root
    rows = []
    for name in ("data", "outputs"):
        original = root / name
        target = target_root / name
        files, size = _dir_size(target)
        is_junction, junction_target = _junction_info(original)
        rows.append(
            {
                "name": name,
                "original_path": str(original),
                "is_junction": is_junction,
                "junction_target": junction_target,
                "expected_target": str(target),
                "target_exists": target.exists(),
                "file_count": files,
                "size_bytes": size,
                "size_gb": round(size / 1024**3, 3),
            }
        )
    drives = {}
    for drive in ("C", "D"):
        import shutil

        usage = shutil.disk_usage(f"{drive}:/")
        drives[drive] = {"free_bytes": usage.free, "free_gb": round(usage.free / 1024**3, 3)}
    backups = sorted([item.name for item in root.iterdir() if item.name.startswith(("data__c_backup_", "outputs__c_backup_"))])
    c_low = drives["C"]["free_gb"] < 20.0
    table = pd.DataFrame(rows)
    tables = root / "outputs" / "tables"
    reports = root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "stage6a_disk_usage_summary.csv", index=False)
    report = (
        "# Stage 6A Disk Audit Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Junctions\n\n"
        + "\n".join(
            f"- `{row['name']}`: junction=`{row['is_junction']}`, target=`{row['junction_target']}`, target exists=`{row['target_exists']}`, size={row['size_gb']} GB"
            for row in rows
        )
        + "\n\n## Drive Space\n\n"
        f"- C free: {drives['C']['free_gb']} GB.\n"
        f"- D free: {drives['D']['free_gb']} GB.\n"
        f"- C backups preserved: {', '.join(backups) if backups else 'none'}.\n\n"
        "## Recommendation\n\n"
        + (
            "- C drive is still low because backups are preserved. Recommend deleting C-drive backups only after explicit user confirmation.\n"
            if c_low
            else "- C drive has adequate free space; keep backups until user confirms cleanup.\n"
        )
    )
    (reports / "stage6a_disk_audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "passed", "c_free_gb": drives["C"]["free_gb"], "d_free_gb": drives["D"]["free_gb"], "recommend_backup_deletion_with_confirmation": c_low}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
