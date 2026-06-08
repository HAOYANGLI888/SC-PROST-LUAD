"""Stage 0 project skeleton validation and smoke test."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scprost_luad.audit import write_audit_report
from scprost_luad.config import load_config_set, require_config_keys
from scprost_luad.paths import (
    REQUIRED_STAGE0_FILES,
    STANDARD_DIRS,
    ProjectPaths,
)


def _status_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative_dir in STANDARD_DIRS:
        path = root / relative_dir
        rows.append(
            {
                "kind": "directory",
                "relative_path": Path(relative_dir).as_posix(),
                "exists": path.is_dir(),
                "size_bytes": "",
            }
        )
    for relative_file in REQUIRED_STAGE0_FILES:
        path = root / relative_file
        rows.append(
            {
                "kind": "file",
                "relative_path": Path(relative_file).as_posix(),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.exists() else "",
            }
        )
    return rows


def _write_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kind", "relative_path", "exists", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_stage0(root: str | Path = ".") -> dict[str, Any]:
    """Validate the Stage 0 skeleton and write audit outputs."""

    project_root = Path(root).resolve()
    paths = ProjectPaths.from_root(project_root)
    paths.ensure_standard_dirs()

    config = load_config_set(project_root)
    require_config_keys(config, ["project", "paths", "data", "model"])

    rows = _status_rows(project_root)
    missing = [
        row["relative_path"]
        for row in rows
        if not row["exists"]
    ]

    manifest_path = project_root / "data" / "metadata" / "stage0_directory_manifest.csv"
    log_path = project_root / "outputs" / "logs" / "stage0_smoke_test.json"
    audit_path = project_root / "outputs" / "audit" / "stage0" / "audit_report.md"

    _write_manifest(rows, manifest_path)

    result = {
        "stage": "stage0",
        "project": config["project"]["name"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(project_root),
        "required_items_checked": len(rows),
        "missing_items": missing,
        "status": "passed" if not missing else "failed",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    issues = [
        "No biomedical data are downloaded in Stage 0 by design.",
        "If conda cannot solve GPU packages on this machine, use the CPU-first pip fallback in README.md.",
        "The Stage 1 TCGA audit entry point is documented but not implemented until Stage 1.",
    ]
    if missing:
        issues.insert(0, f"Missing required Stage 0 items: {', '.join(missing)}")

    write_audit_report(
        output_path=audit_path,
        stage="Stage 0",
        completed=[
            "Created a Windows-compatible project skeleton.",
            "Added conda and pip environment files.",
            "Added YAML configuration files for base, data, and model settings.",
            "Added data directory rules, manifest templates, and manual download guidance.",
            "Added reusable Python helpers for configs, paths, audits, and Stage 0 validation.",
            "Added command-line smoke test and pytest coverage.",
        ],
        input_files=[
            "README.md",
            "AGENTS.md",
            "environment.yml",
            "requirements.txt",
            "configs/base.yaml",
            "configs/data.yaml",
            "configs/model.yaml",
        ],
        output_files=[
            "data/metadata/stage0_directory_manifest.csv",
            "outputs/logs/stage0_smoke_test.json",
            "outputs/audit/stage0/audit_report.md",
        ],
        commands=[
            "conda env create -f environment.yml",
            "conda activate gpu_py310",
            "python -m pip install -e .",
            "python scripts/run_stage0_smoke_test.py --root .",
            "python -m pytest tests -q",
        ],
        potential_issues=issues,
        next_steps=[
            "Stage 1 should implement TCGA-LUAD data availability auditing.",
            "Stage 1 should output patient-level modality matching tables under data/metadata/.",
            "Stage 1 should create its own smoke test and audit report.",
        ],
        metadata={
            "required_items_checked": len(rows),
            "missing_items": len(missing),
            "status": result["status"],
        },
    )

    if missing:
        raise RuntimeError(
            "Stage 0 smoke test failed. See outputs/audit/stage0/audit_report.md"
        )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Stage 0 project skeleton smoke test.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root. Defaults to the current directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_stage0(args.root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
