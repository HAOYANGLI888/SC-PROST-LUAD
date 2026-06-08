"""Run frozen-model GEO validation and TCGA fold-local IPCW metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.external_validation import run_external_validation, run_tcga_ipcw_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the frozen TCGA model to prepared GEO cohorts and compute Stage 2D metrics."
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--small-test", action="store_true", help="Run isolated toy metric checks only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (Path(args.root) / args.config).exists():
        parser.error(f"Config file not found: {Path(args.root) / args.config}")
    try:
        external = run_external_validation(args.root, small_test=args.small_test)
        result = {"external_validation": external}
        if not args.small_test:
            result["tcga_ipcw"] = run_tcga_ipcw_metrics(args.root)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Stage 2D validation failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

