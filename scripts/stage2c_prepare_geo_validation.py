"""Prepare local GEO cohorts for future frozen-model external validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.geo_validation import prepare_geo_validation_readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and prepare GEO expression + OS files for fixed-model validation.")
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--collapse-strategy", choices=["mean", "max_variance"], default="mean")
    parser.add_argument("--small-test", action="store_true", help="Run with a generated miniature GEO fixture.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (Path(args.root) / args.config).exists():
        parser.error(f"Config file not found: {Path(args.root) / args.config}")
    try:
        result = prepare_geo_validation_readiness(
            args.root,
            collapse_strategy=args.collapse_strategy,
            small_test=args.small_test,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Stage 2C GEO readiness failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
