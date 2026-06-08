"""Prepare one GEO cohort and apply the frozen TCGA preprocessing objects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.geo_download import GEO_COHORTS
from validation.external_validation import prepare_geo_cohort, prepare_geo_cohort_small_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one GEO cohort for frozen TCGA PCA_25 + clinical Cox validation."
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--cohort", required=True, choices=sorted(GEO_COHORTS))
    parser.add_argument(
        "--collapse-strategy",
        choices=["mean", "median", "max_variance"],
        default="mean",
        help="Deterministic multi-probe aggregation rule.",
    )
    parser.add_argument("--small-test", action="store_true", help="Generate an isolated miniature cohort.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (Path(args.root) / args.config).exists():
        parser.error(f"Config file not found: {Path(args.root) / args.config}")
    try:
        result = (
            prepare_geo_cohort_small_test(args.root, args.cohort)
            if args.small_test
            else prepare_geo_cohort(
                args.root, args.cohort, collapse_strategy=args.collapse_strategy
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Stage 2D GEO cohort preparation failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

