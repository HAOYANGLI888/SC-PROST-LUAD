"""Validate Stage 5 candidate proteins in CPTAC/PDC abundance data."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.cptac_data_import import stage5b_paths  # noqa: E402
from validation.cptac_protein_analysis import run_candidate_protein_validation  # noqa: E402


def _audit(small_test: bool) -> Path:
    return ROOT / "outputs" / "audit" / "stage5b_small_test" / "audit_report.md" if small_test else ROOT / "audit_report.md"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--small-test", action="store_true", help="Run isolated small-test validation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = stage5b_paths(ROOT, small_test=args.small_test)
    result = run_candidate_protein_validation(paths, small_test=args.small_test)
    _audit(args.small_test).parent.mkdir(parents=True, exist_ok=True)
    with _audit(args.small_test).open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Stage 5B Candidate Protein Validation\n\n"
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
            f"- Mode: `{'toy_small_test' if args.small_test else 'formal'}`.\n"
            f"- Status: `{result['status']}`.\n"
            f"- Candidate proteins available: `{result['candidate_available']}`.\n"
            f"- Module count: `{result['module_count']}`.\n"
        )
    print(f"Stage 5B candidate protein validation complete: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

