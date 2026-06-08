"""Run Stage 5 Human Protein Atlas orthogonal evidence collection."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.hpa_validation import run_hpa_validation  # noqa: E402


def _audit_path(small_test: bool) -> Path:
    return ROOT / "outputs" / "audit" / "stage5_small_test" / "audit_report.md" if small_test else ROOT / "audit_report.md"


def _append_audit(path: Path, result: dict, *, small_test: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Stage 5 HPA Validation\n\n"
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
            f"- Mode: `{'toy_small_test' if small_test else 'formal'}`.\n"
            f"- Candidate count: `{result['candidate_count']}`.\n"
            f"- Evidence table: `{result['evidence']}`.\n"
            f"- IHC links: `{result['links']}`.\n"
            "- Integrity: HPA evidence is qualitative orthogonal support, not causal confirmation.\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--timeout", type=int, default=30, help="HPA request timeout in seconds.")
    parser.add_argument("--small-test", action="store_true", help="Run isolated small-test outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_hpa_validation(ROOT, small_test=args.small_test, timeout=args.timeout)
    _append_audit(_audit_path(args.small_test), result, small_test=args.small_test)
    print(f"Stage 5 HPA validation complete: {result['evidence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

