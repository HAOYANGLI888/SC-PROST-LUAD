"""Select Stage 5 protein/IHC candidate genes from Stage 4 mechanisms."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.protein_candidate_selection import write_candidate_outputs  # noqa: E402


def _audit_path(small_test: bool) -> Path:
    return ROOT / "outputs" / "audit" / "stage5_small_test" / "audit_report.md" if small_test else ROOT / "audit_report.md"


def _append_audit(path: Path, *, output: Path, small_test: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Stage 5 Candidate Gene Selection\n\n"
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
            f"- Mode: `{'toy_small_test' if small_test else 'formal'}`.\n"
            f"- Output: `{output}`.\n"
            "- Integrity: candidates were selected from Stage 4 mechanisms, not from protein outcomes.\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--genes-per-layer", type=int, default=10, help="Maximum candidates per mechanism layer.")
    parser.add_argument("--small-test", action="store_true", help="Run isolated small-test outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = write_candidate_outputs(ROOT, small_test=args.small_test, genes_per_layer=args.genes_per_layer)
    _append_audit(_audit_path(args.small_test), output=output, small_test=args.small_test)
    print(f"Stage 5 candidate genes written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

