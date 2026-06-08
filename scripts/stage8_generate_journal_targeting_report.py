"""Generate the Stage 8 journal targeting strategy report."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting.stage8 import append_stage8_audit, write_journal_targeting_report  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    output = write_journal_targeting_report(ROOT)
    append_stage8_audit(
        ROOT,
        "\n## Stage 8 Journal Targeting\n\n"
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
        f"- Report: `{output}`.\n"
        "- Current recommendation: BMC Medical Genomics as the most realistic first target.\n",
    )
    print(f"Stage 8 journal targeting report generated: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

