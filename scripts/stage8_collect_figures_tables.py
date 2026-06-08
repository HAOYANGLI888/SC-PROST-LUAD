"""Generate Stage 8 figure/table plans and draft legends."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting.stage8 import append_stage8_audit, write_figure_table_outputs  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    result = write_figure_table_outputs(ROOT)
    append_stage8_audit(
        ROOT,
        "\n## Stage 8 Figure And Table Assembly Plan\n\n"
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
        f"- Figure plan: `{result['figure_plan']}`.\n"
        f"- Table plan: `{result['table_plan']}`.\n"
        "- Existing source panels were inventoried; final multi-panel artwork remains to be assembled.\n",
    )
    print(f"Stage 8 figure/table plans generated: {result['figure_plan']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

