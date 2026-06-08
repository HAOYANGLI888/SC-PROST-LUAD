"""Generate Stage 8 results, discussion, and limitations drafts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting.stage8 import append_stage8_audit, write_results_outputs  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    result = write_results_outputs(ROOT)
    append_stage8_audit(
        ROOT,
        "\n## Stage 8 Results Narrative\n\n"
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
        f"- Results: `{result['results']}`.\n"
        f"- Discussion: `{result['discussion']}`.\n"
        f"- Limitations: `{result['limitations']}`.\n"
        "- Evidence values were loaded from existing Stage 2C/2D/4/5/5B/6A outputs.\n",
    )
    print(f"Stage 8 results narrative generated: {result['results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

