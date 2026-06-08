"""Generate the Stage 8 manuscript skeleton in Markdown and DOCX."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting.stage8 import append_stage8_audit, write_manuscript_skeleton  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--no-docx", action="store_true", help="Skip optional DOCX generation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = write_manuscript_skeleton(ROOT, create_docx=not args.no_docx)
    append_stage8_audit(
        ROOT,
        "\n## Stage 8 Manuscript Skeleton\n\n"
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
        f"- Markdown: `{result['markdown']}`.\n"
        f"- DOCX created: `{result['docx_created']}`.\n"
        "- Status: skeleton only; not a submission-ready final manuscript.\n",
    )
    print(f"Stage 8 manuscript skeleton generated: {result['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

