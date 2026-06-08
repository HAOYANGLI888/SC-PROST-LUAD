"""Generate the Journal of Translational Medicine main manuscript DOCX."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting.stage9_jtm import append_stage9_audit, generate_main_manuscript  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    output = generate_main_manuscript(ROOT)
    append_stage9_audit(
        ROOT,
        "\n## Stage 9 JTM Main Manuscript\n\n"
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
        f"- Output: `{output}`.\n"
        "- Scope: manuscript preparation only; no model training or new analysis.\n",
    )
    print(f"Generated JTM main manuscript: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
