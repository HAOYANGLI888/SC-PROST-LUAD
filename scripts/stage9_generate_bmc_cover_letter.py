"""Generate the BMC Cancer cover letter DOCX."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting.stage9_bmc import generate_cover_letter  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    print(f"Generated BMC Cancer cover letter: {generate_cover_letter(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
