"""Complete BMC Cancer references and clean manuscript tables and figure plans."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting.stage10a_bmc_cleanup import run_stage10a  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    outputs = run_stage10a(ROOT)
    print(f"Generated cleaned BMC Cancer manuscript: {outputs['cleaned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
