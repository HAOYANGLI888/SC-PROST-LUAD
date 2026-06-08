"""Summarize Stage 2D external validation and Stage 3 go/no-go decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.external_validation import summarize_external_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Stage 2D reports from real saved outputs without changing the frozen model."
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--small-test", action="store_true", help="Generate an isolated toy report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (Path(args.root) / args.config).exists():
        parser.error(f"Config file not found: {Path(args.root) / args.config}")
    try:
        result = summarize_external_validation(args.root, small_test=args.small_test)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Stage 2D summary failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

