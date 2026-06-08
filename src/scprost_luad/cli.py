"""Top-level command-line interface."""

from __future__ import annotations

import argparse
import json

from scprost_luad.stages.stage0 import run_stage0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scprost-luad",
        description="SC-PROST-LUAD research workflow CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    stage0 = subparsers.add_parser(
        "stage0-smoke",
        help="Validate the Stage 0 project skeleton and write an audit report.",
    )
    stage0.add_argument(
        "--root",
        default=".",
        help="Project root. Defaults to the current directory.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "stage0-smoke":
        result = run_stage0(args.root)
        print(json.dumps(result, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
