"""Build the TCGA-LUAD diagnostic WSI manifest and availability audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.wsi_manifest import build_wsi_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit TCGA-LUAD diagnostic SVS availability from the GDC API.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--small-test", action="store_true", help="Build an isolated toy manifest without network access.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not (Path(args.root) / args.config).exists():
        raise SystemExit(f"Config file not found: {Path(args.root) / args.config}")
    try:
        result = build_wsi_manifest(args.root, small_test=args.small_test)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Stage 6A WSI manifest failed: {exc}") from exc
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

