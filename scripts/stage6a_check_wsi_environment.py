"""Check the Stage 6A-Fix Windows WSI environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.env_check import check_wsi_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit OpenSlide, torch, torchvision and real-SVS reading for Stage 6A-Fix.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--download-pretrained-weights", action="store_true", help="Explicitly allow torchvision to fetch uncached ResNet50 weights.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = check_wsi_environment(args.root, download_pretrained_weights=args.download_pretrained_weights)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
