"""Extract real tissue patch coordinates for the Stage 6A WSI pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.pilot_patch_pipeline import extract_wsi_pilot_patches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract OpenSlide tissue patch indexes for the WSI pilot.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--tissue-threshold", type=float, default=0.35)
    parser.add_argument("--max-patches-per-slide", type=int, default=512)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    frame = extract_wsi_pilot_patches(
        args.root,
        patch_size=args.patch_size,
        tissue_threshold=args.tissue_threshold,
        max_patches_per_slide=args.max_patches_per_slide,
    )
    print(json.dumps({"status": "passed", "slides_extracted": int(frame["patch_status"].eq("extracted").sum()), "slides": len(frame)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

