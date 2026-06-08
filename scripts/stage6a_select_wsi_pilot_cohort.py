"""Select the deterministic 100-slide Stage 6A WSI pilot cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.wsi_pilot_selection import select_wsi_pilot_cohort


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select a deterministic clinical+OS+WSI TCGA-LUAD pilot cohort.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--n-slides", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    frame = select_wsi_pilot_cohort(args.root, n_slides=args.n_slides, seed=args.seed)
    print(json.dumps({"status": "passed", "slides": len(frame), "patients": int(frame["patient_id"].nunique()), "expected_gb": float(frame["expected_size"].sum() / 1e9)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

