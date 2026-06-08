"""Extract CUDA ResNet50 features for the Stage 6A WSI pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.pilot_feature_pipeline import extract_wsi_pilot_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract strict torchvision pretrained ResNet50 WSI pilot features on CUDA.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--backend", default="resnet50", choices=["resnet50"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    frame = extract_wsi_pilot_features(args.root, backend=args.backend, batch_size=args.batch_size, allow_cpu=args.allow_cpu)
    ok = frame["feature_status"].isin(["extracted", "skipped_existing"]).sum()
    print(json.dumps({"status": "passed", "slides_with_features": int(ok), "used_cuda": bool(frame["used_cuda"].any())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
