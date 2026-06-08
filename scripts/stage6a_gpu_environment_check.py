"""Run the Stage 6A GPU environment check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.gpu_env import check_gpu_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify CUDA-enabled PyTorch for Stage 6A-GPU-Pilot.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = check_gpu_environment(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

