"""Run the strict Stage 6A-Fix real-SVS smallset closed loop."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume and process only the selected Stage 6A real 20-slide WSI smallset.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--max-patches", type=int, default=128, help="Engineering-feasibility cap per slide; accepted maximum is 1000.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--download-pretrained-weights", action="store_true")
    return parser


def _run(root: Path, *parts: str) -> None:
    command = [sys.executable, *parts]
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    config = str(args.config)
    env_command = ["scripts/stage6a_check_wsi_environment.py", "--config", config]
    if args.download_pretrained_weights:
        env_command.append("--download-pretrained-weights")
    _run(root, *env_command)
    _run(root, "scripts/stage6a_resume_wsi_smallset.py", "--config", config, "--workers", str(args.workers))
    _run(root, "scripts/stage6a_extract_wsi_patches.py", "--config", config, "--smallset", "--real-svs", "--patch-size", str(args.patch_size), "--max-patches", str(args.max_patches))
    _run(root, "scripts/stage6a_extract_patch_features.py", "--config", config, "--smallset", "--real-svs", "--encoder", "resnet50", "--batch-size", str(args.batch_size), "--device", args.device)
    _run(root, "scripts/train_stage6a_pathology_mil.py", "--config", config, "--smallset", "--real-features", "--epochs", str(args.epochs))
    _run(root, "scripts/evaluate_stage6a_pathology_mil.py", "--config", config, "--smallset", "--real-features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
