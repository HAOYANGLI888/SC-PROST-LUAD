"""Train Stage 6A pathology proof-of-concept MIL models."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.pathology_visualization import plot_attention_coordinates
from training.pathology_train import train_pathology_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage 6A small-test or smallset pathology MIL proof-of-concept.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--small-test", action="store_true")
    parser.add_argument("--smallset", action="store_true")
    parser.add_argument("--real-features", action="store_true", help="Train only on strict real-SVS pretrained feature bags.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.real_features and (args.small_test or not args.smallset):
        raise SystemExit("--real-features requires --smallset and cannot be combined with --small-test.")
    try:
        result = train_pathology_models(args.root, small_test=args.small_test, smallset=args.smallset, epochs=args.epochs)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Stage 6A MIL training failed: {exc}") from exc
    figure_name = "stage6a_real_attention_heatmap_example.png" if args.real_features else "stage6a_pathology_attention_heatmap_example.png"
    plot_attention_coordinates(
        result.coordinates,
        result.attention,
        Path(args.root) / "outputs" / "figures" / figure_name,
        title="Stage 6A real-smallset attention weights" if args.real_features else "Stage 6A attention weights: pipeline example",
    )
    if args.real_features:
        log_path = Path(args.root) / "outputs" / "logs" / "stage6a_real_smallset_training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"{datetime.now().isoformat(timespec='seconds')} real-smallset MIL training completed\n"
            f"backend={result.backend}\nprediction_rows={len(result.predictions)}\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "passed", "backend": result.backend, "prediction_rows": len(result.predictions)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
