"""Extract patch embeddings from Stage 6A patch coordinates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.patch_feature_extraction import extract_features_from_patch_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Stage 6A patch embeddings with skip-existing behavior.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--batch-size", type=int, default=32, help="Reserved for torchvision feature backends.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--backend", default="resnet50", choices=["resnet50", "handcrafted"])
    parser.add_argument("--encoder", choices=["resnet50"], help="Alias for the strict pretrained real-SVS encoder.")
    parser.add_argument("--allow-handcrafted-fallback", action="store_true", help="Allow a labeled limited fallback if torchvision ResNet50 is unavailable.")
    parser.add_argument("--small-test", action="store_true")
    parser.add_argument("--smallset", action="store_true")
    parser.add_argument("--real-svs", action="store_true", help="Require strict pretrained ResNet50 output for real SVS.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.small_test or args.smallset):
        raise SystemExit("Choose --small-test or --smallset. Full-cohort extraction is intentionally disabled in Stage 6A.")
    root = Path(args.root).resolve()
    if args.real_svs and args.small_test:
        raise SystemExit("--real-svs cannot be combined with --small-test.")
    if args.real_svs and args.allow_handcrafted_fallback:
        raise SystemExit("--real-svs does not allow the handcrafted fallback.")
    metadata = root / "data" / "metadata"
    if args.small_test:
        patch_summary = metadata / "stage6a_small_test" / "wsi_patch_summary.csv"
        patch_root = root / "data" / "processed" / "stage6a_small_test" / "wsi_patches"
        feature_root = root / "data" / "processed" / "stage6a_small_test" / "wsi_features"
        summary = metadata / "stage6a_small_test" / "wsi_feature_summary.csv"
    else:
        patch_summary = metadata / "stage6a_wsi_patch_summary.csv"
        patch_root = root / "data" / "processed" / "wsi_patches"
        feature_root = root / "data" / "processed" / "wsi_features"
        summary = metadata / "stage6a_wsi_feature_summary.csv"
    try:
        frame = extract_features_from_patch_summary(
            patch_summary,
            patch_root,
            feature_root,
            backend="handcrafted" if args.small_test else args.encoder or args.backend,
            device=args.device,
            batch_size=args.batch_size,
            allow_handcrafted_fallback=args.allow_handcrafted_fallback,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Stage 6A feature extraction failed: {exc}") from exc
    summary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary, index=False)
    if args.real_svs:
        tables = root / "outputs" / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        frame.to_csv(tables / "stage6a_real_feature_extraction_summary.csv", index=False)
    if not args.small_test:
        frame.to_csv(metadata / "stage6a_wsi_feature_summary.csv", index=False)
    print(json.dumps({"status": "passed", "slides_with_features": int(frame["feature_status"].isin(["extracted", "skipped_existing"]).sum()), "summary": str(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
