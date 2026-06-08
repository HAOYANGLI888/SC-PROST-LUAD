"""Extract WSI tissue-rich patch coordinates and QC images."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.patch_extraction import extract_patch_indexes_from_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract tissue-rich WSI patch coordinates without saving every patch PNG.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--tissue-threshold", type=float, default=0.35)
    parser.add_argument("--max-patches", type=int)
    parser.add_argument("--small-test", action="store_true")
    parser.add_argument("--smallset", action="store_true")
    parser.add_argument("--real-svs", action="store_true", help="Label and export strict real-SVS smallset outputs.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.small_test or args.smallset):
        raise SystemExit("Choose --small-test or --smallset. Full-cohort extraction is intentionally disabled in Stage 6A.")
    root = Path(args.root).resolve()
    metadata = root / "data" / "metadata"
    if args.small_test:
        status = metadata / "stage6a_small_test" / "wsi_smallset_download_status.csv"
        output_root = root / "data" / "processed" / "stage6a_small_test" / "wsi_patches"
        summary = metadata / "stage6a_small_test" / "wsi_patch_summary.csv"
        max_patches = args.max_patches or 32
    else:
        status = metadata / "stage6a_wsi_smallset_download_status.csv"
        output_root = root / "data" / "processed" / "wsi_patches"
        summary = metadata / "stage6a_wsi_patch_summary.csv"
        max_patches = args.max_patches or 1000
    try:
        frame = extract_patch_indexes_from_status(
            status,
            output_root,
            patch_size=args.patch_size,
            tissue_threshold=args.tissue_threshold,
            max_patches=max_patches,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Stage 6A patch extraction failed: {exc}") from exc
    summary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary, index=False)
    if args.real_svs:
        if args.small_test:
            raise SystemExit("--real-svs cannot be combined with --small-test.")
        tables = root / "outputs" / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        frame.to_csv(tables / "stage6a_real_patch_extraction_summary.csv", index=False)
    extracted = frame.loc[frame["patch_status"] == "extracted"]
    if not extracted.empty:
        slide_id = str(extracted.iloc[0]["file_id"])
        figures = root / "outputs" / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_root / slide_id / "qc_tissue_mask.png", figures / "stage6a_wsi_tissue_mask_example.png")
        shutil.copyfile(output_root / slide_id / "qc_patch_grid.png", figures / "stage6a_wsi_patch_grid_example.png")
        if args.real_svs:
            shutil.copyfile(output_root / slide_id / "qc_tissue_mask.png", figures / "stage6a_real_tissue_mask_example.png")
            shutil.copyfile(output_root / slide_id / "qc_patch_grid.png", figures / "stage6a_real_patch_grid_example.png")
    print(json.dumps({"status": "passed", "slides_extracted": len(extracted), "summary": str(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
