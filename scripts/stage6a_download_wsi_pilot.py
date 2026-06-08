"""Download only the selected Stage 6A WSI pilot cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.wsi_pilot_selection import download_wsi_pilot, refresh_wsi_pilot_download_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume-download the selected 100-slide WSI pilot only.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--status-only", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    frame = refresh_wsi_pilot_download_status(args.root) if args.status_only else download_wsi_pilot(
        args.root, workers=args.workers, retries=args.retries, timeout=args.timeout, min_free_gb=args.min_free_gb
    )
    complete = int(frame["download_status"].isin(["complete_existing", "copied_from_smallset", "downloaded_complete"]).sum())
    result = {
        "status": "passed" if complete == len(frame) else "partial",
        "selected_slides": len(frame),
        "complete_slides": complete,
        "partial_slides": int(frame["download_status"].eq("partial_resume_available").sum()),
        "failed_slides": int(frame["download_status"].str.startswith("failed", na=False).sum()),
        "expected_gb": float(frame["expected_size"].sum() / 1e9),
        "local_gb": float(frame["local_size"].sum() / 1e9),
    }
    print(json.dumps(result, indent=2))
    return 0 if args.status_only or result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

