"""Resume only the persisted Stage 6A real WSI smallset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.wsi_patient_map import refresh_smallset_download_status, resume_smallset_download


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume the selected 20-slide real GDC WSI smallset with persistent status checks.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--status-only", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    frame = (
        refresh_smallset_download_status(args.root)
        if args.status_only
        else resume_smallset_download(args.root, workers=args.workers, retries=args.retries, timeout=args.timeout)
    )
    complete = int(frame["download_status"].isin(["complete_existing", "downloaded_complete"]).sum())
    result = {
        "status": "passed" if complete == len(frame) else "partial",
        "selected_slides": len(frame),
        "complete_slides": complete,
        "partial_slides": int(frame["download_status"].eq("partial_resume_available").sum()),
        "expected_gb": float(frame["expected_size"].sum() / 1e9),
        "present_gb": float(frame["local_size"].sum() / 1e9),
        "status_path": str(Path(args.root).resolve() / "data" / "metadata" / "stage6a_wsi_smallset_download_status.csv"),
    }
    print(json.dumps(result, indent=2))
    return 0 if args.status_only or complete == len(frame) else 1


if __name__ == "__main__":
    raise SystemExit(main())
