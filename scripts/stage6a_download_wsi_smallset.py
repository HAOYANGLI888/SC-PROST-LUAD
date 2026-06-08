"""Select and optionally download a balanced TCGA-LUAD diagnostic WSI smallset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.wsi_patient_map import download_smallset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a resumable balanced diagnostic WSI smallset. Never downloads the full cohort.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--n-slides", type=int, default=20)
    parser.add_argument("--gdc-client", help="Optional path to gdc-client.exe.")
    parser.add_argument("--select-only", action="store_true", help="Write the selected real manifest and storage estimate without downloading SVS files.")
    parser.add_argument("--small-test", action="store_true", help="Generate synthetic TIFF slides.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not (Path(args.root) / args.config).exists():
        raise SystemExit(f"Config file not found: {Path(args.root) / args.config}")
    try:
        result = download_smallset(
            args.root,
            n_slides=args.n_slides,
            gdc_client=args.gdc_client,
            select_only=args.select_only,
            small_test=args.small_test,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Stage 6A WSI smallset download failed: {exc}") from exc
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

