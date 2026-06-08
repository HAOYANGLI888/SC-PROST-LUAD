"""Download TCGA-LUAD GDC STAR-counts files with resume checks."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.gdc_rnaseq import download_main


if __name__ == "__main__":
    raise SystemExit(download_main())
