"""Build a TCGA-LUAD TPM matrix from downloaded GDC STAR-counts files."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.gdc_rnaseq import matrix_main


if __name__ == "__main__":
    raise SystemExit(matrix_main())
