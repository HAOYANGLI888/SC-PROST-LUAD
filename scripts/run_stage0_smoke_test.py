"""Command-line wrapper for the Stage 0 smoke test."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scprost_luad.stages.stage0 import main


if __name__ == "__main__":
    raise SystemExit(main())
