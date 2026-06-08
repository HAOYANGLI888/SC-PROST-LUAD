"""Generate the Stage 8 final claim-evidence audit."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting.stage8 import append_stage8_audit, write_final_evidence_audit  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    result = write_final_evidence_audit(ROOT)
    append_stage8_audit(
        ROOT,
        "\n## Stage 8 Final Evidence Audit\n\n"
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
        f"- Report: `{result['report']}`.\n"
        f"- Forbidden overclaim phrase hits in manuscript drafts: `{result['forbidden_phrase_hits']}`.\n"
        "- Stop boundary: no submission-ready final manuscript was generated.\n",
    )
    print(f"Stage 8 evidence audit generated: {result['report']}")
    return 0 if int(result["forbidden_phrase_hits"]) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

