"""Run the Journal of Translational Medicine format and claim-evidence audit."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting.stage9_jtm import append_stage9_audit, write_compliance_outputs  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    outputs = write_compliance_outputs(ROOT)
    append_stage9_audit(
        ROOT,
        "\n## Stage 9 JTM Compliance Audit\n\n"
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
        f"- Compliance report: `{outputs['compliance']}`.\n"
        f"- Evidence-claim audit: `{outputs['claims']}`.\n"
        f"- Submission checklist: `{outputs['checklist']}`.\n"
        "- Status: author metadata and final submission assets remain pending.\n",
    )
    print(f"Generated JTM compliance report: {outputs['compliance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
