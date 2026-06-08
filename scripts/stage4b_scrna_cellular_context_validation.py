"""Validate the cellular context of fixed Stage 4/5B programs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.scrna_cellular_context import (  # noqa: E402
    build_cellular_context_summary,
    build_mechanism_support_matrix,
)


def _paths(small_test: bool) -> dict[str, Path]:
    suffix = "stage4b_small_test" if small_test else ""
    table_dir = ROOT / "outputs" / "tables" / suffix
    report_dir = ROOT / "outputs" / "reports" / suffix
    return {
        "celltype": table_dir / "stage4b_scrna_celltype_program_scores.csv",
        "gene": table_dir / "stage4b_scrna_gene_celltype_expression.csv",
        "summary": table_dir / "stage4b_scrna_cellular_context_summary.csv",
        "support": table_dir / "stage4b_scrna_mechanism_support_matrix.csv",
        "report": report_dir / "stage4b_scrna_cellular_context_validation_report.md",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--small-test", action="store_true", help="Use isolated toy outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = _paths(args.small_test)
    if not paths["celltype"].exists() or not paths["gene"].exists():
        raise SystemExit("Stage 4B program-score tables are missing.")
    celltype = pd.read_csv(paths["celltype"])
    gene = pd.read_csv(paths["gene"])
    support = build_mechanism_support_matrix(celltype, gene)
    summary = build_cellular_context_summary(celltype, gene, support)
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(paths["summary"], index=False)
    support.to_csv(paths["support"], index=False)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(
        f"""# Stage 4B Raw scRNA Cellular-Context Validation

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Mode: `{"toy_small_test" if args.small_test else "formal"}`
- Mechanisms evaluated: `{len(support)}`
- Context-supported mechanisms: `{int(support.support_status.str.contains("supported").sum())}`

## Mechanism Results

{support.to_markdown(index=False)}

## Key-Gene And Interpretation Questions

{summary.to_markdown(index=False)}

## Scientific Boundary

This analysis uses no survival outcome and does not retrain or reselect the Stage 2D model, Stage 4 signatures, or Stage 5B candidates. It evaluates cellular localization and expression context only. It is not a single-cell survival validation and does not establish causality.
""",
        encoding="utf-8",
    )
    print(f"Stage 4B cellular-context validation complete: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
