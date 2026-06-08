"""Summarize fixed Stage 4/5B genes across official cell-type contexts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_raw_import import scrna_paths  # noqa: E402
from validation.scrna_chunked_scoring import (  # noqa: E402
    summarize_selected_gene_expression,
)
from visualization.scrna_lowmem_plots import plot_gene_dotplot  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/base.yaml")
    result.add_argument("--block-size", type=int, default=2048)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source = scrna_paths(ROOT).raw_or_converted_h5ad
    expression, missingness = summarize_selected_gene_expression(
        source, block_size=args.block_size
    )
    table_dir = ROOT / "outputs" / "tables"
    figure_dir = ROOT / "outputs" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    expression_path = table_dir / "stage4b_lowmem_gene_celltype_expression.csv"
    missingness_path = table_dir / "stage4b_lowmem_gene_missingness.csv"
    figure_path = figure_dir / "stage4b_lowmem_key_gene_dotplot.png"
    expression.to_csv(expression_path, index=False)
    missingness.to_csv(missingness_path, index=False)
    plot_gene_dotplot(expression, figure_path)
    print(
        "Stage 4B-LowMem gene expression complete: "
        f"{expression_path}; missing={int(missingness.status.eq('missing').sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
