"""Generate the Stage 4B raw scRNA cellular-context figure set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_raw_import import scrna_paths  # noqa: E402
from visualization.scrna_plots import (  # noqa: E402
    plot_context_summary,
    plot_key_gene_violins,
    plot_marker_dotplot,
    plot_program_heatmap,
    plot_program_umaps,
    plot_umap,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--small-test", action="store_true", help="Use isolated toy outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    suffix = "stage4b_small_test" if args.small_test else ""
    processed = scrna_paths(ROOT, small_test=args.small_test).processed_dir
    scored = processed / "scrna_luad_scored.h5ad"
    if not scored.exists():
        raise SystemExit(f"Scored scRNA h5ad missing: {scored}")
    import scanpy as sc

    data = sc.read_h5ad(scored)
    table_dir = ROOT / "outputs" / "tables" / suffix
    figure_dir = ROOT / "outputs" / "figures" / suffix
    per_cell = pd.read_csv(table_dir / "stage4b_scrna_per_cell_program_scores.csv")
    celltype = pd.read_csv(table_dir / "stage4b_scrna_celltype_program_scores.csv")
    gene = pd.read_csv(table_dir / "stage4b_scrna_gene_celltype_expression.csv")
    support = pd.read_csv(table_dir / "stage4b_scrna_mechanism_support_matrix.csv")
    plot_umap(data, "analysis_cell_type", figure_dir / "stage4b_scrna_umap_celltypes.png", title="Major cell types")
    plot_marker_dotplot(gene, figure_dir / "stage4b_scrna_key_gene_dotplot.png")
    plot_program_heatmap(celltype, figure_dir / "stage4b_scrna_module_score_heatmap.png")
    plot_key_gene_violins(per_cell, data, figure_dir / "stage4b_scrna_ldha_mki67_cdk1_violin.png")
    plot_program_umaps(data, figure_dir / "stage4b_scrna_program_umaps.png")
    plot_context_summary(support, figure_dir / "stage4b_scrna_cellular_context_summary.png")
    print(f"Stage 4B figures generated: {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
