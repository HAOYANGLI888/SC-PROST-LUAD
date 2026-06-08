"""Score fixed Stage 4/5B mechanism programs in Stage 4B scRNA data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_raw_import import scrna_paths  # noqa: E402
from features.scrna_signature_scores import score_programs  # noqa: E402
from visualization.scrna_plots import (  # noqa: E402
    plot_key_gene_violins,
    plot_marker_dotplot,
    plot_program_heatmap,
)


def _artifacts(small_test: bool) -> dict[str, Path]:
    suffix = "stage4b_small_test" if small_test else ""
    processed = scrna_paths(ROOT, small_test=small_test).processed_dir
    table_dir = ROOT / "outputs" / "tables" / suffix
    figure_dir = ROOT / "outputs" / "figures" / suffix
    return {
        "input": processed / "scrna_luad_annotated.h5ad",
        "scored": processed / "scrna_luad_scored.h5ad",
        "per_cell": table_dir / "stage4b_scrna_per_cell_program_scores.csv",
        "celltype": table_dir / "stage4b_scrna_celltype_program_scores.csv",
        "gene": table_dir / "stage4b_scrna_gene_celltype_expression.csv",
        "missing": table_dir / "stage4b_scrna_signature_missingness.csv",
        "dotplot": figure_dir / "stage4b_scrna_program_score_dotplot.png",
        "heatmap": figure_dir / "stage4b_scrna_program_score_heatmap.png",
        "key_dotplot": figure_dir / "stage4b_scrna_key_gene_dotplot.png",
        "key_violin": figure_dir / "stage4b_scrna_key_gene_violin.png",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--small-test", action="store_true", help="Use isolated toy input.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    artifacts = _artifacts(args.small_test)
    if not artifacts["input"].exists():
        raise SystemExit(f"Annotated scRNA h5ad missing: {artifacts['input']}")
    per_cell, celltype, gene, missing, data = score_programs(
        artifacts["input"], artifacts["scored"]
    )
    artifacts["per_cell"].parent.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(artifacts["per_cell"], index=False)
    celltype.to_csv(artifacts["celltype"], index=False)
    gene.to_csv(artifacts["gene"], index=False)
    missing.to_csv(artifacts["missing"], index=False)
    plot_program_heatmap(celltype, artifacts["heatmap"])
    plot_program_heatmap(celltype, artifacts["dotplot"])
    plot_marker_dotplot(gene, artifacts["key_dotplot"], genes=["LDHA", "MKI67", "CDK1"])
    plot_key_gene_violins(per_cell, data, artifacts["key_violin"])
    print(f"Stage 4B program scoring complete: {artifacts['celltype']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
