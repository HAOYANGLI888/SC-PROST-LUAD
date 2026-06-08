"""Preserve or conservatively infer Stage 4B scRNA cell annotations."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_annotation import annotate_cells  # noqa: E402
from data.scrna_raw_import import scrna_paths  # noqa: E402
from visualization.scrna_plots import plot_marker_dotplot, plot_umap  # noqa: E402


def _artifacts(small_test: bool) -> dict[str, Path]:
    suffix = "stage4b_small_test" if small_test else ""
    processed = scrna_paths(ROOT, small_test=small_test).processed_dir
    return {
        "input": processed / "scrna_luad_processed.h5ad",
        "output": processed / "scrna_luad_annotated.h5ad",
        "counts": ROOT / "outputs" / "tables" / suffix / "stage4b_scrna_cell_type_counts.csv",
        "markers": ROOT / "outputs" / "tables" / suffix / "stage4b_scrna_marker_support.csv",
        "umap": ROOT / "outputs" / "figures" / suffix / "stage4b_scrna_umap_celltypes.png",
        "dotplot": ROOT / "outputs" / "figures" / suffix / "stage4b_scrna_marker_dotplot.png",
        "report": ROOT / "outputs" / "reports" / suffix / "stage4b_scrna_cell_annotation_report.md",
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
        raise SystemExit(f"Processed scRNA h5ad missing: {artifacts['input']}")
    counts, markers, data = annotate_cells(artifacts["input"], artifacts["output"])
    artifacts["counts"].parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(artifacts["counts"], index=False)
    markers.to_csv(artifacts["markers"], index=False)
    plot_umap(data, "analysis_cell_type", artifacts["umap"], title="Major cell types")
    gene_rows = []
    for row in markers.itertuples(index=False):
        for gene in str(row.available_markers).split(";"):
            gene_rows.append(
                {
                    "analysis_cell_type": row.analysis_cell_type,
                    "gene_symbol": gene,
                    "average_log_expression": row.mean_log_expression,
                    "detection_fraction": row.detection_fraction,
                }
            )
    plot_marker_dotplot(
        __import__("pandas").DataFrame(gene_rows), artifacts["dotplot"]
    )
    artifacts["report"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["report"].write_text(
        f"""# Stage 4B Cell Annotation Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Mode: `{"toy_small_test" if args.small_test else "formal"}`
- Annotation method: `{data.uns.get("stage4b_annotation_method", "unknown")}`
- Cells: `{data.n_obs}`
- Harmonized major cell types: `{counts.shape[0]}`
- Source annotation retained: `{"yes" if "source_annotation" in data.uns.get("stage4b_annotation_method", "") else "no"}`

Malignant epithelial labels are used only where the public source explicitly annotated malignant cells. Otherwise cells remain epithelial or conservatively marker-labelled.
""",
        encoding="utf-8",
    )
    print(f"Stage 4B annotation complete: {artifacts['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
