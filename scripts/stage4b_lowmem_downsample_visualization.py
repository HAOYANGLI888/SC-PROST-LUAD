"""Create optional stratified downsample UMAPs without full-cohort embedding."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_raw_import import scrna_paths  # noqa: E402
from validation.scrna_chunked_scoring import scores_for_matrix  # noqa: E402
from validation.scrna_lowmem_reader import (  # noqa: E402
    load_obs_with_analysis_labels,
    stratified_downsample_indices,
)
from visualization.scrna_lowmem_plots import (  # noqa: E402
    plot_umap_categories,
    plot_umap_key_genes,
    plot_umap_programs,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/base.yaml")
    result.add_argument("--n-cells", type=int, default=50_000)
    result.add_argument("--seed", type=int, default=20260529)
    return result


def _attempt(source: Path, obs: pd.DataFrame, n_cells: int, seed: int):
    import anndata as ad
    import scanpy as sc

    indices = stratified_downsample_indices(obs, n_cells, seed=seed)
    backed = ad.read_h5ad(source, backed="r")
    try:
        data = backed[indices, :].to_memory()
    finally:
        backed.file.close()
    data.obs["analysis_cell_type"] = (
        obs.iloc[indices]["analysis_cell_type"].astype(str).to_numpy()
    )
    if not sparse.issparse(data.X):
        raise MemoryError("Downsample unexpectedly materialized as a dense matrix.")
    key_values, program_values = scores_for_matrix(data.X.tocsr(), data.var_names)
    sc.pp.normalize_total(data, target_sum=10_000)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes=2000, flavor="seurat")
    if int(data.var["highly_variable"].sum()) < 100:
        raise RuntimeError("Too few highly variable genes for downsample UMAP.")
    reduced = data[:, data.var["highly_variable"]].copy()
    del data
    gc.collect()
    sc.tl.pca(
        reduced,
        n_comps=30,
        zero_center=False,
        random_state=seed,
    )
    sc.pp.neighbors(reduced, n_neighbors=15, n_pcs=30, random_state=seed)
    sc.tl.umap(reduced, random_state=seed)
    return indices, reduced, key_values, program_values


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source = scrna_paths(ROOT).raw_or_converted_h5ad
    obs = load_obs_with_analysis_labels(source)
    requested = min(args.n_cells, len(obs))
    attempts = []
    for number in dict.fromkeys([requested, 30_000, 20_000]):
        if number > len(obs):
            continue
        try:
            indices, data, key_values, program_values = _attempt(
                source, obs, number, args.seed
            )
            attempts.append({"n_cells": number, "status": "success", "error": ""})
            break
        except Exception as exc:
            attempts.append(
                {
                    "n_cells": number,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            gc.collect()
    else:
        data = None

    table_dir = ROOT / "outputs" / "tables"
    figure_dir = ROOT / "outputs" / "figures"
    report_dir = ROOT / "outputs" / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "stage4b_lowmem_visualization_report.md"
    if data is None:
        report_path.write_text(
            f"""# Stage 4B-LowMem Visualization Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Status: `UMAP_failed_fallback_to_dotplot_and_heatmap`
- Attempts:

{pd.DataFrame(attempts).to_markdown(index=False)}

The full-cohort dotplot and program heatmap remain the low-memory visualization
fallback. No full 208,506-cell PCA, neighbors, or UMAP was attempted.
""",
            encoding="utf-8",
        )
        print("Downsample UMAP failed; full-cohort dotplot/heatmap retained.")
        return 0

    index_table = obs.iloc[indices].reset_index()
    index_table.insert(0, "matrix_row_index", indices)
    index_path = table_dir / "stage4b_lowmem_downsample_index.csv"
    index_table.to_csv(index_path, index=False)
    coordinates = np.asarray(data.obsm["X_umap"])
    plot_umap_categories(
        coordinates,
        data.obs["analysis_cell_type"],
        figure_dir / "stage4b_lowmem_umap_celltypes.png",
    )
    plot_umap_key_genes(
        coordinates,
        key_values,
        figure_dir / "stage4b_lowmem_umap_key_genes.png",
    )
    plot_umap_programs(
        coordinates,
        program_values,
        figure_dir / "stage4b_lowmem_umap_program_scores.png",
    )
    report_path.write_text(
        f"""# Stage 4B-LowMem Visualization Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Status: `completed`
- Requested cells: `{requested}`
- Used cells: `{data.n_obs}`
- Stratification: official harmonized cell type plus sample.
- Embedding genes: `2000` highly variable genes selected within the downsample.
- PCA: sparse-safe, `zero_center=False`, 30 components.
- Full-cohort PCA/neighbors/UMAP: `not performed`.

## Attempts

{pd.DataFrame(attempts).to_markdown(index=False)}

These UMAPs are visualization aids only. Full-cohort cellular-context conclusions
are based on chunked statistics across all 208,506 cells.
""",
        encoding="utf-8",
    )
    print(f"Stage 4B-LowMem downsample UMAP complete: n={data.n_obs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
