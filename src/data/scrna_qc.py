"""Quality control and preprocessing for raw scRNA-seq data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.scrna_raw_import import sanitize_anndata_strings


@dataclass(frozen=True)
class QCThresholds:
    min_genes: int = 100
    min_counts: int = 200
    max_genes: int = 10_000
    max_mito_fraction: float = 30.0
    min_cells_per_gene: int = 3
    target_sum: float = 10_000.0
    n_hvg: int = 2_000
    n_pcs: int = 30
    n_neighbors: int = 15


def preprocess_scrna(
    input_h5ad: str | Path,
    output_h5ad: str | Path,
    *,
    thresholds: QCThresholds,
    downsample_cells: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, object]:
    """Run conservative QC, log normalization, PCA, neighbors, and UMAP."""

    import scanpy as sc

    data = sc.read_h5ad(input_h5ad)
    cells_before = int(data.n_obs)
    genes_before = int(data.n_vars)
    data.var_names = data.var_names.astype(str)
    data.var["mt"] = data.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        data, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    keep_cells = (
        (data.obs["n_genes_by_counts"] >= thresholds.min_genes)
        & (data.obs["total_counts"] >= thresholds.min_counts)
        & (data.obs["n_genes_by_counts"] <= thresholds.max_genes)
        & (data.obs["pct_counts_mt"] <= thresholds.max_mito_fraction)
    )
    data = data[keep_cells].copy()
    sc.pp.filter_genes(data, min_cells=thresholds.min_cells_per_gene)
    if downsample_cells and data.n_obs > downsample_cells:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(data.n_obs, downsample_cells, replace=False))
        data = data[selected].copy()
        data.uns["downsampled_for_analysis"] = True
        data.uns["downsample_target"] = int(downsample_cells)
    else:
        data.uns["downsampled_for_analysis"] = False

    data.layers["counts"] = data.X.copy()
    sc.pp.normalize_total(data, target_sum=thresholds.target_sum)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(
        data, n_top_genes=min(thresholds.n_hvg, data.n_vars), flavor="seurat"
    )
    n_pcs = min(thresholds.n_pcs, max(2, int(data.var["highly_variable"].sum()) - 1))
    sc.tl.pca(data, n_comps=n_pcs, use_highly_variable=True, svd_solver="arpack")
    sc.pp.neighbors(
        data,
        n_neighbors=min(thresholds.n_neighbors, max(2, data.n_obs - 1)),
        n_pcs=n_pcs,
        random_state=seed,
    )
    sc.tl.umap(data, random_state=seed)
    data.uns["stage4b_qc_thresholds"] = {
        key: value for key, value in thresholds.__dict__.items()
    }
    output = Path(output_h5ad)
    output.parent.mkdir(parents=True, exist_ok=True)
    sanitize_anndata_strings(data).write_h5ad(output, compression="gzip")
    summary = pd.DataFrame(
        [
            {
                "cells_before_qc": cells_before,
                "cells_after_qc": int(data.n_obs),
                "cells_removed": cells_before - int(data.n_obs),
                "genes_before_qc": genes_before,
                "genes_after_qc": int(data.n_vars),
                "min_genes": thresholds.min_genes,
                "min_counts": thresholds.min_counts,
                "max_genes": thresholds.max_genes,
                "max_mito_fraction": thresholds.max_mito_fraction,
                "min_cells_per_gene": thresholds.min_cells_per_gene,
                "target_sum": thresholds.target_sum,
                "n_hvg": int(data.var["highly_variable"].sum()),
                "downsampled": bool(data.uns["downsampled_for_analysis"]),
            }
        ]
    )
    return summary, data
