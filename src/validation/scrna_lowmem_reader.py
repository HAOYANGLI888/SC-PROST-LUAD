"""Sparse-safe readers and summaries for the Stage 4B low-memory workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy import sparse

from data.scrna_annotation import _source_harmonized_labels


@dataclass(frozen=True)
class LowMemSCRNAInfo:
    path: Path
    n_cells: int
    n_genes: int
    obs_columns: tuple[str, ...]
    has_umap: bool


def load_obs_with_analysis_labels(h5ad_path: str | Path) -> pd.DataFrame:
    """Load only observation metadata and add conservative source-derived labels."""

    import anndata as ad

    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = data.obs.copy()
        obs.index = pd.Index(data.obs_names.astype(str), name="cell_id")
    finally:
        data.file.close()
    labels = _source_harmonized_labels(obs)
    if labels is None:
        raise ValueError("Official GSE131907 cell-type annotation is unavailable.")
    obs["analysis_cell_type"] = labels.astype(str).to_numpy()
    if "sample_id" not in obs and "Sample" in obs:
        obs["sample_id"] = obs["Sample"].astype(str)
    if "patient_id" not in obs and "Sample" in obs:
        patient_number = obs["Sample"].astype(str).str.extract(
            r"(\d+)$", expand=False
        )
        obs["patient_id"] = "P" + patient_number.fillna("unknown").str.zfill(2)
    return obs


def inspect_h5ad(h5ad_path: str | Path) -> LowMemSCRNAInfo:
    import anndata as ad

    path = Path(h5ad_path).resolve()
    data = ad.read_h5ad(path, backed="r")
    try:
        return LowMemSCRNAInfo(
            path=path,
            n_cells=int(data.n_obs),
            n_genes=int(data.n_vars),
            obs_columns=tuple(str(column) for column in data.obs.columns),
            has_umap="X_umap" in data.obsm,
        )
    finally:
        data.file.close()


def iter_sparse_row_blocks(
    h5ad_path: str | Path,
    *,
    block_size: int = 2048,
) -> Iterator[tuple[int, int, sparse.csr_matrix]]:
    """Yield sequential CSR blocks without materializing the complete matrix."""

    import anndata as ad

    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        for start in range(0, data.n_obs, block_size):
            end = min(start + block_size, data.n_obs)
            block = data.X[start:end]
            if sparse.issparse(block):
                matrix = block.tocsr()
            else:
                matrix = sparse.csr_matrix(np.asarray(block))
            yield start, end, matrix
    finally:
        data.file.close()


def compute_lowmem_qc(
    h5ad_path: str | Path,
    *,
    block_size: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute raw-count QC metrics and grouped counts with bounded memory."""

    import h5py

    info = inspect_h5ad(h5ad_path)
    obs = load_obs_with_analysis_labels(h5ad_path)
    total_counts = np.empty(info.n_cells, dtype=np.int64)
    detected_genes = np.empty(info.n_cells, dtype=np.int32)
    for start, end, block in iter_sparse_row_blocks(
        h5ad_path, block_size=block_size
    ):
        total_counts[start:end] = np.asarray(block.sum(axis=1)).ravel().astype(
            np.int64
        )
        detected_genes[start:end] = block.getnnz(axis=1).astype(np.int32)

    with h5py.File(h5ad_path, "r") as handle:
        nonzero_counts = int(handle["X/data"].shape[0])

    def quantile_rows(metric: str, values: np.ndarray) -> list[dict[str, object]]:
        rows = [
            {"metric": f"{metric}_mean", "value": float(np.mean(values))},
            {"metric": f"{metric}_median", "value": float(np.median(values))},
        ]
        for quantile in (0.01, 0.05, 0.25, 0.75, 0.95, 0.99):
            rows.append(
                {
                    "metric": f"{metric}_q{int(quantile * 100):02d}",
                    "value": float(np.quantile(values, quantile)),
                }
            )
        return rows

    summary_rows: list[dict[str, object]] = [
        {"metric": "n_cells", "value": info.n_cells},
        {"metric": "n_genes", "value": info.n_genes},
        {"metric": "nonzero_counts", "value": nonzero_counts},
        {
            "metric": "matrix_density",
            "value": nonzero_counts / (info.n_cells * info.n_genes),
        },
        {"metric": "n_samples", "value": int(obs["sample_id"].nunique())},
        {"metric": "n_patients", "value": int(obs["patient_id"].nunique())},
        {
            "metric": "n_analysis_cell_types",
            "value": int(obs["analysis_cell_type"].nunique()),
        },
    ]
    for field in (
        "Sample",
        "Sample_Origin",
        "Cell_type",
        "Cell_type.refined",
        "Cell_subtype",
        "sample_id",
        "patient_id",
    ):
        summary_rows.append(
            {
                "metric": f"metadata_{field}_present",
                "value": field in obs.columns,
            }
        )
    summary_rows.extend(quantile_rows("total_counts", total_counts))
    summary_rows.extend(quantile_rows("detected_genes", detected_genes))

    metrics = pd.DataFrame(
        {
            "cell_id": obs.index,
            "total_counts": total_counts,
            "detected_genes": detected_genes,
        }
    )
    count_rows: list[pd.DataFrame] = []
    for level, column in (
        ("sample", "sample_id"),
        ("patient", "patient_id"),
        ("cell_type", "analysis_cell_type"),
    ):
        grouped = (
            obs.groupby(column, observed=True)
            .size()
            .rename("cell_count")
            .reset_index()
            .rename(columns={column: "group_id"})
        )
        grouped.insert(0, "grouping_level", level)
        grouped["fraction"] = grouped["cell_count"] / info.n_cells
        count_rows.append(grouped)
    return pd.DataFrame(summary_rows), pd.concat(count_rows), metrics


def stratified_downsample_indices(
    obs: pd.DataFrame,
    n_cells: int,
    *,
    seed: int = 20260529,
) -> np.ndarray:
    """Proportionally sample within official cell-type and sample strata."""

    if n_cells >= len(obs):
        return np.arange(len(obs), dtype=np.int64)
    keys = (
        obs["analysis_cell_type"].astype(str)
        + "||"
        + obs["sample_id"].astype(str)
    )
    counts = keys.value_counts(sort=False)
    raw_quota = counts * (n_cells / len(obs))
    quota = np.floor(raw_quota).astype(int).clip(lower=1)
    quota = np.minimum(quota, counts)
    while int(quota.sum()) > n_cells:
        candidates = quota[quota > 1]
        if candidates.empty:
            break
        index = (quota[candidates.index] - raw_quota[candidates.index]).idxmax()
        quota.loc[index] -= 1
    while int(quota.sum()) < n_cells:
        capacity = counts - quota
        candidates = capacity[capacity > 0]
        if candidates.empty:
            break
        index = (raw_quota[candidates.index] - quota[candidates.index]).idxmax()
        quota.loc[index] += 1
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for key, number in quota.items():
        positions = np.flatnonzero(keys.to_numpy() == key)
        selected.append(rng.choice(positions, size=int(number), replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))
