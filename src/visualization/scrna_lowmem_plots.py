"""Python-only publication-oriented plots for Stage 4B-LowMem."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features.scrna_signature_scores import KEY_GENES


PALETTE = {
    "signal": "#2A6F97",
    "accent": "#C65D3B",
    "neutral": "#8A9299",
    "protective": "#3A8D71",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save(fig, path: str | Path, *, dpi: int = 300) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gene_dotplot(frame: pd.DataFrame, path: str | Path) -> None:
    """Show localization using color for expression and area for detection."""

    _style()
    genes = frame["gene_symbol"].drop_duplicates().tolist()
    cell_types = (
        frame.groupby("analysis_cell_type")["total_cell_count"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    value_max = max(float(frame["average_log1p_cpm"].quantile(0.98)), 0.5)
    fig, axis = plt.subplots(
        figsize=(max(9.5, len(genes) * 0.28), max(4.3, len(cell_types) * 0.33))
    )
    scatter = None
    indexed = frame.set_index(["analysis_cell_type", "gene_symbol"])
    for y, cell_type in enumerate(cell_types):
        for x, gene in enumerate(genes):
            if (cell_type, gene) not in indexed.index:
                continue
            row = indexed.loc[(cell_type, gene)]
            scatter = axis.scatter(
                x,
                y,
                s=8 + 150 * float(row["detection_rate"]),
                c=float(row["average_log1p_cpm"]),
                cmap="viridis",
                vmin=0,
                vmax=value_max,
                linewidths=0,
            )
    axis.set_xticks(range(len(genes)), genes, rotation=55, ha="right")
    axis.set_yticks(range(len(cell_types)), cell_types)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title("Fixed mechanism genes across official cell-type contexts")
    axis.grid(color="#E5E7E9", linewidth=0.35, alpha=0.7)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axis, pad=0.01)
        colorbar.set_label("Mean log1p(CPM)")
    fig.tight_layout()
    _save(fig, path)


def plot_program_heatmap(frame: pd.DataFrame, path: str | Path) -> None:
    """Plot within-program standardized group means for readable localization."""

    _style()
    matrix = frame.pivot(
        index="analysis_cell_type",
        columns="program",
        values="mean_log1p_cpm_score",
    )
    matrix = matrix.loc[
        frame.groupby("analysis_cell_type")["cell_count"]
        .max()
        .sort_values(ascending=False)
        .index
    ]
    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0)
    standardized = standardized.fillna(0)
    limit = max(1.5, float(np.nanquantile(np.abs(standardized), 0.98)))
    fig, axis = plt.subplots(
        figsize=(6.8, max(4.2, len(standardized) * 0.34))
    )
    image = axis.imshow(
        standardized.to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(
        range(len(standardized.columns)),
        [value.replace("_", " ") for value in standardized.columns],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(standardized)), standardized.index)
    axis.set_title("Fixed program localization by official cell type")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Cell-type z score")
    fig.tight_layout()
    _save(fig, path)


def plot_umap_categories(
    coordinates: np.ndarray,
    labels: pd.Series,
    path: str | Path,
) -> None:
    _style()
    categories = sorted(labels.astype(str).unique())
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(categories), 2)))
    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    for category, color in zip(categories, colors):
        mask = labels.astype(str).eq(category).to_numpy()
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=1.2,
            alpha=0.55,
            color=color,
            label=category,
            rasterized=True,
        )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("Stratified downsample: official cell-type context")
    axis.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=6,
        markerscale=4,
    )
    fig.tight_layout()
    _save(fig, path)


def _continuous_panel(
    coordinates: np.ndarray,
    values: dict[str, np.ndarray],
    path: str | Path,
    *,
    title: str,
) -> None:
    _style()
    columns = len(values)
    fig, axes = plt.subplots(
        1, columns, figsize=(3.5 * columns, 3.4), squeeze=False
    )
    for axis, (name, vector) in zip(axes.ravel(), values.items()):
        lower, upper = np.nanquantile(vector, [0.02, 0.98])
        scatter = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=vector,
            s=1.0,
            cmap="magma",
            vmin=lower,
            vmax=max(upper, lower + 1e-6),
            rasterized=True,
            linewidths=0,
        )
        axis.set_title(name.replace("_", " "))
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(scatter, ax=axis, fraction=0.045, pad=0.02)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save(fig, path)


def plot_umap_key_genes(
    coordinates: np.ndarray,
    values: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    _continuous_panel(
        coordinates,
        {gene: values[gene] for gene in KEY_GENES},
        path,
        title="Key protein-supported genes in the stratified downsample",
    )


def plot_umap_programs(
    coordinates: np.ndarray,
    values: dict[str, np.ndarray],
    path: str | Path,
) -> None:
    selected = {
        name: values[name]
        for name in ("hypoxia", "proliferation", "emt_like", "caf_matrix")
    }
    _continuous_panel(
        coordinates,
        selected,
        path,
        title="Fixed mechanism programs in the stratified downsample",
    )
