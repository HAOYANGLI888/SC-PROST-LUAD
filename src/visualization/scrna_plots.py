"""Submission-oriented plots for Stage 4B raw scRNA cellular context."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features.scrna_signature_scores import KEY_GENES, RISK_PROGRAMS


def _save(fig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_qc_violin(data, path: str | Path) -> None:
    columns = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for axis, column in zip(axes, columns):
        values = pd.to_numeric(data.obs[column], errors="coerce").dropna()
        axis.violinplot(values, showmedians=True)
        axis.set_title(column.replace("_", " "))
        axis.set_xticks([])
    fig.tight_layout()
    _save(fig, path)


def plot_umap(data, color: str, path: str | Path, *, title: str | None = None) -> None:
    coordinates = np.asarray(data.obsm["X_umap"])
    labels = data.obs[color].astype(str)
    categories = sorted(labels.unique())
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(2, len(categories))))
    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    for category, category_color in zip(categories, colors):
        mask = labels.eq(category).to_numpy()
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=2,
            alpha=0.55,
            color=category_color,
            label=category,
            rasterized=True,
        )
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title(title or color.replace("_", " "))
    axis.legend(frameon=False, fontsize=7, markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, path)


def plot_marker_dotplot(
    gene_expression: pd.DataFrame,
    path: str | Path,
    *,
    genes: list[str] | None = None,
) -> None:
    genes = genes or [
        "EPCAM", "KRT8", "CD3D", "CD8A", "MS4A1", "MZB1", "LST1", "CD74",
        "COL1A1", "PECAM1", "TPSAB1", "NKG7",
    ]
    subset = gene_expression.loc[gene_expression["gene_symbol"].isin(genes)].copy()
    cell_types = sorted(subset["analysis_cell_type"].unique())
    fig, axis = plt.subplots(figsize=(11, max(4.5, len(cell_types) * 0.38)))
    for y, cell_type in enumerate(cell_types):
        rows = subset.loc[subset["analysis_cell_type"].eq(cell_type)].set_index("gene_symbol")
        for x, gene in enumerate(genes):
            if gene not in rows.index:
                continue
            row = rows.loc[gene]
            axis.scatter(
                x,
                y,
                s=15 + 230 * float(row["detection_fraction"]),
                c=float(row["average_log_expression"]),
                cmap="viridis",
                vmin=0,
                vmax=max(1.0, float(subset["average_log_expression"].quantile(0.95))),
            )
    axis.set_xticks(range(len(genes)), genes, rotation=45, ha="right")
    axis.set_yticks(range(len(cell_types)), cell_types)
    axis.set_title("Marker expression by cell type")
    fig.tight_layout()
    _save(fig, path)


def plot_program_heatmap(celltype_scores: pd.DataFrame, path: str | Path) -> None:
    columns = [f"{program}_score_mean" for program in RISK_PROGRAMS]
    frame = celltype_scores.set_index("analysis_cell_type")[columns]
    fig, axis = plt.subplots(figsize=(7.2, max(4.2, frame.shape[0] * 0.38)))
    image = axis.imshow(frame.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    axis.set_xticks(range(len(columns)), [column.replace("_score_mean", "") for column in columns], rotation=35, ha="right")
    axis.set_yticks(range(len(frame)), frame.index)
    fig.colorbar(image, ax=axis, label="Mean per-cell z score")
    axis.set_title("Mechanism program scores by cell type")
    fig.tight_layout()
    _save(fig, path)


def plot_key_gene_violins(per_cell: pd.DataFrame, data, path: str | Path) -> None:
    lookup = {str(gene).upper(): index for index, gene in enumerate(data.var_names)}
    cell_types = sorted(per_cell["analysis_cell_type"].unique())
    fig, axes = plt.subplots(len(KEY_GENES), 1, figsize=(11, 8.5), sharex=True)
    for axis, gene in zip(axes, KEY_GENES):
        if gene not in lookup:
            axis.set_title(f"{gene} unavailable")
            continue
        matrix = data.X[:, lookup[gene]]
        values = matrix.toarray().ravel() if hasattr(matrix, "toarray") else np.asarray(matrix).ravel()
        groups = [
            values[per_cell["analysis_cell_type"].eq(cell_type).to_numpy()]
            for cell_type in cell_types
        ]
        axis.violinplot(groups, showmedians=True, showextrema=False)
        axis.set_ylabel(gene)
    axes[-1].set_xticks(range(1, len(cell_types) + 1), cell_types, rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, path)


def plot_program_umaps(data, path: str | Path) -> None:
    coordinates = np.asarray(data.obsm["X_umap"])
    programs = ["hypoxia", "proliferation", "emt_like", "caf_matrix"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, program in zip(axes.ravel(), programs):
        values = pd.to_numeric(data.obs[f"{program}_score"], errors="coerce").to_numpy()
        scatter = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values,
            s=2,
            cmap="coolwarm",
            vmin=np.nanquantile(values, 0.02),
            vmax=np.nanquantile(values, 0.98),
            rasterized=True,
        )
        axis.set_title(program.replace("_", " "))
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(scatter, ax=axis, fraction=0.046)
    fig.tight_layout()
    _save(fig, path)


def plot_context_summary(support: pd.DataFrame, path: str | Path) -> None:
    status_score = {
        "supported_primary_context": 3,
        "supported_mixed_context": 2,
        "partially_supported": 1,
        "unclear_or_not_supported": 0,
        "unclear": 0,
    }
    values = support["support_status"].map(status_score).fillna(0)
    colors = values.map({3: "#2a9d8f", 2: "#6a994e", 1: "#e9c46a", 0: "#b0b0b0"})
    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    axis.barh(support["mechanism"], values, color=colors)
    axis.set_xlim(0, 3.2)
    axis.set_xticks([0, 1, 2, 3], ["unclear", "partial", "mixed", "primary"])
    axis.set_title("Raw scRNA cellular-context support")
    for index, row in support.reset_index(drop=True).iterrows():
        axis.text(0.05, index, row["top_observed_cell_types"], va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, path)
