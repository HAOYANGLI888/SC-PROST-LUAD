"""Cell-type harmonization with source-annotation priority and marker fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from data.scrna_malignant_cell import epithelial_confidence_label
from data.scrna_raw_import import sanitize_anndata_strings


MARKERS = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19"],
    "T cells": ["CD3D", "CD3E", "TRAC"],
    "CD8 T cells": ["CD8A", "CD8B"],
    "CD4 T cells": ["CD4", "IL7R"],
    "B cells": ["MS4A1", "CD79A", "CD79B"],
    "Plasma cells": ["MZB1", "JCHAIN", "XBP1"],
    "Macrophages/monocytes": ["LST1", "C1QA", "C1QB", "CD68"],
    "Dendritic cells": ["CD74", "HLA-DRA", "CLEC9A", "FCER1A", "LILRA4"],
    "Fibroblasts/CAF": ["COL1A1", "COL1A2", "COL3A1", "DCN", "ACTA2", "FAP", "PDGFRB"],
    "Endothelial cells": ["PECAM1", "VWF", "KDR"],
    "Mast cells": ["TPSAB1", "TPSB2", "KIT"],
    "NK cells": ["NKG7", "GNLY", "KLRD1"],
}


def _source_harmonized_labels(obs: pd.DataFrame) -> pd.Series | None:
    if not any(
        column in obs.columns
        for column in ("Cell_type.refined", "Cell_type", "Cell_subtype")
    ):
        return None
    refined = obs.get(
        "Cell_type.refined", obs.get("Cell_type", pd.Series("", index=obs.index))
    ).astype(str)
    subtype = obs.get("Cell_subtype", pd.Series("", index=obs.index)).astype(str)
    labels = pd.Series("Other/undetermined", index=obs.index, dtype=object)
    labels.loc[refined.str.contains("epithelial", case=False, na=False)] = "Epithelial"
    labels.loc[subtype.str.contains("malignant", case=False, na=False)] = "Malignant epithelial"
    labels.loc[refined.str.contains("fibro", case=False, na=False)] = "Fibroblasts/CAF"
    labels.loc[refined.str.contains("endothelial", case=False, na=False)] = "Endothelial cells"
    labels.loc[refined.str.contains("mast", case=False, na=False)] = "Mast cells"
    labels.loc[refined.str.contains("myeloid", case=False, na=False)] = "Macrophages/monocytes"
    labels.loc[subtype.str.contains(r"\bDC|dendritic|pDC|CD1c|CD141", case=False, regex=True, na=False)] = "Dendritic cells"
    labels.loc[refined.str.contains(r"\bB lymph", case=False, regex=True, na=False)] = "B cells"
    labels.loc[subtype.str.contains("plasma", case=False, na=False)] = "Plasma cells"
    labels.loc[refined.str.contains("T/NK|T lymph", case=False, regex=True, na=False)] = "T cells"
    labels.loc[subtype.str.contains("NK", case=False, na=False)] = "NK cells"
    labels.loc[subtype.str.contains("CD8", case=False, na=False)] = "CD8 T cells"
    labels.loc[subtype.str.contains("CD4|Treg|Tfh", case=False, regex=True, na=False)] = "CD4 T cells"
    return labels


def _marker_labels(data) -> pd.Series:
    genes = {gene.upper(): index for index, gene in enumerate(data.var_names)}
    scores = {}
    for label, markers in MARKERS.items():
        indices = [genes[gene] for gene in markers if gene in genes]
        if not indices:
            scores[label] = np.full(data.n_obs, -np.inf)
            continue
        matrix = data.X[:, indices]
        values = np.asarray(matrix.mean(axis=1)).ravel()
        scores[label] = values
    frame = pd.DataFrame(scores, index=data.obs_names)
    return frame.idxmax(axis=1).where(frame.max(axis=1) > 0, "Other/undetermined")


def annotate_cells(input_h5ad: str | Path, output_h5ad: str | Path):
    import scanpy as sc

    data = sc.read_h5ad(input_h5ad)
    source = _source_harmonized_labels(data.obs)
    if source is not None and source.ne("Other/undetermined").mean() >= 0.5:
        labels = source
        status = "source_annotation_preserved_and_harmonized"
    else:
        labels = _marker_labels(data)
        status = "marker_based_fallback"
    data.obs["analysis_cell_type"] = labels.astype(str).to_numpy()
    data.obs["epithelial_annotation_confidence"] = epithelial_confidence_label(
        data.obs
    ).to_numpy()
    data.uns["stage4b_annotation_method"] = status
    output = Path(output_h5ad)
    output.parent.mkdir(parents=True, exist_ok=True)
    sanitize_anndata_strings(data).write_h5ad(output, compression="gzip")
    counts = (
        data.obs.groupby("analysis_cell_type", observed=True)
        .size()
        .rename("cell_count")
        .reset_index()
        .sort_values("cell_count", ascending=False)
    )
    counts["fraction"] = counts["cell_count"] / counts["cell_count"].sum()
    return counts, marker_support_table(data), data


def marker_support_table(data) -> pd.DataFrame:
    rows = []
    gene_lookup = {str(gene).upper(): index for index, gene in enumerate(data.var_names)}
    labels = data.obs["analysis_cell_type"].astype(str)
    for cell_type in sorted(labels.unique()):
        mask = labels.eq(cell_type).to_numpy()
        for marker_group, markers in MARKERS.items():
            available = [gene for gene in markers if gene in gene_lookup]
            if not available:
                continue
            indices = [gene_lookup[gene] for gene in available]
            matrix = data.X[mask][:, indices]
            average = float(np.asarray(matrix.mean()).ravel()[0])
            detection = float(
                matrix.getnnz() / (matrix.shape[0] * matrix.shape[1])
                if sparse.issparse(matrix)
                else np.mean(np.asarray(matrix) > 0)
            )
            rows.append(
                {
                    "analysis_cell_type": cell_type,
                    "marker_group": marker_group,
                    "available_markers": ";".join(available),
                    "n_markers": len(available),
                    "mean_log_expression": average,
                    "detection_fraction": detection,
                }
            )
    return pd.DataFrame(rows)
