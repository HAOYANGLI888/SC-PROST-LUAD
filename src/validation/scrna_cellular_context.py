"""Cellular-context validation for prespecified Stage 4/5B mechanisms."""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPECTED_CONTEXTS = {
    "hypoxia": {"Malignant epithelial", "Epithelial"},
    "proliferation": {"Malignant epithelial"},
    "emt_like": {"Malignant epithelial", "Epithelial", "Fibroblasts/CAF"},
    "caf_matrix": {"Fibroblasts/CAF"},
    "dendritic_b_plasma_context": {"Dendritic cells", "B cells", "Plasma cells"},
}


def _top_cell_types(
    celltype_scores: pd.DataFrame, score_column: str, *, n: int = 3
) -> list[str]:
    if score_column not in celltype_scores:
        return []
    return (
        celltype_scores.sort_values(score_column, ascending=False)
        .head(n)["analysis_cell_type"]
        .astype(str)
        .tolist()
    )


def build_mechanism_support_matrix(
    celltype_scores: pd.DataFrame,
    gene_expression: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for program, expected in EXPECTED_CONTEXTS.items():
        score_column = f"{program}_score_mean"
        top = _top_cell_types(celltype_scores, score_column)
        overlap = expected.intersection(top)
        if program == "emt_like":
            status = "supported_mixed_context" if overlap else "unclear"
        elif overlap and top and top[0] in expected:
            status = "supported_primary_context"
        elif overlap:
            status = "partially_supported"
        else:
            status = "unclear_or_not_supported"
        rows.append(
            {
                "mechanism": program,
                "expected_cellular_context": ";".join(sorted(expected)),
                "top_observed_cell_types": ";".join(top),
                "expected_context_in_top3": bool(overlap),
                "support_status": status,
                "interpretation": (
                    "Raw scRNA cellular-context association only; no survival or causal inference."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_cellular_context_summary(
    celltype_scores: pd.DataFrame,
    gene_expression: pd.DataFrame,
    support: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for gene in ("LDHA", "MKI67", "CDK1"):
        subset = gene_expression.loc[gene_expression["gene_symbol"].eq(gene)]
        subset = subset.sort_values(
            ["average_log_expression", "detection_fraction"], ascending=False
        )
        rows.append(
            {
                "question": f"Primary cell types expressing {gene}",
                "result": ";".join(subset.head(3)["analysis_cell_type"].astype(str)),
                "metric": ";".join(
                    f"{row.analysis_cell_type}:avg={row.average_log_expression:.3f},det={row.detection_fraction:.3f}"
                    for row in subset.head(3).itertuples(index=False)
                ),
                "status": "observed" if not subset.empty else "unavailable",
            }
        )
    for row in support.itertuples(index=False):
        rows.append(
            {
                "question": f"Cellular context for {row.mechanism}",
                "result": row.top_observed_cell_types,
                "metric": row.expected_cellular_context,
                "status": row.support_status,
            }
        )
    supported = support["support_status"].isin(
        {"supported_primary_context", "supported_mixed_context", "partially_supported"}
    )
    rows.append(
        {
            "question": "Can wording be upgraded?",
            "result": (
                "raw scRNA-supported cellular-context interpretation"
                if bool(supported.all())
                else "retain single-cell-informed interpretation for unsupported layers"
            ),
            "metric": f"{int(supported.sum())}/{len(support)} mechanisms with contextual support",
            "status": "bounded_interpretation_only",
        }
    )
    return pd.DataFrame(rows)

