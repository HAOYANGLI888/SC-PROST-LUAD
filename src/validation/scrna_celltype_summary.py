"""Rule-based interpretation of fixed programs in official cell annotations."""

from __future__ import annotations

import pandas as pd


EXPECTED_CONTEXTS = {
    "hypoxia": {"Malignant epithelial", "Epithelial"},
    "proliferation": {"Malignant epithelial", "Epithelial"},
    "emt_like": {"Malignant epithelial", "Epithelial", "Fibroblasts/CAF"},
    "caf_matrix": {"Fibroblasts/CAF"},
    "dendritic_b_plasma_context": {
        "Dendritic cells",
        "B cells",
        "Plasma cells",
    },
}


def rank_program_contexts(celltype_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for program, expected in EXPECTED_CONTEXTS.items():
        subset = celltype_scores.loc[
            celltype_scores["program"].eq(program)
        ].sort_values("mean_log1p_cpm_score", ascending=False)
        top = subset.head(3)["analysis_cell_type"].astype(str).tolist()
        overlap = expected.intersection(top)
        if program == "emt_like":
            malignant = bool(
                {"Malignant epithelial", "Epithelial"}.intersection(top)
            )
            fibroblast = "Fibroblasts/CAF" in top
            if malignant and fibroblast:
                status = "supported_mixed_context"
            elif fibroblast:
                status = "supported_stromal_context_only"
            elif malignant:
                status = "partially_supported"
            else:
                status = "unclear_or_not_supported"
        elif top and top[0] in expected:
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
                "top_observed_score": (
                    float(subset.iloc[0]["mean_log1p_cpm_score"])
                    if not subset.empty
                    else float("nan")
                ),
                "expected_context_in_top3": bool(overlap),
                "support_status": status,
                "evidence_scope": (
                    "raw scRNA cellular-context association; "
                    "not survival validation or causal confirmation"
                ),
            }
        )
    return pd.DataFrame(rows)


def top_gene_contexts(gene_expression: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene in ("LDHA", "MKI67", "CDK1"):
        subset = gene_expression.loc[
            gene_expression["gene_symbol"].eq(gene)
        ].sort_values(
            ["average_log1p_cpm", "detection_rate"],
            ascending=False,
        )
        rows.append(
            {
                "question": f"Primary cell types expressing {gene}",
                "result": ";".join(
                    subset.head(3)["analysis_cell_type"].astype(str)
                ),
                "metric": ";".join(
                    (
                        f"{row.analysis_cell_type}:"
                        f"avg={row.average_log1p_cpm:.3f},"
                        f"det={row.detection_rate:.3f}"
                    )
                    for row in subset.head(3).itertuples(index=False)
                ),
                "status": "observed" if not subset.empty else "unavailable",
            }
        )
    return pd.DataFrame(rows)


def cellular_context_summary(
    celltype_scores: pd.DataFrame,
    gene_expression: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = rank_program_contexts(celltype_scores)
    summary = top_gene_contexts(gene_expression)
    mechanism_rows = pd.DataFrame(
        [
            {
                "question": f"Cellular context for {row.mechanism}",
                "result": row.top_observed_cell_types,
                "metric": row.expected_cellular_context,
                "status": row.support_status,
            }
            for row in support.itertuples(index=False)
        ]
    )
    supported = support["support_status"].isin(
        {
            "supported_primary_context",
            "supported_mixed_context",
            "supported_stromal_context_only",
            "partially_supported",
        }
    )
    strong = support["support_status"].isin(
        {"supported_primary_context", "supported_mixed_context"}
    )
    wording = pd.DataFrame(
        [
            {
                "question": "Can wording be upgraded?",
                "result": (
                    "raw scRNA-supported cellular-context interpretation"
                    if bool(supported.all())
                    else (
                        "use raw scRNA-supported wording only for supported "
                        "mechanisms; retain cautious wording elsewhere"
                    )
                ),
                "metric": f"{int(supported.sum())}/{len(support)} mechanisms supported",
                "status": (
                    f"bounded_interpretation_only;"
                    f"strong={int(strong.sum())};limited={int((supported & ~strong).sum())}"
                ),
            }
        ]
    )
    return pd.concat([summary, mechanism_rows, wording], ignore_index=True), support
