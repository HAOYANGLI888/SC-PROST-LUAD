"""Interpret full-cohort fixed-gene and program localization conservatively."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.scrna_celltype_summary import cellular_context_summary  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/base.yaml")
    return result


def main(argv: list[str] | None = None) -> int:
    parser().parse_args(argv)
    table_dir = ROOT / "outputs" / "tables"
    report_dir = ROOT / "outputs" / "reports"
    scores = pd.read_csv(
        table_dir / "stage4b_lowmem_celltype_program_scores.csv"
    )
    expression = pd.read_csv(
        table_dir / "stage4b_lowmem_gene_celltype_expression.csv"
    )
    summary, support = cellular_context_summary(scores, expression)
    summary_path = table_dir / "stage4b_lowmem_cellular_context_summary.csv"
    support_path = table_dir / "stage4b_lowmem_mechanism_support_matrix.csv"
    report_path = (
        report_dir / "stage4b_lowmem_cellular_context_validation_report.md"
    )
    summary.to_csv(summary_path, index=False)
    support.to_csv(support_path, index=False)

    gene_rows = summary.loc[
        summary["question"].str.startswith("Primary cell types")
    ]
    program_rows = support.set_index("mechanism")
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
    cautious = support.loc[~strong, "mechanism"].astype(str).tolist()
    marker_lines = []
    for label, genes in {
        "dendritic": ["CD74", "HLA-DRA", "HLA-DRB1"],
        "B-cell": ["MS4A1", "CD79A", "CD79B"],
        "plasma-cell": ["MZB1", "JCHAIN", "XBP1"],
    }.items():
        marker_subset = expression.loc[expression["gene_symbol"].isin(genes)]
        top_by_gene = []
        for gene in genes:
            marker_gene_rows = marker_subset.loc[
                marker_subset["gene_symbol"].eq(gene)
            ].sort_values("average_log1p_cpm", ascending=False)
            if not marker_gene_rows.empty:
                top_by_gene.append(
                    f"{gene}:{marker_gene_rows.iloc[0]['analysis_cell_type']}"
                )
        marker_lines.append(f"{label} markers " + ", ".join(top_by_gene))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"""# Stage 4B-LowMem Cellular-Context Validation

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Dataset: `GSE131907 raw UMI`
- Cells / genes: `208506 / 29634`
- Annotation: official author-provided annotation, conservatively harmonized.
- Scoring: full-cohort chunked mean `log1p(CPM10000)` for fixed signatures.

## Required Questions

1. **LDHA localization:** `{gene_rows.iloc[0]["result"]}` ({gene_rows.iloc[0]["metric"]}).
2. **MKI67 and CDK1 localization:** `{gene_rows.iloc[1]["result"]}` and `{gene_rows.iloc[2]["result"]}`. Both rank T cells first and malignant epithelial cells second; therefore enrichment is not specific to proliferative malignant epithelium.
3. **Hypoxia context:** `{program_rows.loc["hypoxia", "top_observed_cell_types"]}`; status `{program_rows.loc["hypoxia", "support_status"]}`.
4. **Proliferation context:** `{program_rows.loc["proliferation", "top_observed_cell_types"]}`; status `{program_rows.loc["proliferation", "support_status"]}`.
5. **EMT-like context:** `{program_rows.loc["emt_like", "top_observed_cell_types"]}`; status `{program_rows.loc["emt_like", "support_status"]}`. The fixed bulk signature localizes mainly to stromal/myeloid contexts rather than malignant epithelial cells.
6. **CAF/matrix context:** `{program_rows.loc["caf_matrix", "top_observed_cell_types"]}`; status `{program_rows.loc["caf_matrix", "support_status"]}`.
7. **Dendritic/B/plasma context:** `{program_rows.loc["dendritic_b_plasma_context", "top_observed_cell_types"]}`; status `{program_rows.loc["dendritic_b_plasma_context", "support_status"]}`. Individual-marker localization: `{"; ".join(marker_lines)}`.
8. **Overall support:** `{int(strong.sum())}` layers have primary/mixed-context support and `{int((supported & ~strong).sum())}` have limited contextual support.
9. **Weak or unclear layers:** `{";".join(cautious) if cautious else "none"}` require cautious wording because their leading context is not the prespecified malignant compartment.
10. **Permitted wording:** `{summary.loc[summary.question.eq("Can wording be upgraded?"), "result"].iloc[0]}`.

## Mechanism Support Matrix

{support.to_markdown(index=False)}

## Scientific Boundary

This is raw scRNA-supported cellular-context interpretation. It is not a
single-cell survival validation, does not retrain or reselect Stage 2D genes,
and does not establish a causal mechanism.
""",
        encoding="utf-8",
    )
    print(f"Stage 4B-LowMem validation complete: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
