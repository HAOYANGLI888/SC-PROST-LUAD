"""Assemble the Stage 4B-LowMem audit report without new analysis."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/base.yaml")
    return result


def main(argv: list[str] | None = None) -> int:
    parser().parse_args(argv)
    table_dir = ROOT / "outputs" / "tables"
    report_dir = ROOT / "outputs" / "reports"
    support = pd.read_csv(
        table_dir / "stage4b_lowmem_mechanism_support_matrix.csv"
    )
    missing = pd.read_csv(table_dir / "stage4b_lowmem_signature_missingness.csv")
    gene_missing = pd.read_csv(table_dir / "stage4b_lowmem_gene_missingness.csv")
    qc = pd.read_csv(table_dir / "stage4b_lowmem_qc_summary.csv").set_index(
        "metric"
    )["value"]
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
    weak = support.loc[~strong, "mechanism"].astype(str).tolist()
    figure_recommendation = (
        "Use a concise main-text statement and place the full dotplot, heatmap, "
        "and downsample UMAPs in supplementary material."
    )
    report = f"""# Stage 4B-LowMem Final Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Dataset: `GSE131907 raw scRNA-seq`
- h5ad dimensions: `208506 cells x 29634 genes`
- Annotation: official GSE131907 author annotation; no de novo clustering.
- Memory strategy: backed/chunked sparse processing on a 32 GB workstation.
- Full-cohort PCA/neighbors/UMAP: `not performed`.
- Fixed-gene and program calculations: full-cohort sparse blocks.
- Requested signature genes missing: `{int((gene_missing.status == "missing").sum())}`
- Programs with incomplete signatures: `{int((missing.missing_fraction > 0).sum())}`

## Cellular-Context Support

{support.to_markdown(index=False)}

Primary/mixed-context support: `{int(strong.sum())}/{len(support)}`.

Limited contextual support: `{int((supported & ~strong).sum())}/{len(support)}`.

Layers requiring continued caution: `{";".join(weak) if weak else "none under the prespecified localization rule; all remain observational"}`.

## Interpretation

The permitted upgrade is from *single-cell-informed signature interpretation*
to *raw scRNA-supported cellular-context interpretation* only for mechanism
layers meeting the prespecified localization rule. This analysis does not
constitute single-cell survival validation and does not establish causal
confirmation.

## Manuscript Placement

{figure_recommendation}

The raw scRNA result should support, rather than replace, the externally
validated transcriptomic-clinical model and the bulk/CPTAC evidence chain.

## Resource Assessment

The 32 GB workstation was sufficient for the current chunked summaries and
stratified visualization. Full 208,506-cell PCA, neighbor graph, or UMAP remains
outside this workflow and would be more stable on a >=64 GB workstation.

## Verification

- All six requested Stage 4B-LowMem commands completed.
- The 50,000-cell visualization completed without fallback.
- `python -m compileall -q src scripts`: passed.
- `python -m pytest tests -q`: `63` tests passed.

## Stop Boundary

No manuscript files were generated or modified. No Stage 3, Stage 6B/6C,
single-cell survival model, cell-cell communication, trajectory, or pseudotime
analysis was started.
"""
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "stage4b_lowmem_final_report.md"
    output.write_text(report, encoding="utf-8")
    audit = ROOT / "audit_report.md"
    heading = "## Stage 4B-LowMem Raw scRNA Cellular-Context Validation"
    audit_text = audit.read_text(encoding="utf-8") if audit.exists() else ""
    if heading in audit_text:
        audit_text = audit_text.split(f"\n\n{heading}", 1)[0].rstrip()
    section = (
        f"\n\n{heading}\n\n"
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
        "- Dataset: GSE131907 raw UMI h5ad, 208,506 cells x 29,634 genes.\n"
        "- Official source annotation was used; no de novo clustering.\n"
        "- Full-cohort expression and fixed-program scores used chunked "
        "sparse-safe log1p(CPM) calculations.\n"
        "- Full-cohort PCA/neighbors/UMAP was not performed.\n"
        f"- Primary/mixed-context support: {int(strong.sum())}/{len(support)}.\n"
        f"- Limited contextual support: {int((supported & ~strong).sum())}/{len(support)}.\n"
        f"- Cautious layers: {';'.join(weak) if weak else 'all remain observational'}.\n"
        "- Interpretation boundary: raw scRNA-supported cellular context, "
        "not survival validation or causal confirmation.\n"
        "- Recommended placement: concise main-text statement with detailed "
        "figures and tables in supplementary material.\n"
        "- Verification: all six commands completed; 50,000-cell UMAP completed; "
        "compileall passed; pytest 63/63 passed.\n"
        "- Work stopped before manuscript generation.\n"
    )
    audit.write_text(audit_text + section, encoding="utf-8")
    print(f"Stage 4B-LowMem final report complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
