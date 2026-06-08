"""Generate the final Stage 4B raw scRNA cellular-context report."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_raw_import import scrna_paths  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--small-test", action="store_true", help="Use isolated toy outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    suffix = "stage4b_small_test" if args.small_test else ""
    table_dir = ROOT / "outputs" / "tables" / suffix
    report_dir = ROOT / "outputs" / "reports" / suffix
    processed = scrna_paths(ROOT, small_test=args.small_test).processed_dir
    scored_path = processed / "scrna_luad_scored.h5ad"
    required = [
        table_dir / "stage4b_scrna_qc_summary.csv",
        table_dir / "stage4b_scrna_cell_type_counts.csv",
        table_dir / "stage4b_scrna_cellular_context_summary.csv",
        table_dir / "stage4b_scrna_mechanism_support_matrix.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing or not scored_path.exists():
        if args.small_test:
            raise SystemExit("Stage 4B outputs missing: " + "; ".join(missing))
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "stage4b_scrna_final_report.md"
        inventory_path = table_dir / "stage4b_scrna_data_inventory.csv"
        inventory = (
            pd.read_csv(inventory_path)
            if inventory_path.exists()
            else pd.DataFrame()
        )
        raw_paths = scrna_paths(ROOT, small_test=False)
        parts_dir = raw_paths.raw_rds.with_name(raw_paths.raw_rds.name + ".parts")
        prefix_bytes = raw_paths.raw_rds.stat().st_size if raw_paths.raw_rds.exists() else 0
        range_part_bytes = (
            sum(path.stat().st_size for path in parts_dir.glob("*") if path.is_file())
            if parts_dir.exists()
            else 0
        )
        retained_bytes = prefix_bytes + range_part_bytes
        retained_percent = 100 * retained_bytes / 633_500_069
        report_path.write_text(
            f"""# Stage 4B Raw LUAD scRNA Cellular-Context Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Status: `manual_download_required`
- Preferred dataset: `GSE131907`
- Formal raw-scRNA analysis completed: `no`

## Data Availability

The official GSE131907 cell annotation was downloaded and is readable, but the
633,500,069-byte raw UMI RDS was not fully downloaded during this run. Existing
prefix and range-part files were retained for resume (`{retained_bytes / 1024**2:.1f} MB`,
`{retained_percent:.1f}%` of the expected file). No formal QC, annotation,
program scoring, cellular-context validation, or scientific figure was generated
from the incomplete expression matrix.

## Required Questions

1. Dataset used: `unavailable for formal analysis; GSE131907 download incomplete`.
2. LUAD/NSCLC scope: `GSE131907 is the planned public lung adenocarcinoma dataset`.
3. Total cells/samples/patients: `annotation reports 208,506 cells and 58 samples; patient count must be finalized after complete expression import`.
4. Original cell-type annotation: `yes, official annotation downloaded`.
5. Cells retained after QC: `unavailable`.
6. Major cell types after QC: `unavailable for formal expression analysis`.
7. LDHA/MKI67/CDK1 cellular sources: `unavailable`.
8. Program cellular sources: `unavailable`.
9. Stage 4 mechanisms with raw scRNA support: `none claimed`.
10. Unstable mechanisms: `not evaluated`.
11. Boundary: this is not single-cell survival validation and is not causal confirmation.
12. Figure recommendation: `do not use Stage 4B as a main or supplementary scientific figure until the real matrix is complete`.

## Inventory

{inventory.to_markdown(index=False) if not inventory.empty else "Inventory unavailable."}

## Next Command

```powershell
conda activate gpu_py310
python scripts/stage4b_scrna_download_or_import.py --config configs/base.yaml --workers 4
```

See `docs/stage4b_scrna_manual_download_guide.md`. Rerunning the command retains
complete files and resumes saved range parts.
""",
            encoding="utf-8",
        )
        audit_path = ROOT / "audit_report.md"
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## Stage 4B Raw scRNA Cellular Context\n\n"
                f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
                "- Formal status: `manual_download_required`.\n"
                "- GSE131907 annotation: downloaded and readable.\n"
                "- Raw UMI RDS: incomplete; no formal cellular-context result generated.\n"
                f"- Final status report: `{report_path}`.\n"
                "- Integrity: toy smoke-test results were not used as scientific evidence.\n"
            )
        print(f"Stage 4B stopped: raw data incomplete -> {report_path}")
        return 2
    import scanpy as sc

    data = sc.read_h5ad(scored_path, backed="r")
    qc = pd.read_csv(required[0]).iloc[0]
    counts = pd.read_csv(required[1])
    summary = pd.read_csv(required[2])
    support = pd.read_csv(required[3])
    samples = data.obs["Sample"].astype(str) if "Sample" in data.obs else pd.Series([], dtype=str)
    patient_ids = {
        match.group(1)
        for value in samples.unique()
        if (match := re.search(r"(\d+)$", value))
    }
    source = str(data.uns.get("dataset_accession", "unknown"))
    source_annotation = bool(
        {"Cell_type", "Cell_type.refined", "Cell_subtype"} & set(data.obs.columns)
    )
    key_rows = summary.loc[summary["question"].str.startswith("Primary cell types")]
    mechanism_rows = support[["mechanism", "top_observed_cell_types", "support_status"]]
    recommendation = (
        "Main-text mechanistic figure candidate"
        if not args.small_test
        and support["support_status"].isin(
            {"supported_primary_context", "supported_mixed_context", "partially_supported"}
        ).sum()
        >= 4
        else "Supplementary Figure until real-data support is complete"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "stage4b_scrna_final_report.md"
    report_path.write_text(
        f"""# Stage 4B Raw LUAD scRNA Cellular-Context Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Analysis mode: `{"toy_small_test_not_scientific" if args.small_test else "formal_real_data"}`
- Dataset: `{source}`
- Disease scope: `NSCLC/LUAD public single-cell dataset; GSE131907 contains LUAD primary and metastatic/adjacent compartments`
- Cells before QC: `{int(qc.cells_before_qc)}`
- Cells after QC: `{int(qc.cells_after_qc)}`
- Samples: `{samples.nunique()}`
- Patient identifiers inferred from sample suffixes: `{len(patient_ids)}`
- Original cell-type annotation available: `{"yes" if source_annotation else "no"}`

## Major Cell Types

{counts.to_markdown(index=False)}

## LDHA, MKI67 And CDK1 Cellular Sources

{key_rows.to_markdown(index=False)}

## Program Cellular Context

{mechanism_rows.to_markdown(index=False)}

## Interpretation

Mechanisms labelled supported in the table have raw scRNA cellular-context support for localization only. Unclear or mixed layers should not be overinterpreted. The fixed Stage 4/5B gene sets were not modified after inspecting scRNA results.

This is not single-cell survival validation and it is not causal confirmation. No survival outcome was used, and the Stage 2D model was neither retrained nor reselected.

## Figure Recommendation

`{recommendation}`.
""",
        encoding="utf-8",
    )
    data.file.close()
    audit_path = (
        ROOT / "outputs" / "audit" / "stage4b_small_test" / "audit_report.md"
        if args.small_test
        else ROOT / "audit_report.md"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Stage 4B Raw scRNA Cellular Context\n\n"
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}.\n"
            f"- Mode: `{'toy_small_test' if args.small_test else 'formal_real_data'}`.\n"
            f"- Dataset: `{source}`.\n"
            f"- QC cells: `{int(qc.cells_before_qc)}` -> `{int(qc.cells_after_qc)}`.\n"
            f"- Final report: `{report_path}`.\n"
            "- Integrity: no survival analysis, model retraining, causal claim, or manuscript generation.\n"
        )
    print(f"Stage 4B final report generated: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
