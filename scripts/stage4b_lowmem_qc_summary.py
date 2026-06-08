"""Compute full-cohort raw scRNA QC summaries with sparse row blocks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_raw_import import scrna_paths  # noqa: E402
from validation.scrna_lowmem_reader import compute_lowmem_qc  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/base.yaml")
    result.add_argument("--block-size", type=int, default=2048)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    source = scrna_paths(ROOT).raw_or_converted_h5ad
    if not source.exists():
        raise SystemExit(f"Validated h5ad is missing: {source}")
    summary, counts, _ = compute_lowmem_qc(
        source, block_size=args.block_size
    )
    table_dir = ROOT / "outputs" / "tables"
    report_dir = ROOT / "outputs" / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = table_dir / "stage4b_lowmem_qc_summary.csv"
    counts_path = table_dir / "stage4b_lowmem_celltype_counts.csv"
    report_path = report_dir / "stage4b_lowmem_qc_report.md"
    summary.to_csv(summary_path, index=False)
    counts.to_csv(counts_path, index=False)
    values = summary.set_index("metric")["value"]
    report_path.write_text(
        f"""# Stage 4B-LowMem QC Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Input: `{source}`
- Cells / genes: `{int(float(values["n_cells"]))} / {int(float(values["n_genes"]))}`
- Nonzero counts: `{int(float(values["nonzero_counts"]))}`
- Matrix density: `{float(values["matrix_density"]):.6f}`
- Samples / patients / harmonized cell types: `{int(float(values["n_samples"]))} / {int(float(values["n_patients"]))} / {int(float(values["n_analysis_cell_types"]))}`
- Median total counts per cell: `{float(values["total_counts_median"]):.1f}`
- Median detected genes per cell: `{float(values["detected_genes_median"]):.1f}`
- Read mode: sequential sparse CSR blocks of `{args.block_size}` cells.

## Boundary

This is a raw-count summary, not a filtering decision. The original h5ad was
not modified. No full-matrix dense conversion, PCA, neighbors, UMAP, survival
analysis, or biological inference was performed.
""",
        encoding="utf-8",
    )
    print(f"Stage 4B-LowMem QC complete: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
