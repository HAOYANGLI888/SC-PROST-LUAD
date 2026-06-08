"""Run conservative QC and preprocessing for Stage 4B scRNA-seq data."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.scrna_qc import QCThresholds, preprocess_scrna  # noqa: E402
from data.scrna_raw_import import scrna_paths  # noqa: E402
from visualization.scrna_plots import plot_qc_violin, plot_umap  # noqa: E402


def _artifacts(small_test: bool) -> dict[str, Path]:
    suffix = "stage4b_small_test" if small_test else ""
    return {
        "processed": scrna_paths(ROOT, small_test=small_test).processed_dir / "scrna_luad_processed.h5ad",
        "table": ROOT / "outputs" / "tables" / suffix / "stage4b_scrna_qc_summary.csv",
        "violin": ROOT / "outputs" / "figures" / suffix / "stage4b_scrna_qc_violin.png",
        "umap": ROOT / "outputs" / "figures" / suffix / "stage4b_scrna_umap_overview.png",
        "report": ROOT / "outputs" / "reports" / suffix / "stage4b_scrna_qc_report.md",
    }


def _thresholds(config_path: Path, *, small_test: bool = False) -> QCThresholds:
    if small_test:
        return QCThresholds(
            min_genes=5,
            min_counts=10,
            max_genes=1_000,
            max_mito_fraction=90.0,
            min_cells_per_gene=1,
            n_hvg=80,
            n_pcs=12,
            n_neighbors=8,
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values = config.get("stage4b", {}).get("qc", {})
    allowed = QCThresholds.__dataclass_fields__
    return QCThresholds(**{key: value for key, value in values.items() if key in allowed})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Project YAML config path.")
    parser.add_argument("--small-test", action="store_true", help="Use isolated toy input.")
    parser.add_argument("--downsample-cells", type=int, default=None, help="Optional deterministic cell cap.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source = scrna_paths(ROOT, small_test=args.small_test).imported_h5ad
    if not source.exists():
        raise SystemExit(f"Imported scRNA h5ad missing: {source}. Run Stage 4B import first.")
    artifacts = _artifacts(args.small_test)
    thresholds = _thresholds(ROOT / args.config, small_test=args.small_test)
    summary, data = preprocess_scrna(
        source,
        artifacts["processed"],
        thresholds=thresholds,
        downsample_cells=args.downsample_cells,
    )
    artifacts["table"].parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(artifacts["table"], index=False)
    plot_qc_violin(data, artifacts["violin"])
    plot_umap(data, "Sample" if "Sample" in data.obs else data.obs.columns[0], artifacts["umap"], title="Stage 4B scRNA overview")
    artifacts["report"].parent.mkdir(parents=True, exist_ok=True)
    row = summary.iloc[0]
    artifacts["report"].write_text(
        f"""# Stage 4B scRNA QC Report

- Generated: {datetime.now().isoformat(timespec="seconds")}
- Mode: `{"toy_small_test" if args.small_test else "formal"}`
- Cells before QC: `{int(row.cells_before_qc)}`
- Cells after QC: `{int(row.cells_after_qc)}`
- Genes before QC: `{int(row.genes_before_qc)}`
- Genes after QC: `{int(row.genes_after_qc)}`
- Thresholds: min genes `{thresholds.min_genes}`, min counts `{thresholds.min_counts}`, max genes `{thresholds.max_genes}`, max mitochondrial fraction `{thresholds.max_mito_fraction}%`.
- Normalization: counts per `{thresholds.target_sum:g}` followed by log1p.
- Downsampled: `{bool(row.downsampled)}`

Filtering was intentionally conservative to preserve the structure of the public dataset.
""",
        encoding="utf-8",
    )
    print(f"Stage 4B QC complete: {artifacts['processed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
