"""Evaluate low-complexity pathology baselines for Stage 6A diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize low-complexity pathology baseline performance and overfitting.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    return parser


def _plot(performance: pd.DataFrame, output: Path) -> None:
    test = performance.loc[performance["split"].eq("test")].copy()
    test["c_index"] = pd.to_numeric(test["c_index"], errors="coerce")
    test = test.sort_values("c_index", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(test["model_name"], test["c_index"], color="#3B7A78")
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Test C-index")
    ax.set_title("Low-complexity WSI pilot baselines")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    perf_path = root / "outputs" / "tables" / "stage6a_low_complexity_pathology_performance.csv"
    diag_path = root / "outputs" / "tables" / "stage6a_low_complexity_overfitting_diagnostics.csv"
    performance = pd.read_csv(perf_path)
    diagnostics = pd.read_csv(diag_path)
    figures = root / "outputs" / "figures"
    reports = root / "outputs" / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    _plot(performance, figures / "stage6a_low_complexity_model_comparison.png")
    test = performance.loc[performance["split"].eq("test")].copy()
    test["c_index"] = pd.to_numeric(test["c_index"], errors="coerce")
    clinical = float(test.loc[test["model_name"].eq("clinical_only_cox_low_complexity"), "c_index"].iloc[0])
    pathology = test.loc[test["model_name"].str.contains("pathology") & ~test["model_name"].str.contains("^clinical_", regex=True)]
    best_pathology = pathology.sort_values("c_index", ascending=False).iloc[0]
    fusion = test.loc[test["model_name"].str.startswith("clinical_") & test["model_name"].str.contains("pathology")]
    best_fusion = fusion.sort_values("c_index", ascending=False).iloc[0]
    report = (
        "# Stage 6A Low-Complexity Pathology Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"- Clinical-only test C-index: {clinical:.3f}.\n"
        f"- Best pathology-only low-complexity model: `{best_pathology['model_name']}`, test C-index={float(best_pathology['c_index']):.3f}.\n"
        f"- Best clinical+pathology low-complexity model: `{best_fusion['model_name']}`, test C-index={float(best_fusion['c_index']):.3f}.\n"
        f"- Models flagged for overfitting: {int(diagnostics['overfitting_flag'].astype(bool).sum())}/{len(diagnostics)}.\n"
        + (
            "- Low-complexity pathology did not outperform clinical-only; this argues against a stable WSI prognostic signal in the current pilot.\n"
            if float(best_fusion["c_index"]) <= clinical and float(best_pathology["c_index"]) <= clinical
            else "- Some low-complexity pathology signal is present, but this remains pilot-only evidence.\n"
        )
    )
    (reports / "stage6a_low_complexity_pathology_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "passed", "clinical_test_c": clinical, "best_pathology_test_c": float(best_pathology["c_index"]), "best_fusion_test_c": float(best_fusion["c_index"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

