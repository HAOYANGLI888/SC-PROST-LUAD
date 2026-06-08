"""Evaluate Stage 6A WSI pilot MIL models without making scientific claims."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.pathology_pilot_metrics import overfitting_diagnostics, summarize_pilot_predictions
from evaluation.pathology_visualization import plot_attention_coordinates
from evaluation.plot_survival import plot_km_high_low
from evaluation.survival_metrics import logrank_p_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage 6A WSI pilot model outputs.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    return parser


def _plot_model_comparison(performance: pd.DataFrame, output: Path) -> None:
    test = performance.loc[performance["split"] == "test"].copy()
    test["c_index"] = pd.to_numeric(test["c_index"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.barh(test["model_name"], test["c_index"], color="#2F7F7F")
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test C-index")
    ax.set_title("Stage 6A WSI pilot model comparison")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    pred_path = root / "data" / "processed" / "stage6a_wsi_pilot_predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"Missing predictions: {pred_path}")
    predictions = pd.read_csv(pred_path)
    performance = summarize_pilot_predictions(predictions)
    diagnostics = overfitting_diagnostics(performance)
    tables = root / "outputs" / "tables"
    figures = root / "outputs" / "figures"
    reports = root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    performance.to_csv(tables / "stage6a_wsi_pilot_model_performance.csv", index=False)
    diagnostics.to_csv(tables / "stage6a_wsi_pilot_overfitting_diagnostics.csv", index=False)
    _plot_model_comparison(performance, figures / "stage6a_wsi_pilot_model_comparison.png")
    test = performance.loc[performance["split"] == "test"].copy()
    test["c_index"] = pd.to_numeric(test["c_index"], errors="coerce")
    best_model = test.sort_values("c_index", ascending=False).iloc[0]["model_name"] if not test.empty else ""
    best_frame = predictions.loc[(predictions["model_name"] == best_model) & (predictions["split"] == "test")].copy()
    km_status = "pending"
    if len(best_frame) >= 10 and best_frame["os_event"].nunique() > 1:
        cutoff = float(best_frame["risk_score"].median())
        p_value = logrank_p_value(best_frame["os_time_days"], best_frame["os_event"], best_frame["risk_score"] >= cutoff)
        plot_km_high_low(best_frame["os_time_days"].to_numpy(), best_frame["os_event"].to_numpy(), best_frame["risk_score"].to_numpy(), cutoff, p_value, figures / "stage6a_wsi_pilot_km.png")
        km_status = f"generated for {best_model}, log-rank P={p_value:.4g}"
    attention_path = root / "data" / "processed" / "stage6a_wsi_pilot_attention_example.npz"
    if attention_path.exists():
        payload = np.load(attention_path, allow_pickle=True)
        plot_attention_coordinates(payload["coordinates"], payload["attention"], figures / "stage6a_wsi_pilot_attention_heatmap_example.png", title="Stage 6A WSI pilot attention example")
    report = (
        "# Stage 6A WSI GPU Pilot Model Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Integrity Boundary\n\n"
        "- This is a 100-slide WSI GPU pilot feasibility analysis, not a publication-grade model result.\n"
        "- No Stage 3, Stage 6B, mutation, CNV, methylation, single-cell, protein, or full 541-slide WSI download was started.\n\n"
        "## Model Summary\n\n"
        f"- Best pilot test C-index model: `{best_model}`.\n"
        f"- KM figure status: {km_status}.\n"
        "- Multivariable Cox and calibration are marked pending if event support is insufficient.\n"
        "- Any high pilot C-index should be interpreted as overfitting risk until validated with larger held-out WSI cohorts.\n\n"
    )
    (reports / "stage6a_wsi_pilot_model_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "passed", "best_test_model": best_model, "km_status": km_status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

