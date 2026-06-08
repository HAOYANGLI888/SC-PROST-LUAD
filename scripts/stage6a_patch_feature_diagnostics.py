"""Patch and ResNet50 feature diagnostics for the Stage 6A WSI pilot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.diagnostics_utils import load_feature_artifact, load_pilot_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Stage 6A WSI pilot patches, tissue fractions and ResNet50 features.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    return parser


def _hist(values, output: Path, *, title: str, xlabel: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(values, bins=24, color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Slide count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _pca_plot(matrix: np.ndarray, labels, output: Path, *, title: str, label_name: str) -> None:
    scaled = StandardScaler().fit_transform(matrix)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    labels = pd.Series(labels).astype(str).fillna("NA")
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    palette = ["#2F7F7F", "#B64B3A", "#5A5F9F", "#D08A24", "#6A4C93", "#777777"]
    for index, value in enumerate(sorted(labels.unique())):
        mask = labels == value
        ax.scatter(coords[mask, 0], coords[mask, 1], s=38, alpha=0.85, label=f"{label_name}={value}", color=palette[index % len(palette)])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    tables = load_pilot_tables(root)
    patch_summary = tables["patch"].copy()
    feature_summary = tables["feature"].loc[tables["feature"]["feature_status"].isin(["extracted", "skipped_existing"])].copy()
    predictions_path = root / "data" / "processed" / "stage6a_wsi_pilot_predictions.csv"
    risk = pd.DataFrame()
    if predictions_path.exists():
        pred = pd.read_csv(predictions_path)
        risk = pred.loc[pred["model_name"].eq("clinical_only_cox"), ["patient_id", "risk_score"]].drop_duplicates("patient_id")
        risk["risk_group"] = np.where(risk["risk_score"] >= risk["risk_score"].median(), "high", "low")
    rows = []
    pooled = []
    labels = []
    stages = []
    risk_groups = []
    patch_counts = []
    tissue_means = []
    for row in feature_summary.sort_values("patient_id").to_dict("records"):
        feature_path = Path(row["feature_path"])
        artifact = load_feature_artifact(feature_path)
        tensor = artifact["features"].float()
        pooled.append(tensor.mean(dim=0).numpy())
        labels.append(int(row["os_event"]))
        stages.append(row.get("stage_numeric", np.nan))
        if not risk.empty and row["patient_id"] in set(risk["patient_id"]):
            risk_groups.append(risk.loc[risk["patient_id"] == row["patient_id"], "risk_group"].iloc[0])
        else:
            risk_groups.append("NA")
        patch_index = root / "data" / "processed" / "wsi_pilot_patches" / str(row["file_id"]) / "patch_index.csv"
        patch = pd.read_csv(patch_index)
        patch_counts.append(len(patch))
        tissue_means.append(float(patch["tissue_fraction"].mean()))
        width = float(row.get("width", np.nan))
        height = float(row.get("height", np.nan))
        x_cov = float((patch["x"].max() - patch["x"].min()) / width) if width and width > 0 else np.nan
        y_cov = float((patch["y"].max() - patch["y"].min()) / height) if height and height > 0 else np.nan
        rows.append(
            {
                "patient_id": row["patient_id"],
                "file_id": row["file_id"],
                "patch_count": len(patch),
                "mean_tissue_fraction": float(patch["tissue_fraction"].mean()),
                "min_tissue_fraction": float(patch["tissue_fraction"].min()),
                "x_coverage_fraction": x_cov,
                "y_coverage_fraction": y_cov,
                "feature_mean": float(tensor.mean()),
                "feature_std": float(tensor.std()),
                "feature_min": float(tensor.min()),
                "feature_max": float(tensor.max()),
                "feature_has_nan": bool(torch.isnan(tensor).any()),
                "feature_has_inf": bool(torch.isinf(tensor).any()),
                "feature_all_zero": bool(torch.all(tensor == 0)),
            }
        )
    table = pd.DataFrame(rows)
    tables_dir = root / "outputs" / "tables"
    figs = root / "outputs" / "figures"
    reports = root / "outputs" / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / "stage6a_patch_feature_diagnostics.csv", index=False)
    _hist(patch_counts, figs / "stage6a_patch_count_distribution.png", title="Patch count per pilot slide", xlabel="Patch count", color="#2F7F7F")
    _hist(tissue_means, figs / "stage6a_tissue_ratio_distribution.png", title="Mean tissue fraction per slide", xlabel="Mean tissue fraction", color="#B64B3A")
    matrix = np.vstack(pooled)
    _pca_plot(matrix, labels, figs / "stage6a_feature_pca_by_event.png", title="Pooled ResNet50 feature PCA by OS event", label_name="event")
    _pca_plot(matrix, stages, figs / "stage6a_feature_pca_by_stage.png", title="Pooled ResNet50 feature PCA by stage", label_name="stage")
    _pca_plot(matrix, risk_groups, figs / "stage6a_feature_pca_by_risk_group.png", title="Pooled ResNet50 feature PCA by clinical risk group", label_name="risk")
    report = (
        "# Stage 6A Patch And Feature Diagnostics Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"- Slides diagnosed: {len(table)}.\n"
        f"- Patch count mean/median/min/max: {np.mean(patch_counts):.1f}/{np.median(patch_counts):.1f}/{np.min(patch_counts)}/{np.max(patch_counts)}.\n"
        f"- Mean tissue fraction mean/median/min/max: {np.mean(tissue_means):.3f}/{np.median(tissue_means):.3f}/{np.min(tissue_means):.3f}/{np.max(tissue_means):.3f}.\n"
        f"- Feature NaN slides: {int(table['feature_has_nan'].sum())}; Inf slides: {int(table['feature_has_inf'].sum())}; all-zero slides: {int(table['feature_all_zero'].sum())}.\n"
        "- PCA figures were generated for event, stage and clinical-risk groups. Separation should be interpreted visually and cautiously.\n"
        "- Existing QC tissue mask and patch grid images were reused as real-patch visual checks; patch extraction did not save all PNG patches.\n"
    )
    (reports / "stage6a_patch_feature_diagnostics_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "passed", "slides": len(table), "mean_patch_count": float(np.mean(patch_counts)), "nan_slides": int(table["feature_has_nan"].sum())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

