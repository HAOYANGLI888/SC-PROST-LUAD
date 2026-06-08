"""Resource accounting for the Stage 6A GPU pilot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def directory_size(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(file.stat().st_size for file in root.rglob("*") if file.is_file())


def collect_resource_rows(root: str | Path = ".") -> dict[str, float]:
    root = Path(root).resolve()
    download_status = root / "data" / "metadata" / "stage6a_wsi_pilot_download_status.csv"
    feature_summary = root / "outputs" / "tables" / "stage6a_wsi_pilot_feature_summary.csv"
    raw_size = 0
    if download_status.exists():
        status = pd.read_csv(download_status)
        raw_size = int(status["local_size"].sum())
    peak_gpu = 0.0
    feature_runtime = 0.0
    if feature_summary.exists():
        features = pd.read_csv(feature_summary)
        peak_gpu = float(pd.to_numeric(features.get("peak_gpu_memory_mb", pd.Series(dtype=float)), errors="coerce").max())
        feature_runtime = float(pd.to_numeric(features.get("feature_pipeline_runtime_seconds", pd.Series(dtype=float)), errors="coerce").max())
    return {
        "downloaded_wsi_gb": raw_size / 1e9,
        "patch_metadata_mb": directory_size(root / "data" / "processed" / "wsi_pilot_patches") / 1e6,
        "feature_file_mb": directory_size(root / "data" / "processed" / "wsi_pilot_features") / 1e6,
        "gpu_feature_runtime_seconds": feature_runtime,
        "peak_gpu_memory_mb": peak_gpu,
    }

