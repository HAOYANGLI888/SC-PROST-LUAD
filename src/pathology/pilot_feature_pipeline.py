"""GPU ResNet50 feature extraction for the Stage 6A WSI pilot."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from pathology.patch_feature_extraction import extract_features_from_patch_summary


def extract_wsi_pilot_features(
    root: str | Path = ".",
    *,
    backend: str = "resnet50",
    batch_size: int = 32,
    allow_cpu: bool = False,
) -> pd.DataFrame:
    if backend != "resnet50":
        raise ValueError("Stage 6A-GPU-Pilot only supports strict torchvision pretrained ResNet50.")
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA is not available. Re-run only with --allow-cpu for explicit engineering fallback.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(root).resolve()
    patch_summary = root / "outputs" / "tables" / "stage6a_wsi_pilot_patch_summary.csv"
    patch_root = root / "data" / "processed" / "wsi_pilot_patches"
    feature_root = root / "data" / "processed" / "wsi_pilot_features"
    started = perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    frame = extract_features_from_patch_summary(
        patch_summary,
        patch_root,
        feature_root,
        backend=backend,
        device=device,
        batch_size=batch_size,
        allow_handcrafted_fallback=False,
    )
    elapsed = perf_counter() - started
    frame["used_cuda"] = device == "cuda"
    frame["feature_pipeline_runtime_seconds"] = round(elapsed, 4)
    frame["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 3) if device == "cuda" else 0.0
    tables = root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "stage6a_wsi_pilot_feature_summary.csv", index=False)
    metadata = root / "data" / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metadata / "stage6a_wsi_pilot_feature_summary.csv", index=False)
    return frame

