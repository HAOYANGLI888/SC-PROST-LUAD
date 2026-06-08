"""Patch coordinate extraction for the Stage 6A WSI pilot."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from pathology.patch_extraction import extract_patch_indexes_from_status


def extract_wsi_pilot_patches(
    root: str | Path = ".",
    *,
    patch_size: int = 256,
    tissue_threshold: float = 0.35,
    max_patches_per_slide: int = 512,
) -> pd.DataFrame:
    root = Path(root).resolve()
    status_path = root / "data" / "metadata" / "stage6a_wsi_pilot_download_status.csv"
    output_root = root / "data" / "processed" / "wsi_pilot_patches"
    frame = extract_patch_indexes_from_status(
        status_path,
        output_root,
        patch_size=patch_size,
        tissue_threshold=tissue_threshold,
        max_patches=max_patches_per_slide,
    )
    for index, row in frame.iterrows():
        if row.get("patch_status") == "extracted":
            patch_index = output_root / str(row["file_id"]) / "patch_index.csv"
            patches = pd.read_csv(patch_index)
            frame.loc[index, "mean_tissue_fraction"] = float(patches["tissue_fraction"].mean())
            frame.loc[index, "min_tissue_fraction"] = float(patches["tissue_fraction"].min())
    tables = root / "outputs" / "tables"
    figures = root / "outputs" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "stage6a_wsi_pilot_patch_summary.csv", index=False)
    extracted = frame.loc[frame["patch_status"] == "extracted"]
    if not extracted.empty:
        slide_id = str(extracted.iloc[0]["file_id"])
        shutil.copyfile(output_root / slide_id / "qc_tissue_mask.png", figures / "stage6a_wsi_pilot_tissue_mask_example.png")
        shutil.copyfile(output_root / slide_id / "qc_patch_grid.png", figures / "stage6a_wsi_pilot_patch_grid_example.png")
    return frame

