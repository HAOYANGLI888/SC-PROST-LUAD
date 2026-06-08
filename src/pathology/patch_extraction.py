"""Patch-coordinate extraction without saving every image patch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pathology.tissue_mask import tissue_fraction, tissue_mask_from_thumbnail
from pathology.wsi_io import open_wsi
from pathology.wsi_qc import save_patch_grid_qc, save_tissue_mask_qc


def extract_slide_patch_index(
    slide_path: str | Path,
    output_dir: str | Path,
    *,
    patch_size: int = 256,
    tissue_threshold: float = 0.35,
    max_patches: int = 1000,
) -> dict[str, object]:
    """Select tissue-rich level-0 coordinates and save QC metadata."""

    slide_path = Path(slide_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reader = open_wsi(slide_path)
    try:
        width, height = reader.dimensions
        thumbnail = reader.thumbnail((1024, 1024))
        mask = tissue_mask_from_thumbnail(thumbnail)
        thumb_width, thumb_height = thumbnail.size
        rows = []
        for y in range(0, max(height - patch_size + 1, 1), patch_size):
            for x in range(0, max(width - patch_size + 1, 1), patch_size):
                box = (
                    int(x / width * thumb_width),
                    int(y / height * thumb_height),
                    max(int((x + patch_size) / width * thumb_width), 1),
                    max(int((y + patch_size) / height * thumb_height), 1),
                )
                fraction = tissue_fraction(mask, box)
                if fraction >= tissue_threshold:
                    rows.append({"x": x, "y": y, "level": 0, "patch_size": patch_size, "tissue_fraction": fraction})
        index = pd.DataFrame(rows).sort_values(["tissue_fraction", "y", "x"], ascending=[False, True, True]).head(max_patches)
        if index.empty:
            raise RuntimeError(f"No tissue-rich patches found for {slide_path}")
        index = index.reset_index(drop=True)
        index.insert(0, "patch_id", [f"patch_{value:05d}" for value in range(len(index))])
        index.to_csv(output / "patch_index.csv", index=False)
        metadata = pd.DataFrame(
            [
                {
                    "slide_id": slide_path.stem,
                    "slide_path": str(slide_path),
                    "reader_backend": reader.backend,
                    "width": width,
                    "height": height,
                    "level": 0,
                    "patch_size": patch_size,
                    "selected_patch_count": len(index),
                    "tissue_threshold": tissue_threshold,
                }
            ]
        )
        metadata.to_csv(output / "metadata.csv", index=False)
        save_tissue_mask_qc(thumbnail, mask, output / "qc_tissue_mask.png")
        save_patch_grid_qc(thumbnail, index, output / "qc_patch_grid.png", slide_dimensions=reader.dimensions)
        return metadata.iloc[0].to_dict()
    finally:
        reader.close()


def extract_patch_indexes_from_status(
    status_path: str | Path,
    output_root: str | Path,
    *,
    patch_size: int = 256,
    tissue_threshold: float = 0.35,
    max_patches: int = 1000,
) -> pd.DataFrame:
    status = pd.read_csv(status_path)
    rows = []
    for record in status.to_dict("records"):
        local_path = Path(str(record.get("local_path", "")))
        if not local_path.exists():
            rows.append({**record, "patch_status": "skipped_missing_slide"})
            continue
        expected_size = record.get("expected_size", record.get("file_size"))
        if pd.notna(expected_size) and local_path.stat().st_size != int(expected_size):
            rows.append(
                {
                    **record,
                    "patch_status": "skipped_incomplete_slide",
                    "patch_error": f"size mismatch: expected {int(expected_size)}, observed {local_path.stat().st_size}",
                }
            )
            continue
        slide_id = str(record["file_id"])
        try:
            result = extract_slide_patch_index(
                local_path,
                Path(output_root) / slide_id,
                patch_size=patch_size,
                tissue_threshold=tissue_threshold,
                max_patches=max_patches,
            )
            rows.append({**record, **result, "patch_status": "extracted"})
        except (OSError, RuntimeError, ValueError) as exc:
            rows.append(
                {
                    **record,
                    "patch_status": "failed_reader_or_extraction",
                    "patch_error": str(exc),
                }
            )
    return pd.DataFrame(rows)
