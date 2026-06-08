"""WSI QC visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_tissue_mask_qc(thumbnail: Image.Image, mask: np.ndarray, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    axes[0].imshow(thumbnail)
    axes[0].set_title("Slide thumbnail")
    axes[1].imshow(thumbnail)
    axes[1].imshow(mask, cmap="magma", alpha=0.45)
    axes[1].set_title("Tissue mask")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_patch_grid_qc(
    thumbnail: Image.Image,
    patch_index,
    output_path: str | Path,
    *,
    slide_dimensions: tuple[int, int],
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = slide_dimensions
    thumb_width, thumb_height = thumbnail.size
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.imshow(thumbnail)
    for row in patch_index.itertuples():
        x = row.x / width * thumb_width
        y = row.y / height * thumb_height
        patch_width = row.patch_size / width * thumb_width
        patch_height = row.patch_size / height * thumb_height
        ax.add_patch(
            plt.Rectangle((x, y), patch_width, patch_height, fill=False, edgecolor="#00A6A6", linewidth=0.5)
        )
    ax.axis("off")
    ax.set_title(f"Selected patch coordinates (n={len(patch_index)})")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)

