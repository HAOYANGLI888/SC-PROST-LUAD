"""Thumbnail tissue masking for WSI patch selection."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def tissue_mask_from_thumbnail(
    thumbnail: Image.Image,
    *,
    lightness_threshold: int = 238,
    saturation_threshold: int = 12,
) -> np.ndarray:
    """Identify non-white tissue with lightness and saturation thresholds."""

    rgb = np.asarray(thumbnail.convert("RGB"), dtype=np.float32)
    lightness = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    mask = (lightness < lightness_threshold) & (saturation > saturation_threshold)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    return mask.astype(bool)


def tissue_fraction(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Compute tissue fraction inside thumbnail coordinates."""

    x0, y0, x1, y1 = box
    region = mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return float(region.mean()) if region.size else 0.0

