"""Windows-compatible WSI access with explicit limited fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pathology.openslide_loader import load_openslide


class WSIReadError(RuntimeError):
    """Raised when a slide cannot be read with an available backend."""


@dataclass
class PILSlideReader:
    path: Path

    def __post_init__(self) -> None:
        self._image = Image.open(self.path).convert("RGB")
        self.dimensions = self._image.size
        self.level_dimensions = [self.dimensions]
        self.level_count = 1
        self.backend = "pillow_single_level_fallback"

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Image.Image:
        if level != 0:
            raise WSIReadError("Pillow fallback supports level 0 only.")
        x, y = location
        width, height = size
        return self._image.crop((x, y, x + width, y + height)).convert("RGB")

    def thumbnail(self, size: tuple[int, int]) -> Image.Image:
        image = self._image.copy()
        image.thumbnail(size)
        return image

    def close(self) -> None:
        self._image.close()


@dataclass
class OpenSlideReader:
    path: Path

    def __post_init__(self) -> None:
        openslide = load_openslide()
        self._slide = openslide.OpenSlide(str(self.path))
        self.dimensions = self._slide.dimensions
        self.level_dimensions = list(self._slide.level_dimensions)
        self.level_count = self._slide.level_count
        self.backend = "openslide"

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Image.Image:
        return self._slide.read_region(location, level, size).convert("RGB")

    def thumbnail(self, size: tuple[int, int]) -> Image.Image:
        return self._slide.get_thumbnail(size).convert("RGB")

    def close(self) -> None:
        self._slide.close()


def openslide_status() -> dict[str, str | bool]:
    """Return OpenSlide readiness without crashing on missing Windows DLLs."""

    try:
        openslide = load_openslide()
        version = getattr(openslide, "__version__", "unknown")
        return {"available": True, "backend": "openslide", "detail": f"openslide-python {version}"}
    except Exception as exc:
        return {
            "available": False,
            "backend": "pillow_single_level_fallback",
            "detail": f"{type(exc).__name__}: {exc}",
        }


def open_wsi(path: str | Path):
    """Open a WSI or dependency-light synthetic image."""

    slide_path = Path(path)
    if not slide_path.exists():
        raise FileNotFoundError(f"Slide file not found: {slide_path}")
    suffix = slide_path.suffix.casefold()
    try:
        return OpenSlideReader(slide_path)
    except Exception as exc:
        if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            try:
                return PILSlideReader(slide_path)
            except Exception as fallback_exc:
                raise WSIReadError(f"Could not read fallback image {slide_path}: {fallback_exc}") from fallback_exc
        raise WSIReadError(
            f"OpenSlide could not read {slide_path}. On Windows, install OpenSlide DLLs "
            "and openslide-python. Pillow fallback supports synthetic TIFF/PNG/JPEG only. "
            f"Original error: {exc}"
        ) from exc


def create_synthetic_slide(path: str | Path, *, seed: int = 2026, size: int = 1024) -> Path:
    """Create a deterministic H&E-like TIFF for smoke testing."""

    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size), color=(248, 246, 244))
    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(110):
        x = int(rng.integers(30, size - 180))
        y = int(rng.integers(30, size - 180))
        width = int(rng.integers(70, 240))
        height = int(rng.integers(55, 210))
        color = (
            int(rng.integers(130, 220)),
            int(rng.integers(65, 160)),
            int(rng.integers(130, 210)),
            int(rng.integers(110, 210)),
        )
        draw.ellipse((x, y, x + width, y + height), fill=color)
    for _ in range(320):
        x = int(rng.integers(20, size - 20))
        y = int(rng.integers(20, size - 20))
        radius = int(rng.integers(2, 8))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(75, 35, 110, 150))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
