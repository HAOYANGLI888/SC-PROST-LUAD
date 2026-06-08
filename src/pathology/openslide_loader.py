"""Robust OpenSlide loading for Windows and real-SVS probes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_DLL_HANDLES: list[Any] = []


def _add_openslide_bin_dll_directory() -> str | None:
    """Register openslide-bin DLLs explicitly when Windows needs help finding them."""

    if os.name != "nt":
        return None
    try:
        import openslide_bin
    except Exception:
        return None
    package_dir = Path(openslide_bin.__file__).resolve().parent
    candidates = [package_dir / "bin", package_dir]
    for candidate in candidates:
        if candidate.is_dir() and hasattr(os, "add_dll_directory"):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(str(candidate)))
                return str(candidate)
            except OSError:
                continue
    return None


def load_openslide():
    """Import openslide with a Windows openslide-bin fallback."""

    try:
        import openslide

        return openslide
    except Exception as first_error:
        _add_openslide_bin_dll_directory()
        try:
            import openslide

            return openslide
        except Exception as second_error:
            raise RuntimeError(
                "OpenSlide import failed. On Windows install both openslide-python and "
                f"openslide-bin in the active environment. Initial error: {first_error}; "
                f"after DLL registration: {second_error}"
            ) from second_error


def open_real_svs(path: str | Path):
    """Open one real SVS file with native OpenSlide."""

    slide_path = Path(path)
    if not slide_path.is_file():
        raise FileNotFoundError(f"SVS file not found: {slide_path}")
    openslide = load_openslide()
    return openslide.OpenSlide(str(slide_path))


def probe_real_svs(path: str | Path) -> dict[str, Any]:
    """Read metadata and a thumbnail from one real SVS file."""

    slide_path = Path(path).resolve()
    slide = open_real_svs(slide_path)
    try:
        thumbnail = slide.get_thumbnail((256, 256))
        return {
            "slide_path": str(slide_path),
            "slide_readable": True,
            "width": int(slide.dimensions[0]),
            "height": int(slide.dimensions[1]),
            "level_count": int(slide.level_count),
            "thumbnail_width": int(thumbnail.size[0]),
            "thumbnail_height": int(thumbnail.size[1]),
        }
    finally:
        slide.close()

