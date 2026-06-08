"""Stage 6A-Fix environment audit for Windows WSI processing."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from pathology.openslide_loader import load_openslide, probe_real_svs


def _record(rows: list[dict[str, str]], component: str, status: str, detail: Any) -> None:
    rows.append({"component": component, "status": status, "detail": str(detail)})


def _first_real_svs(root: Path) -> Path | None:
    candidates = sorted((root / "data" / "raw" / "tcga_luad" / "wsi" / "smallset").rglob("*.svs"))
    return candidates[0] if candidates else None


def _torchvision_cached_checkpoint(weights) -> Path:
    import torch

    checkpoint_name = Path(urlparse(weights.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name


def check_wsi_environment(root: str | Path = ".", *, download_pretrained_weights: bool = False) -> dict[str, Any]:
    """Audit the active interpreter, native WSI reader and pretrained baseline."""

    root = Path(root).resolve()
    rows: list[dict[str, str]] = []
    _record(rows, "python_executable", "ok", sys.executable)
    _record(rows, "python_version", "ok", sys.version.split()[0])
    _record(rows, "platform", "ok", platform.platform())
    _record(rows, "conda_default_env", "ok" if os.environ.get("CONDA_DEFAULT_ENV") else "warning", os.environ.get("CONDA_DEFAULT_ENV", "not activated"))
    blockers: list[str] = []

    try:
        import openslide_bin

        _record(rows, "openslide_bin", "ok", getattr(openslide_bin, "__version__", "installed"))
    except Exception as exc:
        _record(rows, "openslide_bin", "failed", f"{type(exc).__name__}: {exc}")
        blockers.append("Install openslide-bin in gpu_py310.")

    try:
        openslide = load_openslide()
        _record(rows, "openslide_python_native", "ok", getattr(openslide, "__version__", "unknown"))
    except Exception as exc:
        _record(rows, "openslide_python_native", "failed", f"{type(exc).__name__}: {exc}")
        blockers.append("Install openslide-python and openslide-bin in gpu_py310.")

    try:
        import torch

        _record(rows, "torch", "ok", torch.__version__)
        _record(rows, "torch_cuda_available", "ok" if torch.cuda.is_available() else "warning", torch.cuda.is_available())
        _record(rows, "torch_cuda_version", "ok" if torch.version.cuda else "warning", torch.version.cuda or "CPU-only torch build")
    except Exception as exc:
        _record(rows, "torch", "failed", f"{type(exc).__name__}: {exc}")
        blockers.append("Install PyTorch in gpu_py310.")

    try:
        import torchvision
        from torchvision.models import ResNet50_Weights, resnet50

        _record(rows, "torchvision", "ok", torchvision.__version__)
        weights = ResNet50_Weights.DEFAULT
        cached_checkpoint = _torchvision_cached_checkpoint(weights)
        if not cached_checkpoint.is_file() and not download_pretrained_weights:
            raise RuntimeError(
                f"Pretrained weights are not cached at {cached_checkpoint}. "
                "Re-run with --download-pretrained-weights to fetch them explicitly."
            )
        model = resnet50(weights=weights)
        del model
        _record(rows, "torchvision_pretrained_resnet50", "ok", f"{weights} | cache={cached_checkpoint}")
    except Exception as exc:
        _record(rows, "torchvision_pretrained_resnet50", "failed", f"{type(exc).__name__}: {exc}")
        blockers.append("Install a torch-compatible torchvision build and ensure pretrained ResNet50 weights can be downloaded.")

    real_slide = _first_real_svs(root)
    if real_slide is None:
        _record(rows, "real_svs_probe", "warning", "No complete local SVS found for probe.")
    else:
        try:
            probe = probe_real_svs(real_slide)
            _record(rows, "real_svs_probe", "ok", f"{probe['slide_path']} | {probe['width']}x{probe['height']} | levels={probe['level_count']}")
        except Exception as exc:
            _record(rows, "real_svs_probe", "failed", f"{type(exc).__name__}: {exc}")
            blockers.append("Repair native OpenSlide loading before real patch extraction.")

    table = pd.DataFrame(rows)
    tables = root / "outputs" / "tables"
    docs = root / "docs"
    tables.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    table_path = tables / "stage6a_wsi_environment_check.csv"
    table.to_csv(table_path, index=False)
    status = "passed" if not blockers else "blocked"
    report = (
        "# Stage 6A-Fix WSI Environment Report\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"Overall status: `{status}`\n\n"
        "## Active Interpreter\n\n"
        f"- Executable: `{sys.executable}`\n"
        f"- Conda environment: `{os.environ.get('CONDA_DEFAULT_ENV', 'not activated')}`\n\n"
        "## Checks\n\n"
        + "\n".join(f"- `{row['component']}`: **{row['status']}** - {row['detail']}" for row in rows)
        + "\n\n## Windows Commands\n\n"
        "```powershell\n"
        "conda activate gpu_py310\n"
        "python -m pip install openslide-python openslide-bin torchvision\n"
        "python scripts/stage6a_check_wsi_environment.py --config configs/base.yaml\n"
        "python scripts/stage6a_check_wsi_environment.py --config configs/base.yaml --download-pretrained-weights\n"
        "```\n\n"
        "## Blockers\n\n"
        + ("\n".join(f"- {item}" for item in blockers) if blockers else "- None. Real-SVS smallset processing can proceed.")
        + "\n"
    )
    report_path = docs / "stage6a_environment_fix_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "status": status,
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", "not activated"),
        "checks": rows,
        "blockers": blockers,
        "table_path": str(table_path),
        "report_path": str(report_path),
    }
