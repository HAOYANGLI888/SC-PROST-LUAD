"""CUDA environment checks for Stage 6A GPU pilot."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def _row(rows: list[dict[str, Any]], component: str, status: str, detail: Any) -> None:
    rows.append({"component": component, "status": status, "detail": str(detail)})


def check_gpu_environment(root: str | Path = ".") -> dict[str, Any]:
    """Verify that PyTorch, torchvision and ResNet50 can actually execute on CUDA."""

    root = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    _row(rows, "python_executable", "ok", sys.executable)
    _row(rows, "python_version", "ok", sys.version.split()[0])
    _row(rows, "platform", "ok", platform.platform())
    _row(rows, "conda_env", "ok" if os.environ.get("CONDA_DEFAULT_ENV") else "warning", os.environ.get("CONDA_DEFAULT_ENV", "not activated"))
    _row(rows, "torch_version", "ok", torch.__version__)
    _row(rows, "torch_cuda_version", "ok" if torch.version.cuda else "failed", torch.version.cuda or "None")
    cuda_available = bool(torch.cuda.is_available())
    _row(rows, "torch_cuda_is_available", "ok" if cuda_available else "failed", cuda_available)
    if not cuda_available:
        blockers.append("torch.cuda.is_available() is False. Stop Stage 6A-GPU-Pilot before downloading new WSI.")
    try:
        import torchvision
        from torchvision.models import ResNet50_Weights, resnet50

        _row(rows, "torchvision_version", "ok", torchvision.__version__)
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        _row(rows, "resnet50_instantiation", "ok", str(weights))
        if cuda_available:
            device = torch.device("cuda")
            tensor = torch.ones((1, 3, 224, 224), device=device)
            _row(rows, "dummy_tensor_to_cuda", "ok", f"device={tensor.device}")
            model = model.to(device).eval()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                output = model(tensor)
            torch.cuda.synchronize()
            _row(rows, "dummy_resnet50_cuda_forward", "ok", f"shape={tuple(output.shape)}")
        del model
    except Exception as exc:
        _row(rows, "torchvision_resnet50_cuda_probe", "failed", f"{type(exc).__name__}: {exc}")
        blockers.append("torchvision pretrained ResNet50 could not run on CUDA.")
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        _row(rows, "gpu_name", "ok", torch.cuda.get_device_name(0))
        _row(rows, "gpu_total_memory_gb", "ok", round(props.total_memory / 1024**3, 3))
        _row(rows, "gpu_memory_allocated_gb", "ok", round(torch.cuda.memory_allocated(0) / 1024**3, 3))
        _row(rows, "gpu_memory_reserved_gb", "ok", round(torch.cuda.memory_reserved(0) / 1024**3, 3))
        _row(rows, "gpu_peak_memory_allocated_gb", "ok", round(torch.cuda.max_memory_allocated(0) / 1024**3, 3))
    status = "passed" if not blockers else "blocked"
    tables = root / "outputs" / "tables"
    reports = root / "outputs" / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    table_path = tables / "stage6a_gpu_environment_check.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    report = (
        "# Stage 6A GPU Environment Check\n\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        f"Overall status: `{status}`\n\n"
        "## Checks\n\n"
        + "\n".join(f"- `{item['component']}`: **{item['status']}** - {item['detail']}" for item in rows)
        + "\n\n## Decision\n\n"
        + ("- CUDA is available and Stage 6A-GPU-Pilot may proceed.\n" if status == "passed" else "- CUDA is not ready. Do not download pilot WSI.\n")
        + "\n## Blockers\n\n"
        + ("\n".join(f"- {item}" for item in blockers) if blockers else "- None.\n")
    )
    report_path = reports / "stage6a_gpu_environment_check_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {"status": status, "blockers": blockers, "table_path": str(table_path), "report_path": str(report_path), "checks": rows}

