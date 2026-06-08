"""Patch embedding extraction with optional torchvision and explicit fallback."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pathology.wsi_io import open_wsi


EXTERNAL_FOUNDATION_MODEL_BACKENDS = ("UNI", "CONCH", "HIPT", "CTransPath")
_RESNET_CACHE: dict[str, tuple[torch.nn.Module, object]] = {}


def _handcrafted_embedding(patch) -> np.ndarray:
    rgb = np.asarray(patch.resize((64, 64)), dtype=np.float32) / 255.0
    features = []
    for channel in range(3):
        values = rgb[:, :, channel]
        features.extend([values.mean(), values.std(), *np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])])
    gray = rgb.mean(axis=2)
    features.extend([gray.mean(), gray.std(), *np.histogram(gray, bins=9, range=(0, 1), density=True)[0]])
    return np.asarray(features, dtype=np.float32)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")


def _torchvision_resnet50(device: torch.device):
    cache_key = str(device)
    if cache_key in _RESNET_CACHE:
        return _RESNET_CACHE[cache_key]
    try:
        from torchvision.models import ResNet50_Weights, resnet50
    except Exception as exc:
        raise RuntimeError(
            "torchvision is unavailable. Install torchvision in gpu_py310 or use "
            "--backend handcrafted for the labeled limited fallback."
        ) from exc
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    encoder = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    result = (encoder, weights.transforms())
    _RESNET_CACHE[cache_key] = result
    return result


def _resnet_embeddings(patches: list[Image.Image], *, device: torch.device, batch_size: int) -> np.ndarray:
    encoder, transform = _torchvision_resnet50(device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.stack([transform(patch) for patch in patches[start:start + batch_size]]).to(device)
            outputs.append(encoder(batch).flatten(1).cpu())
    return torch.cat(outputs).numpy()


def extract_slide_features(
    slide_path: str | Path,
    patch_index_path: str | Path,
    output_path: str | Path,
    *,
    skip_existing: bool = True,
    backend: str = "resnet50",
    device: str = "auto",
    batch_size: int = 32,
    allow_handcrafted_fallback: bool = False,
) -> dict[str, object]:
    """Read selected patches and save one embedding tensor per slide."""

    started = perf_counter()
    output = Path(output_path)
    if skip_existing and output.exists():
        artifact = torch.load(output, map_location="cpu", weights_only=False)
        expected_backend = "torchvision_pretrained_resnet50" if backend == "resnet50" else "deterministic_color_texture_fallback_no_torchvision"
        if artifact.get("backend") == expected_backend:
            return {
                "feature_status": "skipped_existing",
                "feature_backend": artifact["backend"],
                "feature_dim": int(artifact["features"].shape[1]),
                "feature_shape": "x".join(str(value) for value in artifact["features"].shape),
                "patch_count": int(artifact["features"].shape[0]),
                "device": str(artifact.get("device", "unknown")),
                "elapsed_seconds": round(perf_counter() - started, 4),
            }
    index = pd.read_csv(patch_index_path)
    reader = open_wsi(slide_path)
    try:
        patches = []
        for row in index.itertuples():
            patch = reader.read_region((int(row.x), int(row.y)), int(row.level), (int(row.patch_size), int(row.patch_size)))
            patches.append(patch)
        resolved_backend = backend
        resolved_device = _resolve_device(device)
        if backend == "handcrafted":
            features = np.vstack([_handcrafted_embedding(patch) for patch in patches])
            resolved_backend = "deterministic_color_texture_fallback_no_torchvision"
        elif backend == "resnet50":
            try:
                features = _resnet_embeddings(
                    patches, device=resolved_device, batch_size=batch_size
                )
                resolved_backend = "torchvision_pretrained_resnet50"
            except RuntimeError:
                if not allow_handcrafted_fallback:
                    raise
                features = np.vstack([_handcrafted_embedding(patch) for patch in patches])
                resolved_backend = "deterministic_color_texture_fallback_no_torchvision"
        else:
            raise ValueError(
                "backend must be resnet50 or handcrafted. External foundation-model "
                f"adapters are reserved but not bundled: {EXTERNAL_FOUNDATION_MODEL_BACKENDS}"
            )
        tensor = torch.as_tensor(features, dtype=torch.float32)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": tensor,
                "coordinates": torch.as_tensor(index[["x", "y"]].to_numpy(), dtype=torch.int64),
                "patch_ids": index["patch_id"].tolist(),
                "backend": resolved_backend,
                "device": str(resolved_device),
                "note": (
                    "Smoke-test fallback only; formal extraction should use torchvision pretrained ResNet50."
                    if resolved_backend.startswith("deterministic_")
                    else "Pretrained torchvision ResNet50 baseline."
                ),
            },
            output,
        )
        return {
            "feature_status": "extracted",
            "feature_backend": resolved_backend,
            "feature_dim": int(tensor.shape[1]),
            "feature_shape": "x".join(str(value) for value in tensor.shape),
            "patch_count": int(tensor.shape[0]),
            "device": str(resolved_device),
            "elapsed_seconds": round(perf_counter() - started, 4),
        }
    finally:
        reader.close()


def extract_features_from_patch_summary(
    patch_summary_path: str | Path,
    patch_root: str | Path,
    feature_root: str | Path,
    *,
    backend: str = "resnet50",
    device: str = "auto",
    batch_size: int = 32,
    allow_handcrafted_fallback: bool = False,
) -> pd.DataFrame:
    summary = pd.read_csv(patch_summary_path)
    rows = []
    for row in summary.to_dict("records"):
        if row.get("patch_status") != "extracted":
            rows.append({**row, "feature_status": "skipped_no_patches"})
            continue
        slide_id = str(row["file_id"])
        result = extract_slide_features(
            row["local_path"],
            Path(patch_root) / slide_id / "patch_index.csv",
            Path(feature_root) / f"{slide_id}.pt",
            backend=backend,
            device=device,
            batch_size=batch_size,
            allow_handcrafted_fallback=allow_handcrafted_fallback,
        )
        rows.append({**row, **result, "feature_path": str(Path(feature_root) / f"{slide_id}.pt")})
    return pd.DataFrame(rows)
