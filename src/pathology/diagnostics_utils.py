"""Shared utilities for Stage 6A WSI diagnostics."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from evaluation.survival_metrics import concordance_index
from models.survival_losses import cox_ph_loss


def repo_path(root: str | Path = ".") -> Path:
    return Path(root).resolve()


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_pilot_tables(root: str | Path = ".") -> dict[str, pd.DataFrame]:
    root = repo_path(root)
    return {
        "cohort": read_csv(root / "data" / "metadata" / "stage6a_wsi_pilot_cohort.csv"),
        "download": read_csv(root / "data" / "metadata" / "stage6a_wsi_pilot_download_status.csv"),
        "split": read_csv(root / "data" / "metadata" / "stage6a_wsi_pilot_split_seed42.csv"),
        "patch": read_csv(root / "outputs" / "tables" / "stage6a_wsi_pilot_patch_summary.csv"),
        "feature": read_csv(root / "outputs" / "tables" / "stage6a_wsi_pilot_feature_summary.csv"),
    }


def load_feature_artifact(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    features = artifact.get("features")
    if not torch.is_tensor(features):
        raise ValueError(f"Feature artifact has no tensor `features`: {path}")
    return artifact


def load_patient_feature_frame(root: str | Path = ".") -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return patient metadata with mean and max pooled 2048-d ResNet50 embeddings."""

    tables = load_pilot_tables(root)
    features = tables["feature"].loc[
        tables["feature"]["feature_status"].isin(["extracted", "skipped_existing"])
    ].copy()
    split = tables["split"][["patient_id", "split"]].drop_duplicates("patient_id")
    rows = []
    mean_vectors = []
    max_vectors = []
    for row in features.sort_values("patient_id").to_dict("records"):
        artifact = load_feature_artifact(row["feature_path"])
        tensor = artifact["features"].float()
        rows.append(row)
        mean_vectors.append(tensor.mean(dim=0).numpy())
        max_vectors.append(tensor.max(dim=0).values.numpy())
    frame = pd.DataFrame(rows).merge(split, on="patient_id", how="left", validate="one_to_one")
    return frame, np.vstack(mean_vectors), np.vstack(max_vectors)


def split_indices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: np.where(frame["split"].to_numpy() == name)[0] for name in ("train", "validation", "test")}


def clinical_matrix(frame: pd.DataFrame, train_idx: np.ndarray) -> np.ndarray:
    values = frame[["age", "male", "stage_numeric"]].apply(pd.to_numeric, errors="coerce")
    train = values.iloc[train_idx]
    med = train.median().fillna(0.0)
    values = values.fillna(med).fillna(0.0).to_numpy(dtype=np.float32)
    mean = values[train_idx].mean(axis=0)
    std = values[train_idx].std(axis=0)
    std[std == 0] = 1.0
    return ((values - mean) / std).astype(np.float32)


def train_transform_features(
    matrix: np.ndarray,
    train_idx: np.ndarray,
    *,
    n_pca: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix[train_idx])
    all_scaled = scaler.transform(matrix)
    transformer: dict[str, Any] = {"scaler": scaler}
    if n_pca is not None:
        n_components = min(n_pca, scaled.shape[0] - 1, scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(scaled)
        all_scaled = pca.transform(all_scaled)
        transformer["pca"] = pca
    return all_scaled.astype(np.float32), transformer


def train_linear_cox(
    x: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    train_idx: np.ndarray,
    *,
    epochs: int = 500,
    lr: float = 0.01,
    weight_decay: float = 1e-3,
    l1: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    times_tensor = torch.as_tensor(times.astype(np.float32))
    events_tensor = torch.as_tensor(events.astype(np.float32))
    model = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    started = perf_counter()
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        scores = model(x_tensor[train_tensor]).reshape(-1)
        loss = cox_ph_loss(scores, times_tensor[train_tensor], events_tensor[train_tensor])
        if l1:
            loss = loss + l1 * model.weight.abs().sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        risk = model(x_tensor).reshape(-1).numpy()
    return risk, perf_counter() - started


def performance_by_split(
    frame: pd.DataFrame,
    risk: np.ndarray,
    *,
    model_name: str,
    interpretation: str,
) -> list[dict[str, Any]]:
    rows = []
    times = frame["os_time_days"].to_numpy(dtype=float)
    events = frame["os_event"].to_numpy(dtype=int)
    for split_name, idx in split_indices(frame).items():
        rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "c_index": concordance_index(times[idx], events[idx], risk[idx]),
                "patient_count": int(len(idx)),
                "events": int(events[idx].sum()),
                "interpretation": interpretation,
            }
        )
    return rows


def overfitting_from_performance(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in performance.groupby("model_name"):
        values = {row["split"]: row for row in group.to_dict("records")}
        train = float(values.get("train", {}).get("c_index", np.nan))
        val = float(values.get("validation", {}).get("c_index", np.nan))
        test = float(values.get("test", {}).get("c_index", np.nan))
        rows.append(
            {
                "model_name": model_name,
                "train_c_index": train,
                "validation_c_index": val,
                "test_c_index": test,
                "train_test_gap": train - test,
                "overfitting_flag": bool(train - test > 0.15),
            }
        )
    return pd.DataFrame(rows)

