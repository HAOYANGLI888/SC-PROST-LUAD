"""Train low-complexity pathology Cox baselines for Stage 6A diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pathology.diagnostics_utils import (
    clinical_matrix,
    load_patient_feature_frame,
    overfitting_from_performance,
    performance_by_split,
    split_indices,
    train_linear_cox,
    train_transform_features,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train leakage-safe low-complexity pathology baselines on the fixed 100-slide WSI pilot.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--epochs", type=int, default=600)
    return parser


def _fit_model(frame, x, train_idx, *, model_name, epochs, weight_decay=1e-3, l1=0.0):
    risk, runtime = train_linear_cox(
        x,
        frame["os_time_days"].to_numpy(dtype=float),
        frame["os_event"].to_numpy(dtype=int),
        train_idx,
        epochs=epochs,
        weight_decay=weight_decay,
        l1=l1,
    )
    rows = performance_by_split(
        frame,
        risk,
        model_name=model_name,
        interpretation="low-complexity WSI pilot diagnostic only; not publication evidence",
    )
    pred = pd.DataFrame(
        {
            "patient_id": frame["patient_id"],
            "model_name": model_name,
            "risk_score": risk,
            "split": frame["split"],
            "os_time_days": frame["os_time_days"],
            "os_event": frame["os_event"],
            "runtime_seconds": runtime,
        }
    )
    return rows, pred


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    frame, mean_vectors, max_vectors = load_patient_feature_frame(root)
    indices = split_indices(frame)
    train_idx = indices["train"]
    clinical = clinical_matrix(frame, train_idx)
    mean_scaled, _ = train_transform_features(mean_vectors, train_idx)
    max_scaled, _ = train_transform_features(max_vectors, train_idx)
    mean_pca, _ = train_transform_features(mean_vectors, train_idx, n_pca=25)
    rows = []
    preds = []
    specs = [
        ("clinical_only_cox_low_complexity", clinical, 1e-3, 0.0),
        ("mean_pool_pathology_cox", mean_scaled, 1e-3, 0.0),
        ("max_pool_pathology_cox", max_scaled, 1e-3, 0.0),
        ("pca25_pathology_cox", mean_pca, 1e-3, 0.0),
        ("ridge_mean_pool_pathology_cox", mean_scaled, 1e-2, 0.0),
        ("elasticnet_mean_pool_pathology_cox", mean_scaled, 1e-3, 1e-4),
        ("clinical_mean_pool_pathology_cox", np.hstack([clinical, mean_scaled]).astype(np.float32), 1e-3, 0.0),
        ("clinical_pca25_pathology_cox", np.hstack([clinical, mean_pca]).astype(np.float32), 1e-3, 0.0),
    ]
    for model_name, matrix, weight_decay, l1 in specs:
        perf_rows, pred = _fit_model(frame, matrix, train_idx, model_name=model_name, epochs=args.epochs, weight_decay=weight_decay, l1=l1)
        rows.extend(perf_rows)
        preds.append(pred)
    performance = pd.DataFrame(rows)
    diagnostics = overfitting_from_performance(performance)
    predictions = pd.concat(preds, ignore_index=True)
    tables = root / "outputs" / "tables"
    processed = root / "data" / "processed"
    tables.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    performance.to_csv(tables / "stage6a_low_complexity_pathology_performance.csv", index=False)
    diagnostics.to_csv(tables / "stage6a_low_complexity_overfitting_diagnostics.csv", index=False)
    predictions.to_csv(processed / "stage6a_low_complexity_pathology_predictions.csv", index=False)
    print(json.dumps({"status": "passed", "models": len(specs), "patients": int(frame["patient_id"].nunique())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

