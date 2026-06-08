"""Metrics for Stage 6A WSI pilot models."""

from __future__ import annotations

import pandas as pd

from evaluation.pathology_metrics import pathology_survival_metrics


def summarize_pilot_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, split), frame in predictions.groupby(["model_name", "split"], sort=True):
        rows.append(
            {
                "model_name": model_name,
                "split": split,
                **pathology_survival_metrics(frame["os_time_days"], frame["os_event"], frame["risk_score"]),
                "patient_count": frame["patient_id"].nunique(),
                "events": int(frame.drop_duplicates("patient_id")["os_event"].sum()),
                "interpretation": "pilot feasibility only; not a scientific performance estimate",
            }
        )
    return pd.DataFrame(rows)


def overfitting_diagnostics(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, frame in performance.groupby("model_name"):
        values = {row["split"]: row for row in frame.to_dict("records")}
        train_c = float(values.get("train", {}).get("c_index", float("nan")))
        test_c = float(values.get("test", {}).get("c_index", float("nan")))
        val_c = float(values.get("validation", {}).get("c_index", float("nan")))
        rows.append(
            {
                "model_name": model_name,
                "train_c_index": train_c,
                "validation_c_index": val_c,
                "test_c_index": test_c,
                "train_test_gap": train_c - test_c,
                "overfitting_flag": bool(train_c - test_c > 0.15),
            }
        )
    return pd.DataFrame(rows)

