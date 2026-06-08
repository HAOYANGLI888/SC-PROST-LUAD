"""Exploratory multivariable Cox adjustment for Stage 2 risk scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm


def _cox_objective(
    coefficients: np.ndarray,
    x: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> tuple[float, np.ndarray]:
    scores = x @ coefficients
    order = np.argsort(-durations)
    ordered_x = x[order]
    ordered_scores = scores[order]
    ordered_events = events[order]
    event_count = max(float(np.sum(ordered_events)), 1.0)
    loss = 0.0
    gradient = np.zeros(x.shape[1], dtype=float)
    for index, event in enumerate(ordered_events):
        if event != 1:
            continue
        risk_scores = ordered_scores[: index + 1]
        risk_x = ordered_x[: index + 1]
        log_denominator = logsumexp(risk_scores)
        weights = np.exp(risk_scores - log_denominator)
        loss -= ordered_scores[index] - log_denominator
        gradient -= ordered_x[index] - np.sum(weights[:, None] * risk_x, axis=0)
    return float(loss / event_count), gradient / event_count


def _cox_hessian(
    coefficients: np.ndarray,
    x: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> np.ndarray:
    scores = x @ coefficients
    order = np.argsort(-durations)
    ordered_x = x[order]
    ordered_scores = scores[order]
    ordered_events = events[order]
    event_count = max(float(np.sum(ordered_events)), 1.0)
    hessian = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for index, event in enumerate(ordered_events):
        if event != 1:
            continue
        risk_scores = ordered_scores[: index + 1]
        risk_x = ordered_x[: index + 1]
        weights = np.exp(risk_scores - logsumexp(risk_scores))
        mean = np.sum(weights[:, None] * risk_x, axis=0)
        centered = risk_x - mean
        hessian += (centered * weights[:, None]).T @ centered
    return hessian / event_count


def multivariable_cox_adjustment(
    predictions: pd.DataFrame,
    *,
    clinical_covariates: tuple[str, ...] = ("age", "male", "stage_numeric"),
) -> pd.DataFrame:
    """Fit risk score + clinical covariates Cox model on held-out predictions."""

    candidates = ["risk_score", *clinical_covariates]
    missing = sorted(set(candidates) - set(predictions.columns))
    if missing:
        raise ValueError(f"Multivariable Cox input is missing columns: {missing}")
    frame = predictions[["os_time_days", "os_event", *candidates]].copy()
    columns = []
    for column in candidates:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if column != "risk_score" and frame[column].dropna().nunique() < 2:
            continue
        frame[column] = frame[column].fillna(frame[column].median())
        columns.append(column)
    frame = frame[["os_time_days", "os_event", *columns]]
    frame = frame.dropna(subset=["os_time_days", "os_event"])
    if frame.empty or frame["os_event"].sum() == 0:
        raise ValueError("Multivariable Cox adjustment requires observed events.")
    x = frame[columns].to_numpy(dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    x = (x - means) / scales
    durations = frame["os_time_days"].to_numpy(dtype=float)
    events = frame["os_event"].to_numpy(dtype=float)
    result = minimize(
        lambda coefficients: _cox_objective(coefficients, x, durations, events),
        x0=np.zeros(x.shape[1]),
        method="BFGS",
        jac=True,
    )
    coefficients = result.x
    try:
        event_count = max(float(np.sum(events)), 1.0)
        covariance = np.linalg.pinv(_cox_hessian(coefficients, x, durations, events)) / event_count
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    except Exception:
        standard_errors = np.full(len(columns), np.nan)
    z_scores = coefficients / standard_errors
    p_values = 2.0 * norm.sf(np.abs(z_scores))
    return pd.DataFrame(
        {
            "covariate": columns,
            "coefficient_per_sd": coefficients,
            "hazard_ratio_per_sd": np.exp(coefficients),
            "standard_error": standard_errors,
            "z_score": z_scores,
            "p_value": p_values,
            "analysis_scope": "representative held-out test split; exploratory",
            "optimizer_success": bool(result.success),
        }
    )


def univariable_cox_risk_score(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> dict[str, float | bool]:
    """Estimate the hazard ratio per SD for one externally predicted risk score."""

    frame = pd.DataFrame(
        {
            "duration": pd.to_numeric(pd.Series(durations), errors="coerce"),
            "event": pd.to_numeric(pd.Series(events), errors="coerce"),
            "risk_score": pd.to_numeric(pd.Series(risk_scores), errors="coerce"),
        }
    ).dropna()
    if frame.empty or frame["event"].sum() == 0:
        raise ValueError("Univariable Cox analysis requires usable rows and events.")
    x = frame[["risk_score"]].to_numpy(dtype=float)
    scale = float(x.std())
    x = (x - x.mean()) / (scale if scale > 0 else 1.0)
    durations_array = frame["duration"].to_numpy(dtype=float)
    events_array = frame["event"].to_numpy(dtype=float)
    result = minimize(
        lambda coefficients: _cox_objective(
            coefficients, x, durations_array, events_array
        ),
        x0=np.zeros(1),
        method="BFGS",
        jac=True,
    )
    coefficient = float(result.x[0])
    event_count = max(float(np.sum(events_array)), 1.0)
    covariance = (
        np.linalg.pinv(_cox_hessian(result.x, x, durations_array, events_array))
        / event_count
    )
    standard_error = float(np.sqrt(np.clip(covariance[0, 0], 1e-12, None)))
    z_score = coefficient / standard_error
    return {
        "coefficient_per_sd": coefficient,
        "hazard_ratio_per_sd": float(np.exp(coefficient)),
        "standard_error": standard_error,
        "z_score": z_score,
        "p_value": float(2.0 * norm.sf(abs(z_score))),
        "optimizer_success": bool(result.success),
    }
