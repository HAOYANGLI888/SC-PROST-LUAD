"""Dependency-light survival metrics for Stage 2."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

from models.survival_losses import concordance_index


def time_dependent_auc(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    horizon_days: float,
) -> float:
    """Approximate cumulative/dynamic AUC excluding indeterminate censoring."""

    times = np.asarray(durations, dtype=float)
    status = np.asarray(events, dtype=int)
    scores = np.asarray(risk_scores, dtype=float)
    cases = (status == 1) & (times <= horizon_days)
    controls = times > horizon_days
    usable = cases | controls
    labels = cases[usable].astype(int)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores[usable]))


def logrank_p_value(
    durations: np.ndarray,
    events: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Two-group log-rank P value without an external survival package."""

    times = np.asarray(durations, dtype=float)
    status = np.asarray(events, dtype=int)
    high = np.asarray(groups, dtype=bool)
    unique_event_times = np.unique(times[status == 1])
    observed_high = 0.0
    expected_high = 0.0
    variance = 0.0
    for time in unique_event_times:
        at_risk = times >= time
        events_at_time = (times == time) & (status == 1)
        total_risk = int(np.sum(at_risk))
        total_events = int(np.sum(events_at_time))
        high_risk = int(np.sum(at_risk & high))
        high_events = int(np.sum(events_at_time & high))
        if total_risk <= 1:
            continue
        observed_high += high_events
        expected_high += total_events * high_risk / total_risk
        variance += (
            total_events
            * (high_risk / total_risk)
            * (1.0 - high_risk / total_risk)
            * ((total_risk - total_events) / (total_risk - 1.0))
        )
    if variance <= 0:
        return float("nan")
    statistic = (observed_high - expected_high) ** 2 / variance
    return float(chi2.sf(statistic, df=1))


def breslow_baseline_cumulative_hazard(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate baseline cumulative hazard for risk-to-probability conversion."""

    times = np.asarray(durations, dtype=float)
    status = np.asarray(events, dtype=int)
    scores = np.asarray(risk_scores, dtype=float)
    centered = scores - np.mean(scores)
    unique_times = np.sort(np.unique(times[status == 1]))
    cumulative = []
    running = 0.0
    for time in unique_times:
        deaths = np.sum((times == time) & (status == 1))
        risk_sum = np.sum(np.exp(np.clip(centered[times >= time], -20.0, 20.0)))
        if risk_sum > 0:
            running += deaths / risk_sum
        cumulative.append(running)
    return unique_times, np.asarray(cumulative)


def predict_event_probability(
    train_durations: np.ndarray,
    train_events: np.ndarray,
    train_risk_scores: np.ndarray,
    risk_scores: np.ndarray,
    horizon_days: float,
) -> np.ndarray:
    """Convert risk scores to event probabilities at one horizon."""

    event_times, cumulative = breslow_baseline_cumulative_hazard(
        train_durations,
        train_events,
        train_risk_scores,
    )
    if cumulative.size == 0:
        return np.full(len(risk_scores), np.nan)
    position = np.searchsorted(event_times, horizon_days, side="right") - 1
    baseline = cumulative[position] if position >= 0 else 0.0
    centered = np.asarray(risk_scores, dtype=float) - np.mean(train_risk_scores)
    survival = np.exp(-baseline * np.exp(np.clip(centered, -20.0, 20.0)))
    return 1.0 - survival


def calibration_points(
    durations: np.ndarray,
    events: np.ndarray,
    probabilities: np.ndarray,
    horizon_days: float,
    *,
    bins: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Create simple calibration points while excluding early censored samples."""

    times = np.asarray(durations, dtype=float)
    status = np.asarray(events, dtype=int)
    predicted = np.asarray(probabilities, dtype=float)
    usable = ((status == 1) & (times <= horizon_days)) | (times > horizon_days)
    labels = ((status == 1) & (times <= horizon_days)).astype(float)
    table = np.column_stack([predicted[usable], labels[usable]])
    if len(table) < bins:
        return np.array([]), np.array([])
    order = np.argsort(table[:, 0])
    groups = np.array_split(table[order], bins)
    return (
        np.asarray([np.mean(group[:, 0]) for group in groups]),
        np.asarray([np.mean(group[:, 1]) for group in groups]),
    )


__all__ = [
    "calibration_points",
    "concordance_index",
    "logrank_p_value",
    "predict_event_probability",
    "time_dependent_auc",
]
