"""Censoring-aware Brier score, calibration, and decision-curve helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


class IPCWMetricError(RuntimeError):
    """Raised when censoring weights cannot be estimated stably."""


def _km_censoring_curve(
    durations: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(durations, dtype=float)
    status = np.asarray(events, dtype=int)
    censoring = 1 - status
    if len(times) == 0:
        raise IPCWMetricError("Cannot estimate censoring distribution from empty data.")
    unique = np.sort(np.unique(times))
    survival: list[float] = []
    running = 1.0
    for time in unique:
        at_risk = np.sum(times >= time)
        censored = np.sum((times == time) & (censoring == 1))
        if at_risk > 0:
            running *= 1.0 - censored / at_risk
        survival.append(running)
    return unique, np.asarray(survival, dtype=float)


def _step_value(
    curve_times: np.ndarray,
    curve_values: np.ndarray,
    query: np.ndarray | float,
    *,
    before: bool = False,
) -> np.ndarray:
    query_values = np.asarray(query, dtype=float)
    side = "left" if before else "right"
    indexes = np.searchsorted(curve_times, query_values, side=side) - 1
    values = np.ones(query_values.shape, dtype=float)
    valid = indexes >= 0
    values[valid] = curve_values[indexes[valid]]
    return values


def ipcw_brier_score(
    durations: np.ndarray,
    events: np.ndarray,
    event_probabilities: np.ndarray,
    horizon_days: float,
    *,
    minimum_censoring_survival: float = 0.05,
) -> float:
    """Estimate IPCW Brier score at one horizon."""

    times = np.asarray(durations, dtype=float)
    status = np.asarray(events, dtype=int)
    probability = np.asarray(event_probabilities, dtype=float)
    if not (len(times) == len(status) == len(probability)):
        raise ValueError("IPCW inputs must have matching lengths.")
    curve_times, curve_values = _km_censoring_curve(times, status)
    g_horizon = float(_step_value(curve_times, curve_values, horizon_days))
    g_event = _step_value(curve_times, curve_values, times, before=True)
    cases = (status == 1) & (times <= horizon_days)
    controls = times > horizon_days
    if g_horizon < minimum_censoring_survival:
        raise IPCWMetricError(
            f"Censoring survival G(t)={g_horizon:.4f} below stability threshold "
            f"at horizon {horizon_days:g} days."
        )
    if np.any(g_event[cases] < minimum_censoring_survival):
        raise IPCWMetricError("Event IPCW weight is unstable due to near-zero G(T-).")
    weights = np.zeros(len(times), dtype=float)
    weights[cases] = 1.0 / g_event[cases]
    weights[controls] = 1.0 / g_horizon
    outcome = cases.astype(float)
    return float(np.mean(weights * (outcome - probability) ** 2))


def integrated_brier_score(
    durations: np.ndarray,
    events: np.ndarray,
    probability_matrix: np.ndarray,
    horizons_days: np.ndarray,
) -> float:
    """Numerically integrate IPCW Brier scores across requested horizons."""

    horizons = np.asarray(horizons_days, dtype=float)
    matrix = np.asarray(probability_matrix, dtype=float)
    if matrix.shape != (len(durations), len(horizons)):
        raise ValueError("Probability matrix must be samples x horizons.")
    scores = np.asarray(
        [
            ipcw_brier_score(durations, events, matrix[:, index], horizon)
            for index, horizon in enumerate(horizons)
        ]
    )
    if len(horizons) == 1:
        return float(scores[0])
    return float(np.trapz(scores, horizons) / (horizons[-1] - horizons[0]))


def _km_event_probability_at_horizon(
    durations: np.ndarray,
    events: np.ndarray,
    horizon_days: float,
) -> float:
    times = np.asarray(durations, dtype=float)
    status = np.asarray(events, dtype=int)
    running = 1.0
    for time in np.sort(np.unique(times[(times <= horizon_days) & (status == 1)])):
        at_risk = np.sum(times >= time)
        deaths = np.sum((times == time) & (status == 1))
        if at_risk:
            running *= 1.0 - deaths / at_risk
    return float(1.0 - running)


def censoring_aware_calibration(
    durations: np.ndarray,
    events: np.ndarray,
    event_probabilities: np.ndarray,
    horizon_days: float,
    *,
    bins: int = 5,
) -> pd.DataFrame:
    """Return quantile calibration points using KM-observed event probabilities."""

    frame = pd.DataFrame(
        {
            "time": np.asarray(durations, dtype=float),
            "event": np.asarray(events, dtype=int),
            "predicted": np.asarray(event_probabilities, dtype=float),
        }
    ).dropna()
    if len(frame) < bins:
        raise IPCWMetricError("Too few rows for censoring-aware calibration.")
    frame = frame.sort_values("predicted")
    rows: list[dict[str, float | int]] = []
    for index, positions in enumerate(np.array_split(np.arange(len(frame)), bins), start=1):
        group = frame.iloc[positions]
        rows.append(
            {
                "bin": index,
                "n": len(group),
                "predicted_event_probability": float(group["predicted"].mean()),
                "km_observed_event_probability": _km_event_probability_at_horizon(
                    group["time"].to_numpy(),
                    group["event"].to_numpy(),
                    horizon_days,
                ),
                "horizon_days": float(horizon_days),
            }
        )
    return pd.DataFrame(rows)


def ipcw_decision_curve(
    durations: np.ndarray,
    events: np.ndarray,
    event_probabilities: np.ndarray,
    horizon_days: float,
    *,
    thresholds: np.ndarray | None = None,
    minimum_censoring_survival: float = 0.05,
) -> pd.DataFrame:
    """Estimate net benefit with IPCW-weighted case and control definitions."""

    times = np.asarray(durations, dtype=float)
    status = np.asarray(events, dtype=int)
    probability = np.asarray(event_probabilities, dtype=float)
    thresholds = (
        np.asarray(thresholds, dtype=float)
        if thresholds is not None
        else np.linspace(0.05, 0.60, 12)
    )
    curve_times, curve_values = _km_censoring_curve(times, status)
    g_horizon = float(_step_value(curve_times, curve_values, horizon_days))
    g_event = _step_value(curve_times, curve_values, times, before=True)
    cases = (status == 1) & (times <= horizon_days)
    controls = times > horizon_days
    if g_horizon < minimum_censoring_survival or np.any(
        g_event[cases] < minimum_censoring_survival
    ):
        raise IPCWMetricError("Decision curve IPCW weights are unstable.")
    case_weights = np.zeros(len(times), dtype=float)
    control_weights = np.zeros(len(times), dtype=float)
    case_weights[cases] = 1.0 / g_event[cases]
    control_weights[controls] = 1.0 / g_horizon
    prevalence = float(case_weights.sum() / len(times))
    rows = []
    for threshold in thresholds:
        predicted_positive = probability >= threshold
        tp = float(case_weights[predicted_positive].sum() / len(times))
        fp = float(control_weights[predicted_positive].sum() / len(times))
        penalty = threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "net_benefit_model": tp - fp * penalty,
                "net_benefit_treat_all": prevalence - (1.0 - prevalence) * penalty,
                "net_benefit_treat_none": 0.0,
                "horizon_days": float(horizon_days),
            }
        )
    return pd.DataFrame(rows)
