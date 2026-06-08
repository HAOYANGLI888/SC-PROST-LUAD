import numpy as np

from evaluation.ipcw_metrics import (
    censoring_aware_calibration,
    integrated_brier_score,
    ipcw_brier_score,
    ipcw_decision_curve,
)


def test_ipcw_brier_matches_mse_without_censoring():
    durations = np.array([100.0, 500.0, 1200.0, 1600.0])
    events = np.ones(4, dtype=int)
    probabilities = np.array([0.8, 0.7, 0.3, 0.2])
    observed = np.array([1.0, 1.0, 0.0, 0.0])
    assert np.isclose(
        ipcw_brier_score(durations, events, probabilities, 1095),
        np.mean((observed - probabilities) ** 2),
    )


def test_ipcw_diagnostics_are_finite_for_stable_fixture():
    durations = np.array([90, 180, 420, 800, 1200, 1500, 1900, 2200], dtype=float)
    events = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    probability = np.linspace(0.75, 0.15, len(durations))
    matrix = np.column_stack([probability, probability * 0.95, probability * 0.9])
    assert np.isfinite(
        integrated_brier_score(durations, events, matrix, np.array([365, 1095, 1825]))
    )
    assert len(censoring_aware_calibration(durations, events, probability, 1095, bins=4)) == 4
    assert len(ipcw_decision_curve(durations, events, probability, 1095)) > 0

