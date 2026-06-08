"""Matplotlib survival plots for Stage 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _km_steps(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(durations, dtype=float)
    status = np.asarray(events, dtype=int)
    event_times = np.sort(np.unique(times[status == 1]))
    x = [0.0]
    y = [1.0]
    survival = 1.0
    for time in event_times:
        at_risk = np.sum(times >= time)
        deaths = np.sum((times == time) & (status == 1))
        if at_risk:
            survival *= 1.0 - deaths / at_risk
        x.append(float(time))
        y.append(float(survival))
    return np.asarray(x), np.asarray(y)


def plot_km_high_low(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    threshold: float,
    p_value: float,
    output_path: str | Path,
) -> None:
    high = np.asarray(risk_scores) >= threshold
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for group, label, color in (
        (high, "High risk", "#B13C2E"),
        (~high, "Low risk", "#267A73"),
    ):
        x, y = _km_steps(np.asarray(durations)[group], np.asarray(events)[group])
        ax.step(x, y, where="post", label=f"{label} (n={int(np.sum(group))})", color=color, linewidth=2)
    ax.set_xlabel("Overall survival time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0.0, 1.03)
    ax.set_title(f"Kaplan-Meier risk groups, log-rank P={p_value:.3g}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_time_dependent_auc(
    horizons: list[int],
    aucs: list[float],
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    years = np.asarray(horizons) / 365.0
    ax.plot(years, aucs, marker="o", color="#345995", linewidth=2)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(years, [f"{year:g} year" for year in years])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Time-dependent AUC")
    ax.set_title("Held-out test discrimination")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_calibration(
    predicted: np.ndarray,
    observed: np.ndarray,
    output_path: str | Path,
    *,
    horizon_days: int = 1095,
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    if len(predicted):
        ax.plot(predicted, observed, marker="o", color="#6A4C93", linewidth=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted event probability")
    ax.set_ylabel("Observed event fraction")
    ax.set_title(f"Calibration at {horizon_days / 365:g} years")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
