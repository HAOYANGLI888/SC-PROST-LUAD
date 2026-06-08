"""Survival losses shared by Cox and DeepSurv models."""

from __future__ import annotations

import numpy as np
import torch


def cox_ph_loss(
    log_risk: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Breslow tie handling."""

    scores = log_risk.reshape(-1)
    times = durations.reshape(-1)
    observed = events.reshape(-1).to(dtype=scores.dtype)
    if scores.numel() == 0:
        raise ValueError("Cox loss received an empty batch.")
    if torch.sum(observed) <= 0:
        raise ValueError("Cox loss requires at least one observed event.")
    order = torch.argsort(times, descending=True)
    sorted_scores = scores[order]
    sorted_times = times[order]
    sorted_events = observed[order]
    log_risk_sum = torch.logcumsumexp(sorted_scores, dim=0)
    _, tie_counts = torch.unique_consecutive(sorted_times, return_counts=True)
    tie_end_indices = torch.cumsum(tie_counts, dim=0) - 1
    breslow_denominator = torch.repeat_interleave(log_risk_sum[tie_end_indices], tie_counts)
    partial = (sorted_scores - breslow_denominator) * sorted_events
    return -torch.sum(partial) / torch.sum(sorted_events)


def concordance_index(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Compute Harrell's C-index where larger scores indicate higher risk."""

    times = np.asarray(durations, dtype=float).reshape(-1)
    status = np.asarray(events, dtype=int).reshape(-1)
    scores = np.asarray(risk_scores, dtype=float).reshape(-1)
    if not (len(times) == len(status) == len(scores)):
        raise ValueError("Durations, events, and scores must have equal length.")
    permissible = 0.0
    concordant = 0.0
    for index in range(len(times)):
        if status[index] != 1:
            continue
        later = times > times[index]
        comparable_scores = scores[later]
        permissible += float(comparable_scores.size)
        concordant += float(np.sum(scores[index] > comparable_scores))
        concordant += 0.5 * float(np.sum(scores[index] == comparable_scores))
    return float(concordant / permissible) if permissible > 0 else float("nan")
