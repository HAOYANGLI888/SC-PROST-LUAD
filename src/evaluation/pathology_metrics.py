"""Pathology proof-of-concept metric helpers."""

from __future__ import annotations

import numpy as np

from evaluation.survival_metrics import concordance_index, logrank_p_value, time_dependent_auc


def pathology_survival_metrics(times, events, scores) -> dict[str, float]:
    """Return dependency-light Stage 6A survival metrics."""

    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    scores = np.asarray(scores, dtype=float)
    cutoff = float(np.median(scores))
    return {
        "c_index": concordance_index(times, events, scores),
        "auc_1_year": time_dependent_auc(times, events, scores, 365),
        "auc_3_year": time_dependent_auc(times, events, scores, 1095),
        "auc_5_year": time_dependent_auc(times, events, scores, 1825),
        "km_logrank_p": logrank_p_value(times, events, scores >= cutoff),
        "risk_cutoff": cutoff,
    }

