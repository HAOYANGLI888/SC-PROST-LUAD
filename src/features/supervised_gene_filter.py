"""Train-fold-only supervised RNA gene selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from data.rnaseq_preprocess import RNATrainPreprocessor, RNASeqPreprocessError
from models.cox_baseline import TorchCoxElasticNet


def _as_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.fillna(numeric.median(axis=0).fillna(0.0))


def univariate_cox_score(
    frame: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
) -> pd.Series:
    """Return absolute Cox score statistics evaluated at beta=0.

    The score test is vectorized across genes. It is used only to rank genes
    inside a training fold; it is not reported as an inferential result.
    """

    numeric = _as_numeric(frame)
    x = numeric.to_numpy(dtype=float)
    times = np.asarray(durations, dtype=float)
    status = np.asarray(events, dtype=int)
    if x.shape[0] != len(times) or len(times) != len(status):
        raise ValueError("RNA rows, durations, and events must have equal length.")
    if np.sum(status) == 0:
        raise ValueError("Univariate Cox filtering requires at least one observed event.")
    order = np.argsort(-times, kind="mergesort")
    ordered_x = x[order]
    ordered_status = status[order]
    cumulative_mean = np.cumsum(ordered_x, axis=0) / np.arange(1, len(x) + 1)[:, None]
    score = np.sum((ordered_x - cumulative_mean) * ordered_status[:, None], axis=0)
    scales = np.std(x, axis=0, ddof=0)
    scales[scales == 0] = 1.0
    return pd.Series(np.abs(score / scales), index=numeric.columns, dtype=float)


@dataclass
class UnivariateCoxGeneFilter:
    """Select genes by a train-fold-only univariate Cox score."""

    max_features: int = 100
    prefilter_top_variable_genes: int = 1000

    def fit(
        self,
        frame: pd.DataFrame,
        durations: np.ndarray,
        events: np.ndarray,
        patient_ids: Iterable[str] | None = None,
    ) -> "UnivariateCoxGeneFilter":
        self.prefilter_ = RNATrainPreprocessor(
            top_variable_genes=self.prefilter_top_variable_genes,
        ).fit(frame, patient_ids=patient_ids)
        filtered = self.prefilter_.transform(frame)
        scores = univariate_cox_score(filtered, durations, events)
        self.selected_genes_ = scores.sort_values(ascending=False).head(self.max_features).index.tolist()
        if not self.selected_genes_:
            raise RNASeqPreprocessError("Univariate Cox filtering retained no genes.")
        self.fit_patient_ids_ = self.prefilter_.fit_patient_ids_
        self.scores_ = scores.loc[self.selected_genes_]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_genes_"):
            raise RNASeqPreprocessError("Univariate Cox filter must be fit first.")
        return self.prefilter_.transform(frame)[self.selected_genes_]


@dataclass
class ElasticNetGeneSelector:
    """Rank genes using a train-fold-only proximal ElasticNet-Cox fit."""

    max_features: int = 100
    prefilter_top_variable_genes: int = 500
    alpha: float = 0.01
    l1_ratio: float = 0.80
    epochs: int = 55
    seed: int = 42

    def fit(
        self,
        frame: pd.DataFrame,
        durations: np.ndarray,
        events: np.ndarray,
        patient_ids: Iterable[str] | None = None,
    ) -> "ElasticNetGeneSelector":
        self.prefilter_ = RNATrainPreprocessor(
            top_variable_genes=self.prefilter_top_variable_genes,
        ).fit(frame, patient_ids=patient_ids)
        filtered = self.prefilter_.transform(frame)
        model = TorchCoxElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            epochs=self.epochs,
            seed=self.seed,
        ).fit(filtered.to_numpy(), durations, events)
        coefficients = pd.Series(np.abs(model.coef_), index=filtered.columns)
        nonzero = coefficients.loc[coefficients > 1e-8].sort_values(ascending=False)
        ranked = nonzero if len(nonzero) >= min(5, self.max_features) else coefficients.sort_values(ascending=False)
        self.selected_genes_ = ranked.head(self.max_features).index.tolist()
        if not self.selected_genes_:
            raise RNASeqPreprocessError("ElasticNet gene selector retained no genes.")
        self.fit_patient_ids_ = self.prefilter_.fit_patient_ids_
        self.coefficients_ = coefficients.loc[self.selected_genes_]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_genes_"):
            raise RNASeqPreprocessError("ElasticNet gene selector must be fit first.")
        return self.prefilter_.transform(frame)[self.selected_genes_]

