"""Cox and Random Survival Forest baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from models.survival_losses import cox_ph_loss


@dataclass
class TorchCoxElasticNet:
    """Linear Cox model trained with Adam and proximal L1 shrinkage."""

    l1_ratio: float = 0.0
    alpha: float = 0.0
    learning_rate: float = 0.03
    epochs: int = 160
    seed: int = 42

    def fit(self, x: np.ndarray, durations: np.ndarray, events: np.ndarray) -> "TorchCoxElasticNet":
        features = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        times = torch.as_tensor(np.array(durations, copy=True), dtype=torch.float32)
        status = torch.as_tensor(np.array(events, copy=True), dtype=torch.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("Cox model requires a non-empty 2D feature matrix.")
        torch.manual_seed(self.seed)
        weights = torch.nn.Parameter(torch.zeros(features.shape[1], dtype=torch.float32))
        optimizer = torch.optim.Adam([weights], lr=self.learning_rate)
        l2_strength = self.alpha * (1.0 - self.l1_ratio)
        l1_strength = self.alpha * self.l1_ratio
        for _ in range(self.epochs):
            optimizer.zero_grad()
            score = features @ weights
            loss = cox_ph_loss(score, times, status)
            if l2_strength:
                loss = loss + l2_strength * torch.sum(weights.square())
            loss.backward()
            optimizer.step()
            if l1_strength:
                with torch.no_grad():
                    shrink = self.learning_rate * l1_strength
                    weights.copy_(torch.sign(weights) * torch.clamp(torch.abs(weights) - shrink, min=0.0))
        self.coef_ = weights.detach().cpu().numpy()
        self.backend_ = "custom_pytorch_cox_elasticnet"
        return self

    def predict_risk(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Cox model must be fit before prediction.")
        return np.asarray(x, dtype=float) @ self.coef_


@dataclass
class RandomSurvivalForestAdapter:
    """Use scikit-survival RSF when present, otherwise an explicit fallback."""

    n_estimators: int = 120
    min_samples_leaf: int = 5
    seed: int = 42

    def fit(self, x: np.ndarray, durations: np.ndarray, events: np.ndarray) -> "RandomSurvivalForestAdapter":
        features = np.asarray(x, dtype=float)
        times = np.asarray(durations, dtype=float)
        status = np.asarray(events, dtype=bool)
        try:
            from sksurv.ensemble import RandomSurvivalForest

            model: Any = RandomSurvivalForest(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed,
                n_jobs=1,
            )
            target = np.array(
                list(zip(status, times)),
                dtype=[("event", "?"), ("time", "<f8")],
            )
            model.fit(features, target)
            self.model_ = model
            self.backend_ = "scikit_survival_random_survival_forest"
        except ImportError:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed,
                n_jobs=1,
            )
            weights = np.where(status, 1.0, 0.35)
            model.fit(features, np.log1p(times), sample_weight=weights)
            self.model_ = model
            self.backend_ = "sklearn_random_forest_time_proxy_fallback"
        return self

    def predict_risk(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("RSF adapter must be fit before prediction.")
        predictions = self.model_.predict(np.asarray(x, dtype=float))
        return predictions if self.backend_.startswith("scikit_survival") else -predictions
