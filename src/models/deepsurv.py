"""Small PyTorch DeepSurv implementation for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.survival_losses import cox_ph_loss


class DeepSurvNetwork(torch.nn.Module):
    """MLP producing one log-risk score per patient."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        current = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    torch.nn.Linear(current, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                ]
            )
            current = hidden
        layers.append(torch.nn.Linear(current, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).reshape(-1)


@dataclass
class DeepSurvEstimator:
    """Train a compact DeepSurv network with deterministic CPU defaults."""

    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.15
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    epochs: int = 100
    seed: int = 42

    def fit(self, x: np.ndarray, durations: np.ndarray, events: np.ndarray) -> "DeepSurvEstimator":
        features = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        times = torch.as_tensor(np.array(durations, copy=True), dtype=torch.float32)
        status = torch.as_tensor(np.array(events, copy=True), dtype=torch.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("DeepSurv requires a non-empty 2D feature matrix.")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.network_ = DeepSurvNetwork(features.shape[1], self.hidden_dims, self.dropout)
        optimizer = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.network_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = cox_ph_loss(self.network_(features), times, status)
            loss.backward()
            optimizer.step()
        self.backend_ = "custom_pytorch_deepsurv"
        return self

    def predict_risk(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "network_"):
            raise RuntimeError("DeepSurv model must be fit before prediction.")
        self.network_.eval()
        with torch.no_grad():
            features = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            return self.network_(features).cpu().numpy()

    def checkpoint(self) -> dict[str, object]:
        if not hasattr(self, "network_"):
            raise RuntimeError("DeepSurv model must be fit before checkpointing.")
        return {
            "state_dict": self.network_.state_dict(),
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "backend": self.backend_,
        }
