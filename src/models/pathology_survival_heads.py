"""Survival heads for pathology MIL models."""

from __future__ import annotations

import torch


class CoxSurvivalHead(torch.nn.Module):
    """Linear log-risk head for Cox partial-likelihood training."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).reshape(-1)

