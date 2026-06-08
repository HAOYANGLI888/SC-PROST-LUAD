"""Gated attention MIL survival model for pathology bags."""

from __future__ import annotations

import torch

from models.pathology_survival_heads import CoxSurvivalHead


class GatedAttentionMILSurvival(torch.nn.Module):
    """Use gated attention before patient-level Cox risk prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.project = torch.nn.Linear(input_dim, hidden_dim)
        self.attention_v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attention_u = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = torch.nn.Linear(hidden_dim, 1)
        self.head = CoxSurvivalHead(hidden_dim)

    def encode_bag(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.project(bag))
        gated = torch.tanh(self.attention_v(hidden)) * torch.sigmoid(self.attention_u(hidden))
        attention = torch.softmax(self.attention_w(gated).reshape(-1), dim=0)
        pooled = torch.sum(attention[:, None] * hidden, dim=0)
        return pooled, attention

    def forward(self, bags: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        pooled_and_attention = [self.encode_bag(bag) for bag in bags]
        pooled = torch.stack([item[0] for item in pooled_and_attention])
        return self.head(pooled), [item[1] for item in pooled_and_attention]

