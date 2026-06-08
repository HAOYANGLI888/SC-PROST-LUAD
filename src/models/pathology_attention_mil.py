"""Attention MIL survival model for pathology bags."""

from __future__ import annotations

import torch

from models.pathology_survival_heads import CoxSurvivalHead


class AttentionMILSurvival(torch.nn.Module):
    """Aggregate patch embeddings with learned attention and predict log-risk."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.project = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.Tanh())
        self.attention = torch.nn.Linear(hidden_dim, 1)
        self.head = CoxSurvivalHead(hidden_dim)

    def encode_bag(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.project(bag)
        attention = torch.softmax(self.attention(hidden).reshape(-1), dim=0)
        pooled = torch.sum(attention[:, None] * hidden, dim=0)
        return pooled, attention

    def forward(self, bags: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        pooled_and_attention = [self.encode_bag(bag) for bag in bags]
        pooled = torch.stack([item[0] for item in pooled_and_attention])
        return self.head(pooled), [item[1] for item in pooled_and_attention]

