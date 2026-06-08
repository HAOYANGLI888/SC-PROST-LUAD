"""Simple clinical + pathology fusion for Stage 6A proof-of-concept."""

from __future__ import annotations

import torch

from models.pathology_survival_heads import CoxSurvivalHead


class ClinicalPathologyFusionSurvival(torch.nn.Module):
    """Fuse gated patient pathology embedding with core clinical covariates."""

    def __init__(self, pathology_dim: int, clinical_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.project = torch.nn.Linear(pathology_dim, hidden_dim)
        self.attention_v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attention_u = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = torch.nn.Linear(hidden_dim, 1)
        self.head = CoxSurvivalHead(hidden_dim + clinical_dim)

    def encode_bag(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.project(bag))
        gated = torch.tanh(self.attention_v(hidden)) * torch.sigmoid(self.attention_u(hidden))
        attention = torch.softmax(self.attention_w(gated).reshape(-1), dim=0)
        return torch.sum(attention[:, None] * hidden, dim=0), attention

    def forward(self, bags: list[torch.Tensor], clinical: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        encoded = [self.encode_bag(bag) for bag in bags]
        pooled = torch.stack([item[0] for item in encoded])
        return self.head(torch.cat([pooled, clinical], dim=1)), [item[1] for item in encoded]

