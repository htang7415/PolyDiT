"""Projection head for contrastive learning."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[Iterable[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims or []) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
