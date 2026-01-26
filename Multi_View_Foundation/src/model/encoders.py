"""Per-view encoder helpers for multi-view alignment."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """Lightweight encoder for precomputed embeddings."""

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None or hidden_dim == input_dim:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmilesEncoder(BaseEncoder):
    pass


class SelfiesEncoder(BaseEncoder):
    pass


class GroupSelfiesEncoder(BaseEncoder):
    pass


class GraphEncoder(BaseEncoder):
    pass


def build_encoder(view: str, input_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0) -> nn.Module:
    view = view.lower()
    cls_map = {
        "smiles": SmilesEncoder,
        "selfies": SelfiesEncoder,
        "group_selfies": GroupSelfiesEncoder,
        "graph": GraphEncoder,
    }
    cls = cls_map.get(view, BaseEncoder)
    return cls(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
