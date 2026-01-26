"""Multi-view alignment model."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from .encoders import build_encoder
from .projection_head import ProjectionHead


class MultiViewModel(nn.Module):
    def __init__(
        self,
        view_dims: Dict[str, int],
        projection_dim: int,
        encoder_hidden_dim: Optional[int] = None,
        projection_hidden_dims: Optional[Iterable[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.view_dims = view_dims
        self.projection_dim = projection_dim

        self.encoders = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        for view, dim in view_dims.items():
            hidden_dim = encoder_hidden_dim if encoder_hidden_dim else dim
            self.encoders[view] = build_encoder(view, dim, hidden_dim=hidden_dim, dropout=dropout)
            self.heads[view] = ProjectionHead(
                input_dim=hidden_dim,
                output_dim=projection_dim,
                hidden_dims=projection_hidden_dims,
                dropout=dropout,
            )

    def forward(self, view: str, x: torch.Tensor) -> torch.Tensor:
        if view not in self.encoders:
            raise KeyError(f"Unknown view: {view}")
        h = self.encoders[view](x)
        return self.heads[view](h)

    def forward_multi(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {view: self.forward(view, emb) for view, emb in batch.items()}
