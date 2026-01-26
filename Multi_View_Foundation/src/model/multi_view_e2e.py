"""End-to-end multi-view model that fine-tunes view backbones."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .projection_head import ProjectionHead


class MultiViewE2EModel(nn.Module):
    def __init__(
        self,
        view_backbones: Dict[str, nn.Module],
        view_dims: Dict[str, int],
        projection_dim: int,
        projection_hidden_dims=None,
        dropout: float = 0.1,
        timesteps: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.view_backbones = nn.ModuleDict(view_backbones)
        self.view_dims = view_dims
        self.timesteps = timesteps or {}

        self.heads = nn.ModuleDict()
        for view, dim in view_dims.items():
            self.heads[view] = ProjectionHead(
                input_dim=dim,
                output_dim=projection_dim,
                hidden_dims=projection_hidden_dims,
                dropout=dropout,
            )

    def _get_timestep(self, view: str, batch_size: int, device: torch.device) -> torch.Tensor:
        t = int(self.timesteps.get(view, 1))
        return torch.full((batch_size,), t, device=device, dtype=torch.long)

    def forward_view(self, view: str, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        backbone = self.view_backbones[view]
        if view == "graph":
            X = batch["X"]
            E = batch["E"]
            M = batch["M"]
            t = self._get_timestep(view, X.shape[0], X.device)
            pooled = backbone.get_node_embeddings(X, E, t, M, pooling="mean")
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            t = self._get_timestep(view, input_ids.shape[0], input_ids.device)
            pooled = backbone.get_pooled_output(
                input_ids=input_ids,
                timesteps=t,
                attention_mask=attention_mask,
                pooling="mean",
            )
        return self.heads[view](pooled)

    def forward_multi(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        outputs = {}
        for view, view_batch in batch.items():
            outputs[view] = self.forward_view(view, view_batch)
        return outputs
