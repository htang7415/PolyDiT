"""Utilities for consistent checkpoint loading across MVF steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        payload = checkpoint.get("model_state_dict")
        if isinstance(payload, Mapping):
            return payload
        payload = checkpoint.get("state_dict")
        if isinstance(payload, Mapping):
            return payload
        return checkpoint
    raise ValueError("Unsupported checkpoint payload: expected mapping-like object.")


def load_backbone_checkpoint(
    *,
    backbone: torch.nn.Module,
    checkpoint_path: str | Path,
    map_location: str,
    strict: bool = False,
    prefix: str = "backbone.",
    label: str = "backbone",
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load checkpoint weights into a backbone and report key mismatches.

    Returns torch's incompatible-keys object.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)

    cleaned = {
        str(k).replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in state_dict.items()
    }
    scoped = {
        key[len(prefix):]: value
        for key, value in cleaned.items()
        if prefix and key.startswith(prefix)
    }
    final_state = scoped if scoped else cleaned

    incompatible = backbone.load_state_dict(final_state, strict=strict)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"Warning: mismatch when loading {label} weights from {checkpoint_path}.")
        if incompatible.missing_keys:
            print(f"  Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
    return incompatible
