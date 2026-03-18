"""Shared MVF backbone and property-scorer loading helpers.

This module exists so F5/F6 can share scorer-loading logic without importing
step scripts by filename.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.checkpoint_loading import load_backbone_checkpoint
from src.utils.config import load_config
from src.utils.property_names import normalize_property_name
from src.utils.runtime import (
    load_module,
    resolve_path as _shared_resolve_path,
    resolve_with_base as _shared_resolve_with_base,
)

try:  # pragma: no cover
    import joblib
except Exception:  # pragma: no cover
    joblib = None


BASE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BASE_DIR.parent

SUPPORTED_VIEWS = ("smiles", "smiles_bpe", "selfies", "group_selfies", "graph")
VIEW_SPECS = {
    "smiles": {
        "type": "sequence",
        "encoder_key": "smiles_encoder",
        "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SMILES" / "src" / "data" / "tokenizer.py",
        "tokenizer_class": "PSmilesTokenizer",
        "tokenizer_file": "tokenizer.json",
    },
    "smiles_bpe": {
        "type": "sequence",
        "encoder_key": "smiles_bpe_encoder",
        "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SMILES_BPE" / "src" / "data" / "tokenizer.py",
        "tokenizer_class": "PSmilesTokenizer",
        "tokenizer_file": "tokenizer.json",
    },
    "selfies": {
        "type": "sequence",
        "encoder_key": "selfies_encoder",
        "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SELFIES" / "src" / "data" / "selfies_tokenizer.py",
        "tokenizer_class": "SelfiesTokenizer",
        "tokenizer_file": "tokenizer.json",
    },
    "group_selfies": {
        "type": "sequence",
        "encoder_key": "group_selfies_encoder",
        "tokenizer_module": REPO_ROOT / "Bi_Diffusion_Group_SELFIES" / "src" / "data" / "tokenizer.py",
        "tokenizer_class": "GroupSELFIESTokenizer",
        "tokenizer_file": "tokenizer.pkl",
    },
    "graph": {
        "type": "graph",
        "encoder_key": "graph_encoder",
    },
}


def resolve_path(path_str: str | Path) -> Path:
    return _shared_resolve_path(path_str, BASE_DIR)


def resolve_with_base(path_str: str | Path, base_dir: Path) -> Path:
    return _shared_resolve_with_base(path_str, base_dir)


def resolve_view_device(config: dict, view: str) -> str:
    encoder_cfg = config.get(VIEW_SPECS[view]["encoder_key"], {})
    device = str(encoder_cfg.get("device", "auto")).strip() or "auto"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_sequence_backbone(
    encoder_cfg: dict,
    device: str,
    tokenizer_module: Path,
    tokenizer_class: str,
    tokenizer_filename: str,
):
    method_dir = resolve_path(encoder_cfg.get("method_dir"))
    config_path = encoder_cfg.get("config_path")
    if config_path:
        config_path = resolve_path(config_path)
    else:
        config_path = method_dir / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    method_cfg = load_config(str(config_path))
    scales_mod = load_module(
        f"scales_{method_dir.name}",
        method_dir / "src" / "utils" / "model_scales.py",
        REPO_ROOT,
    )
    tokenizer_mod = load_module(
        f"tokenizer_{method_dir.name}",
        tokenizer_module,
        REPO_ROOT,
    )
    backbone_mod = load_module(
        f"backbone_{method_dir.name}",
        method_dir / "src" / "model" / "backbone.py",
        REPO_ROOT,
    )

    get_model_config = scales_mod.get_model_config
    get_results_dir = scales_mod.get_results_dir
    tokenizer_cls = getattr(tokenizer_mod, tokenizer_class)
    diffusion_backbone = backbone_mod.DiffusionBackbone

    model_size = encoder_cfg.get("model_size")
    backbone_config = get_model_config(model_size, method_cfg, model_type="sequence")
    diffusion_steps = method_cfg.get("diffusion", {}).get("num_steps", 50)

    base_results_dir = encoder_cfg.get("results_dir")
    if base_results_dir:
        base_results_dir = resolve_path(base_results_dir)
    else:
        base_results_dir = resolve_with_base(method_cfg["paths"]["results_dir"], method_dir)

    results_dir = Path(get_results_dir(model_size, str(base_results_dir)))

    tokenizer_path = encoder_cfg.get("tokenizer_path")
    if tokenizer_path:
        tokenizer_path = resolve_path(tokenizer_path)
    else:
        tokenizer_path = results_dir / tokenizer_filename
        if not tokenizer_path.exists():
            tokenizer_path = base_results_dir / tokenizer_filename

    checkpoint_path = encoder_cfg.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = resolve_path(checkpoint_path)
    else:
        step_dir = encoder_cfg.get("step_dir", "step1_backbone")
        checkpoint_name = encoder_cfg.get("checkpoint_name", "backbone_best.pt")
        checkpoint_path = results_dir / step_dir / "checkpoints" / checkpoint_name

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer = tokenizer_cls.load(str(tokenizer_path))
    backbone = diffusion_backbone(
        vocab_size=tokenizer.vocab_size,
        hidden_size=backbone_config["hidden_size"],
        num_layers=backbone_config["num_layers"],
        num_heads=backbone_config["num_heads"],
        ffn_hidden_size=backbone_config["ffn_hidden_size"],
        max_position_embeddings=backbone_config.get("max_position_embeddings", 256),
        num_diffusion_steps=diffusion_steps,
        dropout=backbone_config.get("dropout", 0.1),
        pad_token_id=tokenizer.pad_token_id,
    )

    load_backbone_checkpoint(
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        map_location=device,
        strict=False,
        prefix="backbone.",
        label="sequence backbone",
    )
    backbone.to(device)
    backbone.eval()

    return {
        "backbone": backbone,
        "tokenizer": tokenizer,
        "model_size": model_size or "base",
        "pooling": encoder_cfg.get("pooling", "mean"),
        "timestep": int(encoder_cfg.get("timestep", 1)),
        "batch_size": int(encoder_cfg.get("batch_size", 256)),
        "method_dir": method_dir,
        "method_cfg": method_cfg,
        "results_dir": results_dir,
        "base_results_dir": base_results_dir,
        "checkpoint_path": checkpoint_path,
    }


def _load_graph_backbone(encoder_cfg: dict, device: str):
    method_dir = resolve_path(encoder_cfg.get("method_dir"))
    config_path = encoder_cfg.get("config_path")
    if config_path:
        config_path = resolve_path(config_path)
    else:
        config_path = method_dir / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    method_cfg = load_config(str(config_path))
    scales_mod = load_module(
        f"graph_scales_{method_dir.name}",
        method_dir / "src" / "utils" / "model_scales.py",
        REPO_ROOT,
    )
    tokenizer_mod = load_module(
        f"graph_tokenizer_{method_dir.name}",
        method_dir / "src" / "data" / "graph_tokenizer.py",
        REPO_ROOT,
    )
    backbone_mod = load_module(
        f"graph_backbone_{method_dir.name}",
        method_dir / "src" / "model" / "graph_backbone.py",
        REPO_ROOT,
    )

    get_model_config = scales_mod.get_model_config
    get_results_dir = scales_mod.get_results_dir
    graph_tokenizer = tokenizer_mod.GraphTokenizer
    graph_backbone = backbone_mod.GraphDiffusionBackbone

    model_size = encoder_cfg.get("model_size")
    backbone_config = get_model_config(model_size, method_cfg, model_type="graph")
    diffusion_steps = method_cfg.get("diffusion", {}).get("num_steps", 50)

    base_results_dir = encoder_cfg.get("results_dir")
    if base_results_dir:
        base_results_dir = resolve_path(base_results_dir)
    else:
        base_results_dir = resolve_with_base(method_cfg["paths"]["results_dir"], method_dir)

    results_dir = Path(get_results_dir(model_size, str(base_results_dir)))

    tokenizer_path = encoder_cfg.get("tokenizer_path")
    if tokenizer_path:
        tokenizer_path = resolve_path(tokenizer_path)
    else:
        tokenizer_path = results_dir / "graph_tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = base_results_dir / "graph_tokenizer.json"

    graph_config_path = encoder_cfg.get("graph_config_path")
    if graph_config_path:
        graph_config_path = resolve_path(graph_config_path)
    else:
        graph_config_path = results_dir / "graph_config.json"
        if not graph_config_path.exists():
            graph_config_path = base_results_dir / "graph_config.json"

    checkpoint_path = encoder_cfg.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = resolve_path(checkpoint_path)
    else:
        step_dir = encoder_cfg.get("step_dir", "step1_backbone")
        checkpoint_name = encoder_cfg.get("checkpoint_name", "graph_backbone_best.pt")
        checkpoint_path = results_dir / step_dir / "checkpoints" / checkpoint_name

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Graph tokenizer not found: {tokenizer_path}")
    if not graph_config_path.exists():
        raise FileNotFoundError(f"Graph config not found: {graph_config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Graph checkpoint not found: {checkpoint_path}")

    with open(graph_config_path, "r", encoding="utf-8") as f:
        graph_config = json.load(f)

    tokenizer = graph_tokenizer.load(str(tokenizer_path))
    backbone = graph_backbone(
        atom_vocab_size=graph_config["atom_vocab_size"],
        edge_vocab_size=graph_config["edge_vocab_size"],
        Nmax=graph_config["Nmax"],
        hidden_size=backbone_config["hidden_size"],
        num_layers=backbone_config["num_layers"],
        num_heads=backbone_config["num_heads"],
        ffn_hidden_size=backbone_config["ffn_hidden_size"],
        dropout=backbone_config.get("dropout", 0.1),
        num_diffusion_steps=diffusion_steps,
    )

    load_backbone_checkpoint(
        backbone=backbone,
        checkpoint_path=checkpoint_path,
        map_location=device,
        strict=False,
        prefix="backbone.",
        label="graph backbone",
    )
    backbone.to(device)
    backbone.eval()

    return {
        "backbone": backbone,
        "tokenizer": tokenizer,
        "model_size": model_size or "base",
        "pooling": encoder_cfg.get("pooling", "mean"),
        "timestep": int(encoder_cfg.get("timestep", 1)),
        "batch_size": int(encoder_cfg.get("batch_size", 64)),
        "method_dir": method_dir,
        "method_cfg": method_cfg,
        "results_dir": results_dir,
        "base_results_dir": base_results_dir,
        "checkpoint_path": checkpoint_path,
        "graph_config": graph_config,
    }


def load_view_assets(config: dict, view: str, device: str) -> dict:
    spec = VIEW_SPECS[view]
    encoder_cfg = config.get(spec["encoder_key"], {})
    if not encoder_cfg or not encoder_cfg.get("method_dir"):
        raise ValueError(f"Encoder config missing for view={view}")

    if spec["type"] == "graph":
        return _load_graph_backbone(encoder_cfg=encoder_cfg, device=device)
    return _load_sequence_backbone(
        encoder_cfg=encoder_cfg,
        device=device,
        tokenizer_module=spec["tokenizer_module"],
        tokenizer_class=spec["tokenizer_class"],
        tokenizer_filename=spec["tokenizer_file"],
    )


def default_property_model_path(results_dir: Path, property_name: str, view: str) -> Path:
    prop = normalize_property_name(property_name)
    model_dir = Path(results_dir) / "step3_property"
    if prop:
        model_dir = model_dir / prop / "files"
    else:
        model_dir = model_dir / "files"
    canonical = model_dir / f"{prop}_{view}_mlp.pt"
    if view != "smiles":
        return canonical
    legacy = model_dir / f"{prop}_mlp.pt"
    if canonical.exists() or not legacy.exists():
        return canonical
    return legacy


class PropertyMLP(torch.nn.Module):
    def __init__(self, input_dim: int, num_layers: int, neurons: int, dropout: float):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(max(int(num_layers), 1)):
            layers.append(torch.nn.Linear(current_dim, int(neurons)))
            layers.append(torch.nn.ReLU())
            if float(dropout) > 0:
                layers.append(torch.nn.Dropout(float(dropout)))
            current_dim = int(neurons)
        layers.append(torch.nn.Linear(current_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchPropertyPredictor:
    def __init__(self, checkpoint: dict):
        if checkpoint.get("format") != "mvf_torch_mlp":
            raise ValueError("Unsupported torch property model format.")
        self.mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        self.scale = np.where(np.abs(self.scale) < 1e-12, 1.0, self.scale).astype(np.float32, copy=False)
        self.model = PropertyMLP(
            input_dim=int(checkpoint["input_dim"]),
            num_layers=int(checkpoint["num_layers"]),
            neurons=int(checkpoint["neurons"]),
            dropout=float(checkpoint.get("dropout", 0.0)),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features is None or len(features) == 0:
            return np.zeros((0,), dtype=np.float32)
        x = np.asarray(features, dtype=np.float32)
        x = (x - self.mean) / self.scale
        with torch.no_grad():
            preds = self.model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
        return preds


def load_property_model(model_path: Path):
    model_path = Path(model_path)
    if model_path.suffix in {".pt", ".pth"}:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and checkpoint.get("format") == "mvf_torch_mlp":
            return TorchPropertyPredictor(checkpoint)
        raise ValueError(f"Unsupported torch property model checkpoint: {model_path}")

    if joblib is not None:
        return joblib.load(model_path)
    import pickle

    with open(model_path, "rb") as f:
        return pickle.load(f)


# Compatibility aliases for older step-local names.
_TorchPropertyPredictor = TorchPropertyPredictor
_default_property_model_path = default_property_model_path
_load_property_model = load_property_model
_load_view_assets = load_view_assets
_resolve_path = resolve_path
_resolve_view_device = resolve_view_device
