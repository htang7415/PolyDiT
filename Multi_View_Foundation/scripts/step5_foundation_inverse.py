#!/usr/bin/env python
"""F5: Foundation-enhanced inverse design with reranking."""

import argparse
import json
from pathlib import Path
import sys
import time
import importlib.util
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.data.view_converters import smiles_to_selfies
from shared.ood_metrics import knn_distances
from src.model.multi_view_model import MultiViewModel

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


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


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _resolve_with_base(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path)


def _view_to_representation(view: str) -> str:
    mapping = {
        "smiles": "SMILES",
        "smiles_bpe": "SMILES_BPE",
        "selfies": "SELFIES",
        "group_selfies": "Group_SELFIES",
        "graph": "Graph",
    }
    return mapping.get(view, view)


def _is_view_enabled(config: dict, view: str) -> bool:
    views_cfg = config.get("views", {})
    if view in views_cfg and not views_cfg[view].get("enabled", True):
        return False
    return True


def _select_encoder_view(config: dict, override: Optional[str]) -> str:
    requested = override
    if requested is None:
        requested = str(config.get("foundation_inverse", {}).get("encoder_view", "")).strip() or None

    if requested:
        if requested not in SUPPORTED_VIEWS:
            raise ValueError(f"Unsupported encoder_view={requested}. Supported: {', '.join(SUPPORTED_VIEWS)}")
        if not _is_view_enabled(config, requested):
            raise ValueError(f"encoder_view={requested} is disabled in config.views.")
        encoder_key = VIEW_SPECS[requested]["encoder_key"]
        if not config.get(encoder_key, {}).get("method_dir"):
            raise ValueError(f"Encoder config missing for view={requested} ({encoder_key}.method_dir).")
        return requested

    candidate_views = []
    configured_order = config.get("alignment_views", list(SUPPORTED_VIEWS))
    for view in configured_order:
        if view in SUPPORTED_VIEWS:
            candidate_views.append(view)
    for view in SUPPORTED_VIEWS:
        if view not in candidate_views:
            candidate_views.append(view)

    for view in candidate_views:
        if not _is_view_enabled(config, view):
            continue
        encoder_key = VIEW_SPECS[view]["encoder_key"]
        if config.get(encoder_key, {}).get("method_dir"):
            return view
    return "smiles"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_alignment_model(results_dir: Path, view_dims: dict, config: dict, checkpoint_override: Optional[str]):
    ckpt_path = _resolve_path(checkpoint_override) if checkpoint_override else results_dir / "step1_alignment" / "alignment_best.pt"
    if not ckpt_path.exists():
        return None
    model_cfg = config.get("model", {})
    model = MultiViewModel(
        view_dims=view_dims,
        projection_dim=int(model_cfg.get("projection_dim", 256)),
        projection_hidden_dims=model_cfg.get("projection_hidden_dims"),
        dropout=float(model_cfg.get("view_dropout", 0.0)),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def _project_embeddings(model: MultiViewModel, view: str, embeddings: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    if embeddings is None or embeddings.size == 0:
        return embeddings
    model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[start:start + batch_size], device=device, dtype=torch.float32)
            z = model.forward(view, batch)
            outputs.append(z.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _load_sequence_backbone(
    encoder_cfg: dict,
    device: str,
    tokenizer_module: Path,
    tokenizer_class: str,
    tokenizer_filename: str,
):
    method_dir = _resolve_path(encoder_cfg.get("method_dir"))
    config_path = encoder_cfg.get("config_path")
    if config_path:
        config_path = _resolve_path(config_path)
    else:
        config_path = method_dir / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    method_cfg = load_config(str(config_path))

    scales_mod = _load_module(
        f"scales_{method_dir.name}",
        method_dir / "src" / "utils" / "model_scales.py",
    )
    tokenizer_mod = _load_module(
        f"tokenizer_{method_dir.name}",
        tokenizer_module,
    )
    backbone_mod = _load_module(
        f"backbone_{method_dir.name}",
        method_dir / "src" / "model" / "backbone.py",
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
        base_results_dir = _resolve_path(base_results_dir)
    else:
        base_results_dir = _resolve_with_base(method_cfg["paths"]["results_dir"], method_dir)

    results_dir = Path(get_results_dir(model_size, str(base_results_dir)))

    tokenizer_path = encoder_cfg.get("tokenizer_path")
    if tokenizer_path:
        tokenizer_path = _resolve_path(tokenizer_path)
    else:
        tokenizer_path = results_dir / tokenizer_filename
        if not tokenizer_path.exists():
            tokenizer_path = base_results_dir / tokenizer_filename

    checkpoint_path = encoder_cfg.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = _resolve_path(checkpoint_path)
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    cleaned = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
    backbone_state = {
        key[len("backbone."):]: value
        for key, value in cleaned.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = cleaned

    backbone.load_state_dict(backbone_state, strict=False)
    backbone.to(device)
    backbone.eval()

    return {
        "backbone": backbone,
        "tokenizer": tokenizer,
        "model_size": model_size or "base",
        "pooling": encoder_cfg.get("pooling", "mean"),
        "timestep": int(encoder_cfg.get("timestep", 1)),
        "batch_size": int(encoder_cfg.get("batch_size", 256)),
    }


def _load_graph_backbone(encoder_cfg: dict, device: str):
    method_dir = _resolve_path(encoder_cfg.get("method_dir"))
    config_path = encoder_cfg.get("config_path")
    if config_path:
        config_path = _resolve_path(config_path)
    else:
        config_path = method_dir / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    method_cfg = load_config(str(config_path))

    scales_mod = _load_module(
        f"graph_scales_{method_dir.name}",
        method_dir / "src" / "utils" / "model_scales.py",
    )
    tokenizer_mod = _load_module(
        f"graph_tokenizer_{method_dir.name}",
        method_dir / "src" / "data" / "graph_tokenizer.py",
    )
    backbone_mod = _load_module(
        f"graph_backbone_{method_dir.name}",
        method_dir / "src" / "model" / "graph_backbone.py",
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
        base_results_dir = _resolve_path(base_results_dir)
    else:
        base_results_dir = _resolve_with_base(method_cfg["paths"]["results_dir"], method_dir)

    results_dir = Path(get_results_dir(model_size, str(base_results_dir)))

    tokenizer_path = encoder_cfg.get("tokenizer_path")
    if tokenizer_path:
        tokenizer_path = _resolve_path(tokenizer_path)
    else:
        tokenizer_path = results_dir / "graph_tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = base_results_dir / "graph_tokenizer.json"

    graph_config_path = encoder_cfg.get("graph_config_path")
    if graph_config_path:
        graph_config_path = _resolve_path(graph_config_path)
    else:
        graph_config_path = results_dir / "graph_config.json"
        if not graph_config_path.exists():
            graph_config_path = base_results_dir / "graph_config.json"

    checkpoint_path = encoder_cfg.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = _resolve_path(checkpoint_path)
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

    with open(graph_config_path, "r") as f:
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    cleaned = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
    backbone_state = {
        key[len("backbone."):]: value
        for key, value in cleaned.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = cleaned

    backbone.load_state_dict(backbone_state, strict=False)
    backbone.to(device)
    backbone.eval()

    return {
        "backbone": backbone,
        "tokenizer": tokenizer,
        "model_size": model_size or "base",
        "pooling": encoder_cfg.get("pooling", "mean"),
        "timestep": int(encoder_cfg.get("timestep", 1)),
        "batch_size": int(encoder_cfg.get("batch_size", 64)),
    }


def _embed_sequence(inputs: List[str], assets: dict, device: str) -> np.ndarray:
    if not inputs:
        return np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32)
    embeddings = []
    for start in range(0, len(inputs), assets["batch_size"]):
        batch = inputs[start:start + assets["batch_size"]]
        encoded = assets["tokenizer"].batch_encode(batch)
        input_ids = torch.tensor(encoded["input_ids"], device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=device)
        timesteps = torch.full((input_ids.size(0),), int(assets["timestep"]), device=device, dtype=torch.long)
        with torch.no_grad():
            pooled = assets["backbone"].get_pooled_output(
                input_ids=input_ids,
                timesteps=timesteps,
                attention_mask=attention_mask,
                pooling=assets["pooling"],
            )
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _embed_graph(smiles_list: List[str], assets: dict, device: str) -> Tuple[np.ndarray, List[int]]:
    if not smiles_list:
        return np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32), []
    valid_indices = []
    graph_batches = []
    for idx, smi in enumerate(smiles_list):
        try:
            data = assets["tokenizer"].encode(smi)
            graph_batches.append(data)
            valid_indices.append(idx)
        except Exception:
            continue
    embeddings = []
    for start in range(0, len(graph_batches), assets["batch_size"]):
        batch = graph_batches[start:start + assets["batch_size"]]
        if not batch:
            continue
        X = np.stack([b["X"] for b in batch])
        E = np.stack([b["E"] for b in batch])
        M = np.stack([b["M"] for b in batch])
        X_t = torch.tensor(X, device=device)
        E_t = torch.tensor(E, device=device)
        M_t = torch.tensor(M, device=device)
        timesteps = torch.full((X_t.size(0),), int(assets["timestep"]), device=device, dtype=torch.long)
        with torch.no_grad():
            pooled = assets["backbone"].get_node_embeddings(X_t, E_t, timesteps, M_t, pooling=assets["pooling"])
        embeddings.append(pooled.cpu().numpy())
    if embeddings:
        return np.concatenate(embeddings, axis=0), valid_indices
    return np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32), valid_indices


def _load_view_assets(config: dict, view: str, device: str) -> dict:
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


def _sanitize_sequence_inputs(inputs: List[str], tokenizer) -> Tuple[List[str], List[int]]:
    valid_inputs = []
    valid_indices = []
    for idx, text in enumerate(inputs):
        try:
            tokenizer.encode(text, add_special_tokens=True, padding=True, return_attention_mask=True)
            valid_inputs.append(text)
            valid_indices.append(idx)
        except Exception:
            continue
    return valid_inputs, valid_indices


def _embed_candidates(view: str, smiles_list: List[str], assets: dict, device: str) -> Tuple[np.ndarray, List[int]]:
    if view == "graph":
        return _embed_graph(smiles_list, assets, device)

    if view == "selfies":
        seq_inputs = []
        seq_indices = []
        for idx, smi in enumerate(smiles_list):
            converted = smiles_to_selfies(smi)
            if not converted:
                continue
            seq_inputs.append(converted)
            seq_indices.append(idx)
        seq_inputs, filtered_local_indices = _sanitize_sequence_inputs(seq_inputs, assets["tokenizer"])
        filtered_indices = [seq_indices[i] for i in filtered_local_indices]
        embeddings = _embed_sequence(seq_inputs, assets, device) if seq_inputs else np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32)
        return embeddings, filtered_indices

    seq_inputs, valid_indices = _sanitize_sequence_inputs(smiles_list, assets["tokenizer"])
    embeddings = _embed_sequence(seq_inputs, assets, device) if seq_inputs else np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32)
    return embeddings, valid_indices


def _load_d2_embeddings(results_dir: Path, view: str) -> np.ndarray:
    direct_path = results_dir / f"embeddings_{view}_d2.npy"
    if direct_path.exists():
        return np.load(direct_path)
    if view == "smiles":
        legacy_path = results_dir / "embeddings_d2.npy"
        if legacy_path.exists():
            return np.load(legacy_path)
    raise FileNotFoundError(
        f"D2 embeddings not found for view={view}. Expected {direct_path} "
        f"(or legacy embeddings_d2.npy for smiles). Run F1 with this view enabled first."
    )


def _default_property_model_path(results_dir: Path, property_name: str, view: str) -> Path:
    model_dir = results_dir / "step3_property"
    if view == "smiles":
        return model_dir / f"{property_name}_mlp.pt"
    return model_dir / f"{property_name}_{view}_mlp.pt"


class _PropertyMLP(torch.nn.Module):
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


class _TorchPropertyPredictor:
    def __init__(self, checkpoint: dict):
        if checkpoint.get("format") != "mvf_torch_mlp":
            raise ValueError("Unsupported torch property model format.")
        self.mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
        self.scale = np.where(np.abs(self.scale) < 1e-12, 1.0, self.scale).astype(np.float32, copy=False)
        self.model = _PropertyMLP(
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


def _load_property_model(model_path: Path):
    if model_path.suffix in {".pt", ".pth"}:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and checkpoint.get("format") == "mvf_torch_mlp":
            return _TorchPropertyPredictor(checkpoint)
        raise ValueError(f"Unsupported torch property model checkpoint: {model_path}")

    if joblib is not None:
        return joblib.load(model_path)
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _check_validity(smiles: str) -> bool:
    if Chem is None:
        return True
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _count_stars(smiles: str) -> int:
    return smiles.count("*") if isinstance(smiles, str) else 0


def _load_training_smiles(path: Path) -> set:
    if not path.exists():
        return set()
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    col = "p_smiles" if "p_smiles" in df.columns else "SMILES" if "SMILES" in df.columns else None
    if col is None:
        return set()
    return set(df[col].astype(str).tolist())


def _achievement_rates(preds: np.ndarray, target: float) -> dict:
    if preds is None or len(preds) == 0:
        return {
            "achievement_5p": 0.0,
            "achievement_10p": 0.0,
            "achievement_15p": 0.0,
            "achievement_20p": 0.0,
        }
    denom = max(abs(float(target)), 1e-9)
    return {
        "achievement_5p": float(np.mean(np.abs(preds - target) <= 0.05 * denom)),
        "achievement_10p": float(np.mean(np.abs(preds - target) <= 0.10 * denom)),
        "achievement_15p": float(np.mean(np.abs(preds - target) <= 0.15 * denom)),
        "achievement_20p": float(np.mean(np.abs(preds - target) <= 0.20 * denom)),
    }


def main(args):
    config = load_config(args.config)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, results_dir / "config_used.yaml")

    candidates_path = _resolve_path(args.candidates_csv)
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {candidates_path}")

    candidates_df = pd.read_csv(candidates_path)
    smiles_col = args.smiles_column
    if smiles_col is None:
        for candidate in ["smiles", "p_smiles", "psmiles"]:
            if candidate in candidates_df.columns:
                smiles_col = candidate
                break
    if smiles_col is None or smiles_col not in candidates_df.columns:
        raise ValueError("Candidates CSV must include a SMILES column.")

    smiles_list = candidates_df[smiles_col].astype(str).tolist()
    structural_validity_mask = [
        _check_validity(smi) and _count_stars(smi) == 2
        for smi in smiles_list
    ]
    structurally_valid_smiles = [s for s, keep in zip(smiles_list, structural_validity_mask) if keep]

    encoder_view = _select_encoder_view(config, args.encoder_view)
    encoder_cfg = config.get(VIEW_SPECS[encoder_view]["encoder_key"], {})
    device = encoder_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assets = _load_view_assets(config=config, view=encoder_view, device=device)

    t0 = time.time()
    embeddings, kept_indices = _embed_candidates(
        view=encoder_view,
        smiles_list=structurally_valid_smiles,
        assets=assets,
        device=device,
    )
    scored_smiles = [structurally_valid_smiles[i] for i in kept_indices]
    embed_time = time.time() - t0

    alignment_model = None
    if args.use_alignment:
        view_dims = {encoder_view: embeddings.shape[1] if embeddings.size else assets["backbone"].hidden_size}
        alignment_model = _load_alignment_model(results_dir, view_dims, config, args.alignment_checkpoint)
        if alignment_model is None:
            raise FileNotFoundError("Alignment checkpoint not found for --use_alignment")
        device_proj = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = _project_embeddings(alignment_model, encoder_view, embeddings, device=device_proj)

    model_path = args.property_model_path
    if model_path is None:
        model_path = _default_property_model_path(results_dir, args.property, encoder_view)
    else:
        model_path = _resolve_path(model_path)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Property model not found: {model_path}")

    model = _load_property_model(Path(model_path))
    preds = model.predict(embeddings) if len(scored_smiles) else np.array([], dtype=np.float32)
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)

    target_value = float(args.target)
    epsilon = float(args.epsilon)
    hits = np.abs(preds - target_value) <= epsilon

    n_generated = len(smiles_list)
    n_valid = len(structurally_valid_smiles)
    n_scored = len(scored_smiles)
    n_hits = int(hits.sum()) if n_scored else 0
    success_rate = n_hits / n_valid if n_valid else 0.0

    validity = n_valid / n_generated if n_generated else 0.0
    uniqueness = len(set(structurally_valid_smiles)) / n_valid if n_valid else 0.0

    train_smiles_path = _resolve_path(config["paths"]["polymer_file"])
    training_set = _load_training_smiles(train_smiles_path)
    novelty = 0.0
    if n_valid:
        novelty = sum(1 for s in structurally_valid_smiles if s not in training_set) / n_valid

    d2_distance_scores = None
    rerank_metrics = {
        "rerank_applied": False,
    }

    if args.rerank_strategy == "d2_distance":
        d2_embeddings = _load_d2_embeddings(results_dir, encoder_view)
        if args.use_alignment and alignment_model is not None:
            device_proj = "cuda" if torch.cuda.is_available() else "cpu"
            d2_embeddings = _project_embeddings(alignment_model, encoder_view, d2_embeddings, device=device_proj)
        distances = knn_distances(embeddings, d2_embeddings, k=args.ood_k)
        d2_distance_scores = distances.mean(axis=1)
        order = np.argsort(d2_distance_scores)
        top_k = min(int(args.rerank_top_k), len(order))
        if top_k > 0:
            top_hits = hits[order[:top_k]]
            rerank_metrics = {
                "rerank_applied": True,
                "rerank_strategy": "d2_distance",
                "rerank_top_k": top_k,
                "rerank_hits": int(top_hits.sum()),
                "rerank_success_rate": round(float(top_hits.sum()) / top_k, 4),
            }
        else:
            rerank_metrics = {"rerank_applied": False}

    metrics_row = {
        "method": "Multi_View_Foundation",
        "representation": _view_to_representation(encoder_view),
        "model_size": assets["model_size"],
        "property": args.property,
        "target_value": target_value,
        "epsilon": epsilon,
        "n_generated": n_generated,
        "n_valid": n_valid,
        "n_hits": n_hits,
        "success_rate": round(success_rate, 4),
        "validity": round(validity, 4),
        "validity_two_stars": round(validity, 4),
        "uniqueness": round(uniqueness, 4),
        "novelty": round(novelty, 4),
        "avg_diversity": None,
        **_achievement_rates(preds, target_value),
        "sampling_time_sec": round(embed_time, 2),
        "valid_per_compute": round(n_valid / max(embed_time, 1e-9), 4) if n_valid else 0.0,
        **rerank_metrics,
    }

    if rerank_metrics.get("rerank_applied"):
        metrics_row["valid_per_compute_rerank"] = metrics_row["valid_per_compute"]
    else:
        metrics_row["rerank_strategy"] = args.rerank_strategy

    pd.DataFrame([metrics_row]).to_csv(results_dir / "metrics_inverse.csv", index=False)

    out_dir = results_dir / "step5_foundation_inverse"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_rows = pd.DataFrame({
        "smiles": scored_smiles,
        "prediction": preds,
        "abs_error": np.abs(preds - target_value) if len(preds) else [],
        "hit": hits,
        "d2_distance": d2_distance_scores if d2_distance_scores is not None else None,
    })
    valid_rows.to_csv(out_dir / "candidate_scores.csv", index=False)

    with open(out_dir / "run_meta.json", "w") as f:
        json.dump({
            "candidates_csv": str(candidates_path),
            "smiles_column": smiles_col,
            "encoder_view": encoder_view,
            "property": args.property,
            "target_value": target_value,
            "epsilon": epsilon,
            "rerank_strategy": args.rerank_strategy,
            "rerank_top_k": int(args.rerank_top_k),
            "use_alignment": bool(args.use_alignment),
            "n_generated": int(n_generated),
            "n_structurally_valid": int(n_valid),
            "n_scored": int(n_scored),
        }, f, indent=2)

    print(f"Saved metrics_inverse.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--encoder_view", type=str, default=None, choices=list(SUPPORTED_VIEWS))
    parser.add_argument("--candidates_csv", type=str, required=True)
    parser.add_argument("--smiles_column", type=str, default=None)
    parser.add_argument("--property", type=str, required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--epsilon", type=float, default=30.0)
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument("--rerank_strategy", type=str, default="d2_distance")
    parser.add_argument("--rerank_top_k", type=int, default=100)
    parser.add_argument("--ood_k", type=int, default=5)
    parser.add_argument("--use_alignment", action="store_true")
    parser.add_argument("--alignment_checkpoint", type=str, default=None)
    main(parser.parse_args())
