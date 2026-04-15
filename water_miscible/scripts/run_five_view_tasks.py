#!/usr/bin/env python
"""Train five-view chi regression and water-miscible classification heads."""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import math
import os
import random
import resource
import shlex
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/water_miscible_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:  # pragma: no cover
    import optuna
except Exception:  # pragma: no cover
    optuna = None

try:  # pragma: no cover
    import selfies as sf
except Exception:  # pragma: no cover
    sf = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
REPRESENTATION_ORDER = ["smiles", "smiles_bpe", "selfies", "group_selfies", "graph"]
MODEL_SIZES = {"small", "medium", "large", "xl"}
TASK_KINDS = {"chi_regression": "regression", "water_classification": "classification"}
TASK_ALIASES = {
    "chi": "chi_regression",
    "chi_regression": "chi_regression",
    "water": "water_classification",
    "water_miscible": "water_classification",
    "water_classification": "water_classification",
}
REPRESENTATION_LABELS = {
    "smiles": "SMILES",
    "smiles_bpe": "SMILES_BPE",
    "selfies": "SELFIES",
    "group_selfies": "Group_SELFIES",
    "graph": "Graph",
}
PLOT_VIEW_LABELS = {
    "smiles": "SMILES",
    "smiles_bpe": "BPE",
    "selfies": "SELFIES",
    "group_selfies": "G-SELFIES",
    "graph": "Graph",
}
REPRESENTATION_COLORS = {
    "smiles": "#4E79A7",
    "smiles_bpe": "#76B7B2",
    "selfies": "#F28E2B",
    "group_selfies": "#59A14F",
    "graph": "#9C755F",
}
METRIC_LABELS = {
    "r2": "R2",
    "rmse": "RMSE",
    "mae": "MAE",
    "poly_nrmse": "NRMSE",
    "balanced_accuracy": "BA",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "brier": "Brier",
    "val_r2": "R2",
    "val_poly_nrmse": "NRMSE",
}
FINAL_TRAINING_POLICY = "train_plus_val_fixed_epoch_budget_from_cv_no_final_validation"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, sort_keys=False)


def apply_model_size_override(config: Dict[str, Any], model_size: Optional[str]) -> Dict[str, Any]:
    if model_size is None:
        return config
    size = str(model_size).strip().lower()
    if size not in MODEL_SIZES:
        raise ValueError(f"model_size must be one of {sorted(MODEL_SIZES)}, got {model_size!r}.")

    cfg = copy.deepcopy(config)
    cfg.setdefault("paths", {})["results_dir"] = f"water_miscible/results_{size}"
    backbones = cfg.setdefault("backbones", {})
    backbones["model_size"] = size
    for view, view_cfg in list(backbones.items()):
        if not isinstance(view_cfg, dict) or "method_dir" not in view_cfg:
            continue
        method_dir = str(view_cfg["method_dir"]).rstrip("/")
        view_cfg["model_size"] = size
        view_cfg["results_dir"] = f"{method_dir}/results_{size}"
        checkpoint_name = "graph_backbone_best.pt" if view == "graph" else "backbone_best.pt"
        view_cfg["checkpoint_path"] = f"{method_dir}/results_{size}/step1_backbone/checkpoints/{checkpoint_name}"
    return cfg


def normalize_task_names(raw: Optional[str]) -> List[str]:
    if raw is None or str(raw).strip() == "":
        return list(TASK_KINDS.keys())
    tasks = []
    for item in str(raw).split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in TASK_ALIASES:
            raise ValueError(f"Unknown task '{item}'. Use chi_regression or water_classification.")
        task = TASK_ALIASES[key]
        if task not in tasks:
            tasks.append(task)
    return tasks or list(TASK_KINDS.keys())


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)


def repo_relative(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def command_line() -> str:
    return " ".join(shlex.quote(arg) for arg in sys.argv)


def runtime_resources() -> Dict[str, Any]:
    resources: Dict[str, Any] = {
        "max_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
    }
    if torch.cuda.is_available():
        resources["cuda_max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)
        resources["cuda_max_memory_reserved_mb"] = float(torch.cuda.max_memory_reserved()) / (1024.0 ** 2)
    return resources


def make_run_context(args: argparse.Namespace, config_path: Path, results_dir: Path, tasks: Sequence[str], views: Sequence[str]) -> Dict[str, Any]:
    return {
        "command": command_line(),
        "config_path": repo_relative(config_path),
        "results_dir": repo_relative(results_dir),
        "model_size": args.model_size,
        "tasks": list(tasks),
        "views": list(views),
        "cache_tag": args.cache_tag,
        "fresh_embeddings": bool(args.fresh_embeddings),
        "precompute_embeddings_only": bool(args.precompute_embeddings_only),
        "skip_postprocess": bool(args.skip_postprocess),
        "postprocess_only": bool(args.postprocess_only),
        "strict_postprocess": bool(getattr(args, "strict_postprocess", False)),
        "tune_override": True if args.tune else False if args.no_tune else None,
        "max_rows": args.max_rows,
    }


def load_module(module_name: str, path: str | Path):
    module_path = Path(path).resolve()
    import_name = None
    try:
        rel = module_path.relative_to(REPO_ROOT.resolve())
        if rel.suffix == ".py":
            parts = list(rel.with_suffix("").parts)
            if parts and all(part.isidentifier() for part in parts):
                import_name = ".".join(parts)
    except Exception:
        import_name = None
    if import_name:
        try:
            return importlib.import_module(import_name)
        except Exception:
            pass
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_backbone_checkpoint(backbone: nn.Module, checkpoint_path: Path, device: str, prefix: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, Mapping):
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint payload at {checkpoint_path}")
    cleaned = {str(k).replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    scoped = {k[len(prefix):]: v for k, v in cleaned.items() if prefix and k.startswith(prefix)}
    final_state = scoped if scoped else cleaned
    incompatible = backbone.load_state_dict(final_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"Warning: checkpoint mismatch for {checkpoint_path}: "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
        )


def stable_seed(seed: int, *parts: object) -> int:
    text = "::".join([str(seed), *[str(p) for p in parts]])
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 2_147_483_647
    return int(value or seed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_config(config: Dict[str, Any]) -> str:
    requested = str(config.get("backbones", {}).get("device", "auto")).strip().lower() or "auto"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


@dataclass
class ViewInfo:
    view: str
    view_type: str
    model_size: str
    hidden_size: int
    backbone_num_layers: int
    pooling: str
    timestep: int
    batch_size: int
    checkpoint_path: Path


@dataclass
class ViewAssets:
    view: str
    view_type: str
    model_size: str
    tokenizer: Any
    backbone: nn.Module
    hidden_size: int
    pooling: str
    timestep: int
    batch_size: int
    checkpoint_path: Path


def _method_model_config(method_dir: Path, method_cfg: Dict[str, Any], model_size: str, model_type: str) -> Dict[str, Any]:
    scales_mod = load_module(f"scales_{method_dir.name}", method_dir / "src" / "utils" / "model_scales.py")
    return scales_mod.get_model_config(model_size, method_cfg, model_type=model_type)


def get_view_info(config: Dict[str, Any], view: str) -> ViewInfo:
    backbones_cfg = config.get("backbones", {})
    view_cfg = backbones_cfg.get(view, {})
    if not view_cfg:
        raise ValueError(f"Missing backbones.{view} config.")
    model_size = str(view_cfg.get("model_size", backbones_cfg.get("model_size", "small")))
    pooling = str(view_cfg.get("pooling", backbones_cfg.get("pooling", "mean")))
    timestep = int(view_cfg.get("timestep", backbones_cfg.get("timestep", 1)))
    method_dir = resolve_path(view_cfg["method_dir"])
    method_cfg = load_yaml(resolve_path(view_cfg.get("config_path", method_dir / "configs" / "config.yaml")))
    checkpoint_path = resolve_path(view_cfg["checkpoint_path"])
    if view == "graph":
        model_type = "graph"
        view_type = "graph"
        batch_size = int(view_cfg.get("batch_size", backbones_cfg.get("graph_embedding_batch_size", 64)))
    else:
        model_type = "sequence"
        view_type = "sequence"
        batch_size = int(view_cfg.get("batch_size", backbones_cfg.get("embedding_batch_size", 256)))
    backbone_config = _method_model_config(method_dir, method_cfg, model_size, model_type)
    return ViewInfo(
        view=view,
        view_type=view_type,
        model_size=model_size,
        hidden_size=int(backbone_config["hidden_size"]),
        backbone_num_layers=int(backbone_config.get("num_layers", 0)),
        pooling=pooling,
        timestep=timestep,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
    )


def load_view_assets(config: Dict[str, Any], view: str, device: str) -> ViewAssets:
    backbones_cfg = config.get("backbones", {})
    view_cfg = backbones_cfg.get(view, {})
    if not view_cfg:
        raise ValueError(f"Missing backbones.{view} config.")
    model_size = str(view_cfg.get("model_size", backbones_cfg.get("model_size", "small")))
    pooling = str(view_cfg.get("pooling", backbones_cfg.get("pooling", "mean")))
    timestep = int(view_cfg.get("timestep", backbones_cfg.get("timestep", 1)))
    method_dir = resolve_path(view_cfg["method_dir"])
    method_cfg = load_yaml(resolve_path(view_cfg.get("config_path", method_dir / "configs" / "config.yaml")))
    checkpoint_path = resolve_path(view_cfg["checkpoint_path"])

    if view == "graph":
        batch_size = int(view_cfg.get("batch_size", backbones_cfg.get("graph_embedding_batch_size", 64)))
        tokenizer_mod = load_module("graph_tokenizer_water", method_dir / "src" / "data" / "graph_tokenizer.py")
        backbone_mod = load_module("graph_backbone_water", method_dir / "src" / "model" / "graph_backbone.py")
        graph_config = json.loads(resolve_path(view_cfg["graph_config_path"]).read_text(encoding="utf-8"))
        tokenizer = tokenizer_mod.GraphTokenizer.load(str(resolve_path(view_cfg["tokenizer_path"])))
        backbone_config = _method_model_config(method_dir, method_cfg, model_size, "graph")
        backbone = backbone_mod.GraphDiffusionBackbone(
            atom_vocab_size=graph_config["atom_vocab_size"],
            edge_vocab_size=graph_config["edge_vocab_size"],
            Nmax=graph_config["Nmax"],
            hidden_size=backbone_config["hidden_size"],
            num_layers=backbone_config["num_layers"],
            num_heads=backbone_config["num_heads"],
            ffn_hidden_size=backbone_config["ffn_hidden_size"],
            dropout=backbone_config.get("dropout", 0.1),
            num_diffusion_steps=method_cfg.get("diffusion", {}).get("num_steps", 50),
        )
        load_backbone_checkpoint(backbone, checkpoint_path, device=device, prefix="backbone.")
        backbone.to(device).eval()
        return ViewAssets(view, "graph", model_size, tokenizer, backbone, int(backbone.hidden_size), pooling, timestep, batch_size, checkpoint_path)

    batch_size = int(view_cfg.get("batch_size", backbones_cfg.get("embedding_batch_size", 256)))
    if view == "selfies":
        tokenizer_path = resolve_path(view_cfg["tokenizer_path"])
        tokenizer_mod_path = method_dir / "src" / "data" / "selfies_tokenizer.py"
        tokenizer_class = "SelfiesTokenizer"
    elif view == "group_selfies":
        tokenizer_path = resolve_path(view_cfg["tokenizer_path"])
        tokenizer_mod_path = method_dir / "src" / "data" / "tokenizer.py"
        tokenizer_class = "GroupSELFIESTokenizer"
    else:
        tokenizer_path = resolve_path(view_cfg["tokenizer_path"])
        tokenizer_mod_path = method_dir / "src" / "data" / "tokenizer.py"
        tokenizer_class = "PSmilesTokenizer"

    tokenizer_mod = load_module(f"tokenizer_{view}_water", tokenizer_mod_path)
    backbone_mod = load_module(f"backbone_{view}_water", method_dir / "src" / "model" / "backbone.py")
    tokenizer = getattr(tokenizer_mod, tokenizer_class).load(str(tokenizer_path))
    backbone_config = _method_model_config(method_dir, method_cfg, model_size, "sequence")
    backbone = backbone_mod.DiffusionBackbone(
        vocab_size=tokenizer.vocab_size,
        hidden_size=backbone_config["hidden_size"],
        num_layers=backbone_config["num_layers"],
        num_heads=backbone_config["num_heads"],
        ffn_hidden_size=backbone_config["ffn_hidden_size"],
        max_position_embeddings=backbone_config.get("max_position_embeddings", 256),
        num_diffusion_steps=method_cfg.get("diffusion", {}).get("num_steps", 50),
        dropout=backbone_config.get("dropout", 0.1),
        pad_token_id=tokenizer.pad_token_id,
    )
    load_backbone_checkpoint(backbone, checkpoint_path, device=device, prefix="backbone.")
    backbone.to(device).eval()
    return ViewAssets(view, "sequence", model_size, tokenizer, backbone, int(backbone.hidden_size), pooling, timestep, batch_size, checkpoint_path)


def smiles_to_selfies_text(smiles: str) -> Optional[str]:
    if sf is None:
        raise ImportError("selfies is required for the SELFIES view.")
    text = str(smiles).strip()
    if not text:
        return None
    try:
        return sf.encoder(text.replace("*", "[I+3]"))
    except Exception:
        return None


def transform_text_for_view(smiles: str, view: str) -> Optional[str]:
    if view == "selfies":
        return smiles_to_selfies_text(smiles)
    return str(smiles).strip() or None


def encode_sequence_text(text: str, assets: ViewAssets) -> Optional[Dict[str, np.ndarray]]:
    try:
        encoded = assets.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
        )
    except Exception:
        return None
    return {
        "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
    }


@torch.no_grad()
def embed_sequence_raw_inputs(raw_inputs: Dict[str, np.ndarray], assets: ViewAssets, device: str) -> np.ndarray:
    input_ids = np.asarray(raw_inputs["input_ids"], dtype=np.int64)
    attention_mask = np.asarray(raw_inputs["attention_mask"], dtype=np.int64)
    if input_ids.shape[0] == 0:
        return np.zeros((0, assets.hidden_size), dtype=np.float32)
    embeddings: List[np.ndarray] = []
    for start in range(0, input_ids.shape[0], assets.batch_size):
        ids = torch.tensor(input_ids[start:start + assets.batch_size], dtype=torch.long, device=device)
        mask = torch.tensor(attention_mask[start:start + assets.batch_size], dtype=torch.long, device=device)
        timesteps = torch.full((ids.shape[0],), assets.timestep, dtype=torch.long, device=device)
        pooled = assets.backbone.get_pooled_output(
            input_ids=ids,
            timesteps=timesteps,
            attention_mask=mask,
            pooling=assets.pooling,
        )
        embeddings.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, assets.hidden_size), dtype=np.float32)


@torch.no_grad()
def embed_sequence_texts(texts: Sequence[str], assets: ViewAssets, device: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, assets.hidden_size), dtype=np.float32)
    encoded = assets.tokenizer.batch_encode(list(texts))
    raw_inputs = {
        "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
    }
    return embed_sequence_raw_inputs(raw_inputs, assets, device)


@torch.no_grad()
def embed_graph_raw_inputs(raw_inputs: Dict[str, np.ndarray], assets: ViewAssets, device: str) -> np.ndarray:
    X = np.asarray(raw_inputs["X"], dtype=np.int64)
    E = np.asarray(raw_inputs["E"], dtype=np.int64)
    M = np.asarray(raw_inputs["M"], dtype=np.float32)
    if X.shape[0] == 0:
        return np.zeros((0, assets.hidden_size), dtype=np.float32)
    embeddings: List[np.ndarray] = []
    for start in range(0, X.shape[0], assets.batch_size):
        x_t = torch.tensor(X[start:start + assets.batch_size], dtype=torch.long, device=device)
        e_t = torch.tensor(E[start:start + assets.batch_size], dtype=torch.long, device=device)
        m_t = torch.tensor(M[start:start + assets.batch_size], dtype=torch.float32, device=device)
        timesteps = torch.full((x_t.shape[0],), assets.timestep, dtype=torch.long, device=device)
        pooled = assets.backbone.get_node_embeddings(x_t, e_t, timesteps, m_t, pooling=assets.pooling)
        embeddings.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, assets.hidden_size), dtype=np.float32)


@torch.no_grad()
def embed_graph_smiles(smiles_values: Sequence[str], assets: ViewAssets, device: str) -> Tuple[List[str], np.ndarray]:
    valid_smiles: List[str] = []
    rows: List[Dict[str, np.ndarray]] = []
    for smi in smiles_values:
        try:
            encoded = assets.tokenizer.encode(str(smi))
        except Exception:
            continue
        valid_smiles.append(str(smi))
        rows.append(
            {
                "X": np.asarray(encoded["X"], dtype=np.int64),
                "E": np.asarray(encoded["E"], dtype=np.int64),
                "M": np.asarray(encoded["M"], dtype=np.float32),
            }
        )
    if not rows:
        return valid_smiles, np.zeros((0, assets.hidden_size), dtype=np.float32)
    X = np.stack([r["X"] for r in rows], axis=0)
    E = np.stack([r["E"] for r in rows], axis=0)
    M = np.stack([r["M"] for r in rows], axis=0)
    return valid_smiles, embed_graph_raw_inputs({"X": X, "E": E, "M": M}, assets, device)


def view_cache_metadata(view: str, view_type: str, model_size: str, pooling: str, timestep: int, hidden_size: int, backbone_num_layers: int, checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    stat = checkpoint.stat() if checkpoint.exists() else None
    return {
        "format_version": 2,
        "view": view,
        "view_type": view_type,
        "model_size": model_size,
        "pooling": pooling,
        "timestep": int(timestep),
        "hidden_size": int(hidden_size),
        "backbone_num_layers": int(backbone_num_layers),
        "checkpoint_path": repo_relative(checkpoint),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns) if stat is not None else None,
        "checkpoint_size": int(stat.st_size) if stat is not None else None,
    }


def embedding_cache_metadata(assets: ViewAssets) -> Dict[str, Any]:
    return view_cache_metadata(
        view=assets.view,
        view_type=assets.view_type,
        model_size=assets.model_size,
        pooling=assets.pooling,
        timestep=assets.timestep,
        hidden_size=assets.hidden_size,
        backbone_num_layers=get_backbone_num_layers(assets.backbone),
        checkpoint_path=assets.checkpoint_path,
    )


def embedding_cache_metadata_from_info(info: ViewInfo) -> Dict[str, Any]:
    return view_cache_metadata(
        view=info.view,
        view_type=info.view_type,
        model_size=info.model_size,
        pooling=info.pooling,
        timestep=info.timestep,
        hidden_size=info.hidden_size,
        backbone_num_layers=info.backbone_num_layers,
        checkpoint_path=info.checkpoint_path,
    )


def parse_cache_metadata(payload: np.lib.npyio.NpzFile) -> Optional[Dict[str, Any]]:
    if "metadata_json" not in payload.files:
        return None
    try:
        raw = payload["metadata_json"].tolist()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))
    except Exception:
        return None


def cache_metadata_matches(stored: Optional[Mapping[str, Any]], expected: Mapping[str, Any]) -> bool:
    if not stored:
        return False
    for key, expected_value in expected.items():
        if stored.get(key) != expected_value:
            return False
    return True


def build_or_load_embeddings(
    *,
    smiles_values: Sequence[str],
    assets: ViewAssets,
    cache_path: Path,
    device: str,
    force_rebuild: bool = False,
) -> Dict[str, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted({str(smi).strip() for smi in smiles_values if str(smi).strip()})
    requested = set(unique)
    metadata = embedding_cache_metadata(assets)
    cached = load_embedding_cache(cache_path, requested, expected_metadata=metadata) if not force_rebuild else None
    if cached is not None:
        return cached
    if cache_path.exists() and not force_rebuild:
        print(f"Rebuilding stale or incomplete embedding cache for view={assets.view}: {cache_path}")

    if assets.view == "graph":
        valid_smiles, embeddings = embed_graph_smiles(unique, assets, device)
    else:
        valid_smiles = []
        texts = []
        for smi in unique:
            transformed = transform_text_for_view(smi, assets.view)
            if transformed:
                valid_smiles.append(smi)
                texts.append(transformed)
        embeddings = embed_sequence_texts(texts, assets, device)
    np.savez_compressed(
        cache_path,
        smiles=np.asarray(valid_smiles, dtype=object),
        embeddings=embeddings,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=object),
    )
    return {smi: embeddings[i] for i, smi in enumerate(valid_smiles)}


def load_embedding_cache(
    cache_path: Path,
    requested: set[str],
    expected_metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    if not cache_path.exists():
        return None
    payload = np.load(cache_path, allow_pickle=True)
    if expected_metadata is not None and not cache_metadata_matches(parse_cache_metadata(payload), expected_metadata):
        print(f"Ignoring stale embedding cache metadata: {cache_path}")
        return None
    smiles = [str(x) for x in payload["smiles"].tolist()]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if len(smiles) == embeddings.shape[0] and requested.issubset(set(smiles)):
        return {smi: embeddings[i] for i, smi in enumerate(smiles)}
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]
    return out


def fill_polymer_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Polymer" not in out.columns:
        out["Polymer"] = ""
    out["Polymer"] = out["Polymer"].where(out["Polymer"].notna(), "").astype(str).str.strip()
    missing = out["Polymer"].eq("") | out["Polymer"].str.lower().isin({"nan", "none", "null"})
    out.loc[missing, "Polymer"] = out.loc[missing, "SMILES"].astype(str)
    return out


def add_polymer_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = fill_polymer_names(df).reset_index(drop=True).copy()
    out["split_group_key"] = out["SMILES"].astype(str).str.strip()
    group_order = sorted(out["split_group_key"].unique())
    group_to_id = {group: idx for idx, group in enumerate(group_order)}
    # Keep the existing column name because downstream loss/metrics use it,
    # but derive it from SMILES to avoid relying on incomplete Polymer names.
    out["polymer_id"] = out["split_group_key"].map(group_to_id).astype(int)
    out["row_id"] = np.arange(len(out), dtype=np.int64)
    return out


def load_chi_dataset(config: Dict[str, Any], max_rows: Optional[int] = None) -> pd.DataFrame:
    path = resolve_path(config["paths"]["chi_dataset"])
    df = normalize_columns(pd.read_csv(path, nrows=max_rows))
    required = {"SMILES", "temperature", "phi", "chi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Chi dataset missing required columns: {sorted(missing)}")
    if "water_miscible" not in df.columns:
        df["water_miscible"] = 0
    df = df.copy()
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["phi"] = pd.to_numeric(df["phi"], errors="coerce")
    df["chi"] = pd.to_numeric(df["chi"], errors="coerce")
    df["water_miscible"] = pd.to_numeric(df["water_miscible"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["SMILES", "temperature", "phi", "chi"]).reset_index(drop=True)
    return add_polymer_ids(df)


def class_balanced_debug_sample(df: pd.DataFrame, max_rows: int, label_col: str, seed: int) -> pd.DataFrame:
    n = max(1, int(max_rows))
    if len(df) <= n or label_col not in df.columns:
        return df.copy()
    labels = pd.to_numeric(df[label_col], errors="coerce")
    classes = sorted(labels.dropna().astype(int).unique().tolist())
    if len(classes) < 2:
        return df.head(n).copy()

    label_int = labels.fillna(-1_000_000_000).astype(int)
    base = n // len(classes)
    remainder = n % len(classes)
    selected_parts = []
    selected_indices: set[int] = set()
    for pos, cls in enumerate(classes):
        take = base + (1 if pos < remainder else 0)
        cls_pool = df.loc[label_int == int(cls)]
        take = min(int(take), len(cls_pool))
        cls_rows = cls_pool.sample(n=take, random_state=int(seed) + 1009 * (pos + 1)) if take > 0 else cls_pool.iloc[0:0]
        selected_parts.append(cls_rows)
        selected_indices.update(int(idx) for idx in cls_rows.index)

    selected = pd.concat(selected_parts, axis=0) if selected_parts else df.iloc[0:0].copy()
    if len(selected) < n:
        filler_pool = df.loc[~df.index.isin(selected_indices)]
        filler_n = min(n - len(selected), len(filler_pool))
        filler = filler_pool.sample(n=filler_n, random_state=int(seed) + 7919) if filler_n > 0 else filler_pool.iloc[0:0]
        selected = pd.concat([selected, filler], axis=0)
    return selected.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)


def load_water_dataset(config: Dict[str, Any], max_rows: Optional[int] = None) -> pd.DataFrame:
    paths = [
        resolve_path(config["paths"]["water_miscible_dataset"]),
        resolve_path(config["paths"]["water_immiscible_dataset"]),
    ]
    frames = [normalize_columns(pd.read_csv(path)) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    required = {"SMILES", "water_miscible"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Water dataset missing required columns: {sorted(missing)}")
    df = df.copy()
    df["water_miscible"] = pd.to_numeric(df["water_miscible"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["SMILES", "water_miscible"]).reset_index(drop=True)
    if max_rows is not None:
        df = class_balanced_debug_sample(df, int(max_rows), "water_miscible", seed=int(config.get("data", {}).get("random_seed", 42)))
    defaults = config.get("data", {})
    df["temperature"] = float(defaults.get("default_temperature", 293.15))
    df["phi"] = float(defaults.get("default_phi", 0.2))
    df["chi"] = float(defaults.get("default_chi", 0.0))
    return add_polymer_ids(df)


def regression_strata(values: Sequence[float], max_bins: int = 5) -> Optional[np.ndarray]:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if arr.size < 4 or len(np.unique(arr[finite])) < 2:
        return None
    for bins in range(min(int(max_bins), int(finite.sum() // 2), len(np.unique(arr[finite]))), 1, -1):
        try:
            labels = pd.qcut(arr, q=bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        labels = np.asarray(labels, dtype=float)
        if np.any(~np.isfinite(labels)):
            continue
        labels_int = labels.astype(int)
        counts = np.bincount(labels_int)
        if len(counts) >= 2 and int(counts.min()) >= 2:
            return labels_int
    return None


def stratification_labels(df: pd.DataFrame, target: str) -> Optional[np.ndarray]:
    target = str(target).strip().lower()
    if target == "chi" and "chi" in df.columns:
        return regression_strata(df["chi"].to_numpy(dtype=float))
    labels = df["water_miscible"].to_numpy(dtype=int)
    counts = np.bincount(labels)
    return labels if len(np.unique(labels)) > 1 and int(counts.min()) >= 2 else None


def split_units_for_stratification(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("polymer_id", as_index=False)
        .agg(water_miscible=("water_miscible", lambda x: int(round(float(np.mean(x))))), chi=("chi", "mean"))
        .reset_index(drop=True)
    )


def safe_train_test_split(*arrays, stratify=None, **kwargs):
    try:
        return train_test_split(*arrays, stratify=stratify, **kwargs)
    except ValueError:
        if stratify is None:
            raise
        return train_test_split(*arrays, stratify=None, **kwargs)


def make_splits(
    df: pd.DataFrame,
    split_mode: str,
    test_ratio: float,
    cv_folds: int,
    seed: int,
    stratify_target: str = "water_miscible",
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    split_mode = "smiles" if split_mode == "polymer" else split_mode
    dev_ratio = 1.0 - float(test_ratio)
    val_ratio = dev_ratio / float(max(2, int(cv_folds)))
    val_fraction_of_dev = val_ratio / dev_ratio
    out["split"] = "train"

    if split_mode == "smiles":
        units = split_units_for_stratification(out)
        unit_ids = units["polymer_id"].to_numpy()
        if len(unit_ids) < 3:
            return make_splits(df, split_mode="random", test_ratio=test_ratio, cv_folds=cv_folds, seed=seed, stratify_target=stratify_target)
        stratify = stratification_labels(units, stratify_target)
        train_units, test_units = safe_train_test_split(unit_ids, test_size=test_ratio, random_state=seed, shuffle=True, stratify=stratify)
        if len(train_units) < 2:
            return make_splits(df, split_mode="random", test_ratio=test_ratio, cv_folds=cv_folds, seed=seed, stratify_target=stratify_target)
        train_units_df = units.set_index("polymer_id").loc[train_units].reset_index()
        stratify_dev = stratification_labels(train_units_df, stratify_target)
        train_units, val_units = safe_train_test_split(
            train_units,
            test_size=val_fraction_of_dev,
            random_state=seed + 17,
            shuffle=True,
            stratify=stratify_dev,
        )
        out.loc[out["polymer_id"].isin(test_units), "split"] = "test"
        out.loc[out["polymer_id"].isin(val_units), "split"] = "val"
        out.loc[out["polymer_id"].isin(train_units), "split"] = "train"
        return out

    if split_mode != "random":
        raise ValueError("split_mode must be 'smiles' or 'random'")
    if len(out) < 3:
        out["split"] = "train"
        if len(out) >= 1:
            out.loc[out.index[-1], "split"] = "test"
        return out
    indices = np.arange(len(out))
    stratify = stratification_labels(out, stratify_target)
    dev_idx, test_idx = safe_train_test_split(indices, test_size=test_ratio, random_state=seed, shuffle=True, stratify=stratify)
    stratify_dev = stratification_labels(out.iloc[dev_idx], stratify_target)
    train_idx, val_idx = safe_train_test_split(dev_idx, test_size=val_fraction_of_dev, random_state=seed + 17, shuffle=True, stratify=stratify_dev)
    out.loc[test_idx, "split"] = "test"
    out.loc[val_idx, "split"] = "val"
    out.loc[train_idx, "split"] = "train"
    return out


def align_embeddings(split_df: pd.DataFrame, embedding_map: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, np.ndarray]:
    keys = split_df["SMILES"].astype(str).str.strip()
    mask = keys.isin(embedding_map)
    if not bool(mask.any()):
        return split_df.iloc[0:0].copy(), np.zeros((0, 0), dtype=np.float32)
    filtered = split_df.loc[mask].reset_index(drop=True)
    filtered_keys = keys.loc[mask].reset_index(drop=True)
    embeddings = np.stack([embedding_map[smi] for smi in filtered_keys], axis=0).astype(np.float32, copy=False)
    return filtered, embeddings


def chi_feature_matrix(split_df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    aux = split_df[["temperature", "phi"]].to_numpy(dtype=np.float32)
    return np.concatenate([embeddings, aux], axis=1).astype(np.float32, copy=False)


def get_backbone_num_layers(backbone: nn.Module) -> int:
    if hasattr(backbone, "layers"):
        return int(len(backbone.layers))
    if hasattr(backbone, "blocks"):
        return int(len(backbone.blocks))
    return int(getattr(backbone, "num_layers", 0))


def normalize_numeric_options(options, default: Sequence[Any], cast) -> List[Any]:
    if options is None:
        options = default
    if isinstance(options, (int, float, str)):
        options = [options]
    normalized = []
    for option in options:
        try:
            normalized.append(cast(option))
        except Exception:
            continue
    return normalized or [cast(v) for v in default]


def normalize_finetune_options(options, backbone_num_layers: int) -> List[int]:
    max_layers = max(int(backbone_num_layers), 0)
    if options is None:
        return [0, max_layers]
    values = normalize_numeric_options(options, [0, max_layers], int)
    clipped = sorted({min(max(int(value), 0), max_layers) for value in values})
    return clipped or [0, max_layers]


def suggest_finetune_last_layers(trial, options: Sequence[int]) -> int:
    values = sorted({int(v) for v in options})
    if not values:
        return 0
    if len(values) == 1:
        return int(values[0])
    if len(values) == 2:
        return int(trial.suggest_int("finetune_last_layers", int(values[0]), int(values[1])))
    return int(trial.suggest_categorical("finetune_last_layers", values))


def set_backbone_finetune_mode(backbone: nn.Module, view_type: str, finetune_last_layers: int) -> None:
    for param in backbone.parameters():
        param.requires_grad = False

    finetune_last_layers = max(int(finetune_last_layers), 0)
    if finetune_last_layers <= 0:
        return

    if view_type == "graph":
        trainable_blocks = list(backbone.blocks[-finetune_last_layers:]) if hasattr(backbone, "blocks") else []
        for block in trainable_blocks:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(backbone, "ln_final"):
            for param in backbone.ln_final.parameters():
                param.requires_grad = True
        return

    trainable_layers = list(backbone.layers[-finetune_last_layers:]) if hasattr(backbone, "layers") else []
    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True
    if hasattr(backbone, "final_norm"):
        for param in backbone.final_norm.parameters():
            param.requires_grad = True


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], dropout: float, output_dim: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        current = int(input_dim)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(current, int(hidden)))
            layers.append(nn.ReLU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            current = int(hidden)
        layers.append(nn.Linear(current, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SequenceFineTunePredictor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: MLP,
        pooling: str,
        default_timestep: int,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pooling = str(pooling)
        self.default_timestep = int(default_timestep)
        self.register_buffer("feature_mean", torch.tensor(np.asarray(scaler_mean, dtype=np.float32), dtype=torch.float32))
        safe_scale = np.where(np.abs(np.asarray(scaler_scale, dtype=np.float32)) < 1e-12, 1.0, scaler_scale)
        self.register_buffer("feature_scale", torch.tensor(np.asarray(safe_scale, dtype=np.float32), dtype=torch.float32))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        timesteps = torch.full((input_ids.size(0),), self.default_timestep, device=input_ids.device, dtype=torch.long)
        pooled = self.backbone.get_pooled_output(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
            pooling=self.pooling,
        )
        features = torch.cat([pooled, aux.to(dtype=pooled.dtype)], dim=1) if aux is not None else pooled
        scaled = (features - self.feature_mean) / self.feature_scale
        return self.head(scaled)


class GraphFineTunePredictor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: MLP,
        pooling: str,
        default_timestep: int,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pooling = str(pooling)
        self.default_timestep = int(default_timestep)
        self.register_buffer("feature_mean", torch.tensor(np.asarray(scaler_mean, dtype=np.float32), dtype=torch.float32))
        safe_scale = np.where(np.abs(np.asarray(scaler_scale, dtype=np.float32)) < 1e-12, 1.0, scaler_scale)
        self.register_buffer("feature_scale", torch.tensor(np.asarray(safe_scale, dtype=np.float32), dtype=torch.float32))

    def forward(self, X: torch.Tensor, E: torch.Tensor, M: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        timesteps = torch.full((X.size(0),), self.default_timestep, device=X.device, dtype=torch.long)
        pooled = self.backbone.get_node_embeddings(X, E, timesteps, M, pooling=self.pooling)
        features = torch.cat([pooled, aux.to(dtype=pooled.dtype)], dim=1) if aux is not None else pooled
        scaled = (features - self.feature_mean) / self.feature_scale
        return self.head(scaled)


def sanitize_scaler(scaler: StandardScaler) -> StandardScaler:
    scaler.scale_ = np.where(np.abs(scaler.scale_) < 1e-12, 1.0, scaler.scale_).astype(np.float32, copy=False)
    scaler.mean_ = np.asarray(scaler.mean_, dtype=np.float32)
    scaler.var_ = np.asarray(scaler.var_, dtype=np.float32)
    return scaler


def build_finetune_predictor(
    *,
    assets: ViewAssets,
    scaler: StandardScaler,
    params: Dict[str, Any],
    device: str,
) -> nn.Module:
    backbone = copy.deepcopy(assets.backbone).to(device)
    set_backbone_finetune_mode(backbone, assets.view_type, int(params.get("finetune_last_layers", 0)))
    head = MLP(int(scaler.mean_.shape[0]), params["hidden_sizes"], dropout=float(params["dropout"])).to(device)
    if assets.view_type == "graph":
        predictor = GraphFineTunePredictor(
            backbone=backbone,
            head=head,
            pooling=assets.pooling,
            default_timestep=assets.timestep,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
        )
    else:
        predictor = SequenceFineTunePredictor(
            backbone=backbone,
            head=head,
            pooling=assets.pooling,
            default_timestep=assets.timestep,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
        )
    return predictor.to(device)


def capture_predictor_state(predictor: nn.Module) -> Dict[str, torch.Tensor]:
    trainable_keys = {name for name, param in predictor.named_parameters() if param.requires_grad}
    keep_keys = trainable_keys | {"feature_mean", "feature_scale"}
    return {
        key: value.detach().cpu().clone()
        for key, value in predictor.state_dict().items()
        if key in keep_keys
    }


def extract_trainable_backbone_state(backbone: nn.Module) -> Dict[str, torch.Tensor]:
    trainable_keys = {name for name, param in backbone.named_parameters() if param.requires_grad}
    return {
        key: value.detach().cpu().clone()
        for key, value in backbone.state_dict().items()
        if key in trainable_keys
    }


def empty_raw_inputs(assets: ViewAssets) -> Dict[str, np.ndarray]:
    if assets.view_type == "graph":
        nmax = int(getattr(assets.tokenizer, "Nmax", 0))
        return {
            "X": np.zeros((0, nmax), dtype=np.int64),
            "E": np.zeros((0, nmax, nmax), dtype=np.int64),
            "M": np.zeros((0, nmax), dtype=np.float32),
        }
    max_length = int(getattr(assets.tokenizer, "max_length", 0))
    return {
        "input_ids": np.zeros((0, max_length), dtype=np.int64),
        "attention_mask": np.zeros((0, max_length), dtype=np.int64),
    }


def raw_input_cache_metadata(assets: ViewAssets) -> Dict[str, Any]:
    metadata = embedding_cache_metadata(assets)
    metadata["cache_kind"] = "raw_inputs"
    return metadata


def raw_input_cache_metadata_from_info(info: ViewInfo) -> Dict[str, Any]:
    metadata = embedding_cache_metadata_from_info(info)
    metadata["cache_kind"] = "raw_inputs"
    return metadata


def view_type_from_source(source: ViewAssets | ViewInfo | str) -> str:
    if isinstance(source, str):
        return source
    return str(source.view_type)


def empty_raw_inputs_for_view(view_type: str) -> Dict[str, np.ndarray]:
    if view_type == "graph":
        return {
            "X": np.zeros((0, 0), dtype=np.int64),
            "E": np.zeros((0, 0, 0), dtype=np.int64),
            "M": np.zeros((0, 0), dtype=np.float32),
        }
    return {
        "input_ids": np.zeros((0, 0), dtype=np.int64),
        "attention_mask": np.zeros((0, 0), dtype=np.int64),
    }


def stack_raw_rows(raw_rows: Sequence[Dict[str, np.ndarray]], assets: ViewAssets | ViewInfo | str) -> Dict[str, np.ndarray]:
    view_type = view_type_from_source(assets)
    if not raw_rows:
        return empty_raw_inputs(assets) if isinstance(assets, ViewAssets) else empty_raw_inputs_for_view(view_type)
    if view_type == "graph":
        return {
            "X": np.stack([row["X"] for row in raw_rows], axis=0).astype(np.int64, copy=False),
            "E": np.stack([row["E"] for row in raw_rows], axis=0).astype(np.int64, copy=False),
            "M": np.stack([row["M"] for row in raw_rows], axis=0).astype(np.float32, copy=False),
        }
    return {
        "input_ids": np.stack([row["input_ids"] for row in raw_rows], axis=0).astype(np.int64, copy=False),
        "attention_mask": np.stack([row["attention_mask"] for row in raw_rows], axis=0).astype(np.int64, copy=False),
    }


def load_raw_input_cache(
    cache_path: Path,
    requested: set[str],
    assets: Optional[ViewAssets] = None,
    expected_metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    if not cache_path.exists():
        return None
    payload = np.load(cache_path, allow_pickle=True)
    if expected_metadata is not None and not cache_metadata_matches(parse_cache_metadata(payload), expected_metadata):
        print(f"Ignoring stale raw-input cache metadata: {cache_path}")
        return None
    smiles = [str(x) for x in payload["smiles"].tolist()]
    if not requested.issubset(set(smiles)):
        return None
    view_type = assets.view_type if assets is not None else str((expected_metadata or {}).get("view_type", "sequence"))
    if view_type == "graph":
        keys = ["X", "E", "M"]
    else:
        keys = ["input_ids", "attention_mask"]
    if any(key not in payload.files for key in keys):
        return None
    arrays = {key: np.asarray(payload[key]) for key in keys}
    if any(arrays[key].shape[0] != len(smiles) for key in keys):
        return None
    return {
        smi: {key: np.asarray(arrays[key][idx]).copy() for key in keys}
        for idx, smi in enumerate(smiles)
    }


def build_or_load_raw_input_cache(
    *,
    smiles_values: Sequence[str],
    assets: ViewAssets,
    cache_path: Path,
    force_rebuild: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted({str(smi).strip() for smi in smiles_values if str(smi).strip()})
    requested = set(unique)
    metadata = raw_input_cache_metadata(assets)
    cached = load_raw_input_cache(cache_path, requested, assets, expected_metadata=metadata) if not force_rebuild else None
    if cached is not None:
        return cached
    if cache_path.exists() and not force_rebuild:
        print(f"Rebuilding stale or incomplete raw-input cache for view={assets.view}: {cache_path}")

    raw_map: Dict[str, Dict[str, np.ndarray]] = {}
    for smi in unique:
        if assets.view_type == "graph":
            try:
                encoded = assets.tokenizer.encode(smi)
            except Exception:
                continue
            raw_map[smi] = {
                "X": np.asarray(encoded["X"], dtype=np.int64),
                "E": np.asarray(encoded["E"], dtype=np.int64),
                "M": np.asarray(encoded["M"], dtype=np.float32),
            }
        else:
            text = transform_text_for_view(smi, assets.view)
            if not text:
                continue
            encoded = encode_sequence_text(text, assets)
            if encoded is None:
                continue
            raw_map[smi] = encoded

    valid_smiles = list(raw_map.keys())
    raw_inputs = stack_raw_rows([raw_map[smi] for smi in valid_smiles], assets)
    np.savez_compressed(
        cache_path,
        smiles=np.asarray(valid_smiles, dtype=object),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=object),
        **raw_inputs,
    )
    return raw_map


def save_embedding_cache(cache_path: Path, valid_smiles: Sequence[str], embeddings: np.ndarray, metadata: Mapping[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        smiles=np.asarray(list(valid_smiles), dtype=object),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True), dtype=object),
    )


def save_raw_input_cache(cache_path: Path, valid_smiles: Sequence[str], raw_inputs: Dict[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        smiles=np.asarray(list(valid_smiles), dtype=object),
        metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True), dtype=object),
        **raw_inputs,
    )


def build_embedding_and_raw_caches(
    *,
    smiles_values: Sequence[str],
    assets: ViewAssets,
    embedding_cache_path: Path,
    raw_cache_path: Path,
    device: str,
    force_rebuild: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    unique = sorted({str(smi).strip() for smi in smiles_values if str(smi).strip()})
    requested = set(unique)
    embedding_metadata = embedding_cache_metadata(assets)
    raw_metadata = raw_input_cache_metadata(assets)
    if not force_rebuild:
        embedding_map = load_embedding_cache(embedding_cache_path, requested, expected_metadata=embedding_metadata)
        raw_map = load_raw_input_cache(raw_cache_path, requested, assets=assets, expected_metadata=raw_metadata)
        if embedding_map is not None and raw_map is not None:
            return embedding_map, raw_map

    raw_map: Dict[str, Dict[str, np.ndarray]] = {}
    valid_smiles: List[str] = []
    raw_rows: List[Dict[str, np.ndarray]] = []
    for smi in unique:
        if assets.view_type == "graph":
            try:
                encoded = assets.tokenizer.encode(smi)
            except Exception:
                continue
            payload = {
                "X": np.asarray(encoded["X"], dtype=np.int64),
                "E": np.asarray(encoded["E"], dtype=np.int64),
                "M": np.asarray(encoded["M"], dtype=np.float32),
            }
        else:
            text = transform_text_for_view(smi, assets.view)
            if not text:
                continue
            payload = encode_sequence_text(text, assets)
            if payload is None:
                continue
        raw_map[smi] = payload
        valid_smiles.append(smi)
        raw_rows.append(payload)

    raw_inputs = stack_raw_rows(raw_rows, assets)
    if assets.view_type == "graph":
        embeddings = embed_graph_raw_inputs(raw_inputs, assets, device)
    else:
        embeddings = embed_sequence_raw_inputs(raw_inputs, assets, device)
    save_embedding_cache(embedding_cache_path, valid_smiles, embeddings, embedding_metadata)
    save_raw_input_cache(raw_cache_path, valid_smiles, raw_inputs, raw_metadata)
    return {smi: embeddings[idx] for idx, smi in enumerate(valid_smiles)}, raw_map


def align_embeddings_and_raw(
    split_df: pd.DataFrame,
    embedding_map: Dict[str, np.ndarray],
    raw_map: Dict[str, Dict[str, np.ndarray]],
    assets: ViewAssets | ViewInfo | str,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
    keys = split_df["SMILES"].astype(str).str.strip()
    mask = keys.isin(set(embedding_map)) & keys.isin(set(raw_map))
    if not bool(mask.any()):
        empty_raw = empty_raw_inputs(assets) if isinstance(assets, ViewAssets) else empty_raw_inputs_for_view(view_type_from_source(assets))
        return split_df.iloc[0:0].copy(), np.zeros((0, 0), dtype=np.float32), empty_raw
    filtered = split_df.loc[mask].reset_index(drop=True)
    filtered_keys = keys.loc[mask].reset_index(drop=True)
    embeddings = np.stack([embedding_map[smi] for smi in filtered_keys], axis=0).astype(np.float32, copy=False)
    raw_inputs = stack_raw_rows([raw_map[smi] for smi in filtered_keys], assets)
    return filtered, embeddings, raw_inputs


def slice_raw_inputs(raw_inputs: Dict[str, np.ndarray], rows: Sequence[int] | np.ndarray) -> Dict[str, np.ndarray]:
    row_idx = np.asarray(rows, dtype=np.int64)
    return {key: np.asarray(value[row_idx]).copy() for key, value in raw_inputs.items()}


def make_model_batch(
    raw_inputs: Dict[str, np.ndarray],
    rows: Sequence[int] | np.ndarray,
    view_type: str,
    device: str,
    aux_features: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    row_idx = np.asarray(rows, dtype=np.int64)
    if view_type == "graph":
        batch = {
            "X": torch.tensor(raw_inputs["X"][row_idx], device=device, dtype=torch.long),
            "E": torch.tensor(raw_inputs["E"][row_idx], device=device, dtype=torch.long),
            "M": torch.tensor(raw_inputs["M"][row_idx], device=device, dtype=torch.float32),
        }
    else:
        batch = {
            "input_ids": torch.tensor(raw_inputs["input_ids"][row_idx], device=device, dtype=torch.long),
            "attention_mask": torch.tensor(raw_inputs["attention_mask"][row_idx], device=device, dtype=torch.long),
        }
    if aux_features is not None:
        batch["aux"] = torch.tensor(np.asarray(aux_features[row_idx], dtype=np.float32), device=device, dtype=torch.float32)
    return batch


def tensor_dataset(X: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None) -> torch.utils.data.TensorDataset:
    x_t = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1), dtype=torch.float32)
    if weight is None:
        w_t = torch.ones_like(y_t)
    else:
        w_t = torch.tensor(np.asarray(weight, dtype=np.float32).reshape(-1), dtype=torch.float32)
    return torch.utils.data.TensorDataset(x_t, y_t, w_t)


def compute_polymer_weights(split_df: pd.DataFrame, clip_ratio: float) -> np.ndarray:
    stats = split_df.groupby("polymer_id")["chi"].agg(["count", "var"]).reset_index()
    stats["var"] = pd.to_numeric(stats["var"], errors="coerce").fillna(0.0)
    stats["raw"] = 1.0 / (stats["count"].astype(float) * np.sqrt(stats["var"].astype(float) + 1e-4))
    median = float(np.nanmedian(stats["raw"].to_numpy(dtype=float))) if not stats.empty else 1.0
    stats["weight"] = np.minimum(stats["raw"].astype(float), float(clip_ratio) * median)
    mean = float(np.nanmean(stats["weight"].to_numpy(dtype=float))) if not stats.empty else 1.0
    if mean > 0 and np.isfinite(mean):
        stats["weight"] = stats["weight"].astype(float) / mean
    mapping = dict(zip(stats["polymer_id"].astype(int), stats["weight"].astype(float)))
    return split_df["polymer_id"].map(mapping).fillna(1.0).to_numpy(dtype=np.float32)


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    y = np.asarray(labels, dtype=int).reshape(-1)
    if y.size == 0 or len(np.unique(y)) < 2:
        return np.ones_like(y, dtype=np.float32)
    counts = np.bincount(y, minlength=2).astype(float)
    counts[counts == 0.0] = np.nan
    weights = 1.0 / counts[y]
    mean = float(np.nanmean(weights)) if weights.size else 1.0
    if mean > 0 and np.isfinite(mean):
        weights = weights / mean
    return np.nan_to_num(weights, nan=1.0).astype(np.float32)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def polymer_balanced_nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    polymer_ids: np.ndarray,
    std_floor: float,
    clip: float,
) -> float:
    rows = []
    df = pd.DataFrame({"polymer_id": polymer_ids, "true": y_true, "pred": y_pred})
    for _, sub in df.groupby("polymer_id"):
        if len(sub) == 0:
            continue
        denom = max(float(np.std(sub["true"].to_numpy(dtype=float))), float(std_floor))
        val = rmse(sub["true"].to_numpy(dtype=float), sub["pred"].to_numpy(dtype=float)) / denom
        rows.append(min(float(val), float(clip)))
    return float(np.mean(rows)) if rows else float("nan")


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {k: float("nan") for k in ["balanced_accuracy", "auroc", "auprc", "brier"]}
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= 0.5).astype(int)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, np.clip(p, 0.0, 1.0))),
    }
    if len(np.unique(y)) >= 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def train_eval_model(
    *,
    task: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    hidden_sizes: Sequence[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    use_scheduler: bool,
    scheduler_min_lr: float,
    device: str,
    seed: int,
    objective_metric: str,
    train_weight: Optional[np.ndarray] = None,
    val_polymer_ids: Optional[np.ndarray] = None,
    nrmse_std_floor: float = 0.02,
    nrmse_clip: float = 10.0,
    predict_batch_size: int = 1024,
) -> Tuple[MLP, StandardScaler, Dict[str, List[float]], Dict[str, float]]:
    set_seed(seed)
    scaler = StandardScaler()
    train_Xs = scaler.fit_transform(train_X).astype(np.float32)
    val_Xs = scaler.transform(val_X).astype(np.float32) if len(val_X) else np.zeros((0, train_X.shape[1]), dtype=np.float32)
    model = MLP(train_X.shape[1], hidden_sizes, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(num_epochs)), eta_min=float(scheduler_min_lr))
        if use_scheduler
        else None
    )
    loader = torch.utils.data.DataLoader(
        tensor_dataset(train_Xs, train_y, train_weight),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
    )
    has_val = len(val_Xs) > 0
    best_state = None
    best_value = float("inf") if objective_metric in {"rmse", "loss", "poly_nrmse", "val_poly_nrmse"} else -float("inf")
    wait = 0
    history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_loss": [], "metric": []}
    loss_fn_cls = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(1, int(num_epochs) + 1):
        model.train()
        batch_losses = []
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if task == "classification":
                loss_vec = loss_fn_cls(pred, yb)
                loss = torch.mean(wb * loss_vec)
            else:
                loss = torch.mean(wb * (pred - yb) ** 2)
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            opt.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        if scheduler is not None:
            scheduler.step()

        if has_val:
            val_pred = predict_model(model, scaler, val_X, device=device, task=task, batch_size=predict_batch_size)
            if task == "classification":
                val_loss = float(np.mean(-(val_y * np.log(np.clip(val_pred, 1e-7, 1.0)) + (1.0 - val_y) * np.log(np.clip(1.0 - val_pred, 1e-7, 1.0)))))
                metric = classification_metrics(val_y, val_pred)["balanced_accuracy"]
                lower = False
            else:
                val_loss = float(np.mean((val_pred - val_y) ** 2))
                if objective_metric in {"val_r2", "r2"}:
                    metric = regression_metrics(val_y, val_pred)["r2"]
                    lower = False
                elif objective_metric in {"val_poly_nrmse", "poly_nrmse"} and val_polymer_ids is not None:
                    metric = polymer_balanced_nrmse(
                        val_y,
                        val_pred,
                        np.asarray(val_polymer_ids, dtype=int),
                        std_floor=float(nrmse_std_floor),
                        clip=float(nrmse_clip),
                    )
                    lower = True
                else:
                    metric = rmse(val_y, val_pred)
                    lower = True
            improved = np.isfinite(metric) and ((lower and metric < best_value) or ((not lower) and metric > best_value))
            if improved or best_state is None:
                best_value = float(metric) if np.isfinite(metric) else best_value
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= int(patience):
                    break
        else:
            val_loss = float("nan")
            metric = float("nan")
            best_state = copy.deepcopy(model.state_dict())

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(batch_losses)) if batch_losses else float("nan"))
        history["val_loss"].append(val_loss)
        history["metric"].append(float(metric) if np.isfinite(metric) else float("nan"))

    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {"best_metric": float(best_value) if np.isfinite(best_value) else float("nan"), "epochs_trained": int(len(history["epoch"]))}
    return model, scaler, history, summary


@torch.no_grad()
def predict_model(model: MLP, scaler: StandardScaler, X: np.ndarray, *, device: str, task: str, batch_size: int = 1024) -> np.ndarray:
    if len(X) == 0:
        return np.asarray([], dtype=np.float32)
    xs = scaler.transform(X).astype(np.float32)
    preds: List[np.ndarray] = []
    model.eval()
    batch_size = max(1, int(batch_size))
    for start in range(0, xs.shape[0], batch_size):
        xb = torch.tensor(xs[start:start + batch_size], dtype=torch.float32, device=device)
        out = model(xb)
        if task == "classification":
            out = torch.sigmoid(out)
        preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


@torch.no_grad()
def predict_joint_model(
    predictor: nn.Module,
    raw_inputs: Dict[str, np.ndarray],
    rows: Sequence[int] | np.ndarray,
    *,
    view_type: str,
    device: str,
    task: str,
    aux_features: Optional[np.ndarray] = None,
    batch_size: int = 1024,
) -> np.ndarray:
    row_idx = np.asarray(rows, dtype=np.int64)
    if row_idx.size == 0:
        return np.asarray([], dtype=np.float32)
    predictor = predictor.to(device)
    predictor.eval()
    preds: List[np.ndarray] = []
    batch_size = max(1, int(batch_size))
    for start in range(0, row_idx.size, batch_size):
        batch_rows = row_idx[start:start + batch_size]
        batch = make_model_batch(raw_inputs, batch_rows, view_type=view_type, device=device, aux_features=aux_features)
        out = predictor(**batch)
        if task == "classification":
            out = torch.sigmoid(out)
        preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def train_eval_joint_model(
    *,
    task: str,
    features: np.ndarray,
    y: np.ndarray,
    raw_inputs: Dict[str, np.ndarray],
    train_rows: Sequence[int] | np.ndarray,
    val_rows: Sequence[int] | np.ndarray,
    assets: ViewAssets,
    aux_features: Optional[np.ndarray],
    hidden_sizes: Sequence[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    use_scheduler: bool,
    scheduler_min_lr: float,
    device: str,
    seed: int,
    objective_metric: str,
    finetune_last_layers: int,
    train_weight: Optional[np.ndarray] = None,
    val_polymer_ids: Optional[np.ndarray] = None,
    nrmse_std_floor: float = 0.02,
    nrmse_clip: float = 10.0,
    predict_batch_size: int = 1024,
) -> Tuple[nn.Module, StandardScaler, Dict[str, List[float]], Dict[str, float]]:
    train_rows = np.asarray(train_rows, dtype=np.int64)
    val_rows = np.asarray(val_rows, dtype=np.int64)
    if train_rows.size == 0:
        raise ValueError("Cannot train joint backbone/head model with an empty train set.")

    set_seed(seed)
    scaler = sanitize_scaler(StandardScaler().fit(features[train_rows]))
    params = {
        "hidden_sizes": [int(x) for x in hidden_sizes],
        "dropout": float(dropout),
        "finetune_last_layers": int(finetune_last_layers),
    }
    predictor = build_finetune_predictor(assets=assets, scaler=scaler, params=params, device=device)
    trainable_params = [param for param in predictor.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("Joint training has no trainable parameters.")
    opt = torch.optim.AdamW(trainable_params, lr=float(learning_rate), weight_decay=float(weight_decay))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(num_epochs)), eta_min=float(scheduler_min_lr))
        if use_scheduler
        else None
    )

    local_idx = np.arange(train_rows.size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if train_weight is None:
        train_weight_arr = np.ones(train_rows.size, dtype=np.float32)
    else:
        train_weight_arr = np.asarray(train_weight, dtype=np.float32).reshape(-1)
        if train_weight_arr.shape[0] != train_rows.size:
            raise ValueError("train_weight length must match train_rows length for joint training.")

    has_val = val_rows.size > 0
    best_state = None
    best_value = float("inf") if objective_metric in {"rmse", "loss", "poly_nrmse", "val_poly_nrmse"} else -float("inf")
    wait = 0
    history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_loss": [], "metric": []}
    loss_fn_cls = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(1, int(num_epochs) + 1):
        predictor.train()
        rng.shuffle(local_idx)
        batch_losses = []
        for start in range(0, local_idx.size, max(1, int(batch_size))):
            batch_local = local_idx[start:start + max(1, int(batch_size))]
            batch_rows = train_rows[batch_local]
            batch = make_model_batch(raw_inputs, batch_rows, view_type=assets.view_type, device=device, aux_features=aux_features)
            yb = torch.tensor(y[batch_rows].astype(np.float32, copy=False), dtype=torch.float32, device=device)
            wb = torch.tensor(train_weight_arr[batch_local].astype(np.float32, copy=False), dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            pred = predictor(**batch)
            if task == "classification":
                loss_vec = loss_fn_cls(pred, yb)
                loss = torch.mean(wb * loss_vec)
            else:
                loss = torch.mean(wb * (pred - yb) ** 2)
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(gradient_clip_norm))
            opt.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        if scheduler is not None:
            scheduler.step()

        should_stop = False
        if has_val:
            val_pred = predict_joint_model(
                predictor,
                raw_inputs,
                val_rows,
                view_type=assets.view_type,
                device=device,
                task=task,
                aux_features=aux_features,
                batch_size=predict_batch_size,
            )
            val_y = y[val_rows]
            if task == "classification":
                val_loss = float(np.mean(-(val_y * np.log(np.clip(val_pred, 1e-7, 1.0)) + (1.0 - val_y) * np.log(np.clip(1.0 - val_pred, 1e-7, 1.0)))))
                metric = classification_metrics(val_y, val_pred)["balanced_accuracy"]
                lower = False
            else:
                val_loss = float(np.mean((val_pred - val_y) ** 2))
                if objective_metric in {"val_r2", "r2"}:
                    metric = regression_metrics(val_y, val_pred)["r2"]
                    lower = False
                elif objective_metric in {"val_poly_nrmse", "poly_nrmse"} and val_polymer_ids is not None:
                    metric = polymer_balanced_nrmse(
                        val_y,
                        val_pred,
                        np.asarray(val_polymer_ids, dtype=int),
                        std_floor=float(nrmse_std_floor),
                        clip=float(nrmse_clip),
                    )
                    lower = True
                else:
                    metric = rmse(val_y, val_pred)
                    lower = True
            improved = np.isfinite(metric) and ((lower and metric < best_value) or ((not lower) and metric > best_value))
            if improved or best_state is None:
                best_value = float(metric) if np.isfinite(metric) else best_value
                best_state = capture_predictor_state(predictor)
                wait = 0
            else:
                wait += 1
                should_stop = wait >= int(patience)
        else:
            val_loss = float("nan")
            metric = float("nan")
            best_state = capture_predictor_state(predictor)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(batch_losses)) if batch_losses else float("nan"))
        history["val_loss"].append(val_loss)
        history["metric"].append(float(metric) if np.isfinite(metric) else float("nan"))
        if should_stop:
            break

    if best_state is not None:
        predictor.load_state_dict(best_state, strict=False)
    summary = {"best_metric": float(best_value) if np.isfinite(best_value) else float("nan"), "epochs_trained": int(len(history["epoch"]))}
    return predictor, scaler, history, summary


def suggest_hyperparameters(trial, search: Dict[str, Any], backbone_num_layers: int) -> Dict[str, Any]:
    num_layers_space = list(search.get("num_layers", [2, 3, 4]))
    if len(num_layers_space) == 2:
        num_layers = trial.suggest_int("num_layers", int(min(num_layers_space)), int(max(num_layers_space)))
    else:
        num_layers = int(trial.suggest_categorical("num_layers", [int(x) for x in num_layers_space]))
    hidden_units = [int(x) for x in search.get("hidden_units", [128, 256, 512])]
    hidden_sizes = [int(trial.suggest_categorical(f"hidden_{i}", hidden_units)) for i in range(num_layers)]
    dropout = float(trial.suggest_categorical("dropout", [float(x) for x in search.get("dropout", [0.1])]))
    lr_space = [float(x) for x in search.get("learning_rate", [1e-3])]
    if len(lr_space) == 2:
        lr = float(trial.suggest_float("learning_rate", min(lr_space), max(lr_space), log=bool(search.get("learning_rate_log", True))))
    else:
        lr = float(trial.suggest_categorical("learning_rate", lr_space))
    wd_space = [float(x) for x in search.get("weight_decay", [1e-5])]
    if len(wd_space) == 2:
        wd = float(trial.suggest_float("weight_decay", min(wd_space), max(wd_space), log=bool(search.get("weight_decay_log", True))))
    else:
        wd = float(trial.suggest_categorical("weight_decay", wd_space))
    batch_size = int(trial.suggest_categorical("batch_size", [int(x) for x in search.get("batch_size", [128])]))
    finetune_options = normalize_finetune_options(search.get("finetune_last_layers"), int(backbone_num_layers))
    return {
        "num_layers": int(num_layers),
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "learning_rate": lr,
        "weight_decay": wd,
        "batch_size": batch_size,
        "finetune_last_layers": suggest_finetune_last_layers(trial, finetune_options),
    }


def params_from_config(task_cfg: Dict[str, Any]) -> Dict[str, Any]:
    hidden = [int(x) for x in task_cfg.get("hidden_sizes", [256, 128])]
    return {
        "num_layers": len(hidden),
        "hidden_sizes": hidden,
        "dropout": float(task_cfg.get("dropout", 0.1)),
        "learning_rate": float(task_cfg.get("learning_rate", 1e-3)),
        "weight_decay": float(task_cfg.get("weight_decay", 1e-5)),
        "batch_size": int(task_cfg.get("batch_size", 128)),
        "finetune_last_layers": int(task_cfg.get("finetune_last_layers", 0)),
    }


def task_allows_backbone_finetune(task_cfg: Dict[str, Any], backbone_num_layers: int, force_tune: Optional[bool]) -> bool:
    tune_enabled = bool(task_cfg.get("tune", False)) if force_tune is None else bool(force_tune)
    if tune_enabled:
        search = task_cfg.get("optuna_search_space", {})
        options = normalize_finetune_options(search.get("finetune_last_layers"), int(backbone_num_layers))
        return any(int(value) > 0 for value in options)
    return int(task_cfg.get("finetune_last_layers", 0)) > 0


def build_cv_folds(split_df: pd.DataFrame, split_mode: str, cv_folds: int, seed: int, task: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    split_mode = "smiles" if split_mode == "polymer" else split_mode
    dev = split_df[split_df["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    stratify_target = "chi" if task == "regression" else "water_miscible"
    if split_mode == "smiles":
        units = split_units_for_stratification(dev)
        unit_ids = units["polymer_id"].to_numpy()
        if len(unit_ids) < 2:
            split_mode = "random"
        else:
            labels = stratification_labels(units, stratify_target)
            if labels is not None:
                n_splits = min(int(cv_folds), len(unit_ids), int(np.min(np.bincount(labels))))
                splitter = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)
            else:
                n_splits = min(int(cv_folds), len(unit_ids))
                splitter = KFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)
            folds = []
            split_iter = splitter.split(unit_ids, labels) if labels is not None else splitter.split(unit_ids)
            for tr_u, va_u in split_iter:
                train_units = set(unit_ids[tr_u].tolist())
                val_units = set(unit_ids[va_u].tolist())
                train_idx = dev.index[dev["polymer_id"].isin(train_units)].to_numpy(dtype=np.int64)
                val_idx = dev.index[dev["polymer_id"].isin(val_units)].to_numpy(dtype=np.int64)
                folds.append((train_idx, val_idx))
            return folds

    if split_mode == "random":
        indices = np.arange(len(dev), dtype=np.int64)
        if len(indices) < 2:
            if len(indices) == 0:
                raise ValueError("No train/val rows available for cross-validation.")
            return [(indices, indices)]
        labels = stratification_labels(dev, stratify_target)
        if labels is not None:
            n_splits = min(int(cv_folds), int(np.min(np.bincount(labels))))
            splitter = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)
            return [(indices[tr], indices[va]) for tr, va in splitter.split(indices, labels)]
        splitter = KFold(n_splits=max(2, min(int(cv_folds), len(indices))), shuffle=True, random_state=seed)
        return [(indices[tr], indices[va]) for tr, va in splitter.split(indices)]

    raise ValueError("split_mode must be 'smiles' or 'random'")


def evaluate_params_cv(
    *,
    task: str,
    split_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    task_cfg: Dict[str, Any],
    params: Dict[str, Any],
    split_mode: str,
    seed: int,
    device: str,
    objective_metric: str,
    epochs: int,
    patience: int,
    raw_inputs: Optional[Dict[str, np.ndarray]] = None,
    assets: Optional[ViewAssets] = None,
    aux_features: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    dev_mask = split_df["split"].isin(["train", "val"]).to_numpy()
    dev_df = split_df.loc[dev_mask].reset_index(drop=True)
    dev_X = X[dev_mask]
    dev_y = y[dev_mask]
    dev_indices = np.flatnonzero(dev_mask)
    dev_raw_inputs = slice_raw_inputs(raw_inputs, dev_indices) if raw_inputs is not None else None
    dev_aux = np.asarray(aux_features[dev_mask], dtype=np.float32) if aux_features is not None else None
    folds = build_cv_folds(dev_df, split_mode=split_mode, cv_folds=int(task_cfg.get("tuning_cv_folds", 5)), seed=seed, task=task)
    predict_batch_size = int(task_cfg.get("predict_batch_size", 1024))
    fold_rows = []
    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        weights = None
        if task == "regression" and task_cfg.get("loss_weighting", "uniform") == "polymer_balanced":
            weights = compute_polymer_weights(dev_df.iloc[train_idx], clip_ratio=float(task_cfg.get("loss_weight_clip_ratio", 10.0)))
        elif task == "classification" and task_cfg.get("loss_weighting", "uniform") == "class_balanced":
            weights = compute_class_weights(dev_y[train_idx])
        finetune_last_layers = int(params.get("finetune_last_layers", 0))
        if finetune_last_layers > 0:
            if assets is None or dev_raw_inputs is None:
                raise RuntimeError("Backbone fine-tuning requested but view assets/raw inputs are unavailable.")
            predictor, _, _, summary = train_eval_joint_model(
                task="classification" if task == "classification" else "regression",
                features=dev_X,
                y=dev_y,
                raw_inputs=dev_raw_inputs,
                train_rows=train_idx,
                val_rows=val_idx,
                assets=assets,
                aux_features=dev_aux,
                hidden_sizes=params["hidden_sizes"],
                dropout=params["dropout"],
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                batch_size=params["batch_size"],
                num_epochs=epochs,
                patience=patience,
                gradient_clip_norm=float(task_cfg.get("gradient_clip_norm", 1.0)),
                use_scheduler=bool(task_cfg.get("use_scheduler", True)),
                scheduler_min_lr=float(task_cfg.get("scheduler_min_lr", 1e-6)),
                device=device,
                seed=seed + 1009 * fold_id,
                objective_metric=objective_metric,
                finetune_last_layers=finetune_last_layers,
                train_weight=weights,
                val_polymer_ids=dev_df.iloc[val_idx]["polymer_id"].to_numpy(dtype=int) if task == "regression" else None,
                nrmse_std_floor=float(task_cfg.get("nrmse_std_floor", 0.02)),
                nrmse_clip=float(task_cfg.get("nrmse_clip", 10.0)),
                predict_batch_size=predict_batch_size,
            )
            pred = predict_joint_model(
                predictor,
                dev_raw_inputs,
                val_idx,
                view_type=assets.view_type,
                device=device,
                task="classification" if task == "classification" else "regression",
                aux_features=dev_aux,
                batch_size=predict_batch_size,
            )
        else:
            model, scaler, _, summary = train_eval_model(
                task="classification" if task == "classification" else "regression",
                train_X=dev_X[train_idx],
                train_y=dev_y[train_idx],
                val_X=dev_X[val_idx],
                val_y=dev_y[val_idx],
                hidden_sizes=params["hidden_sizes"],
                dropout=params["dropout"],
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                batch_size=params["batch_size"],
                num_epochs=epochs,
                patience=patience,
                gradient_clip_norm=float(task_cfg.get("gradient_clip_norm", 1.0)),
                use_scheduler=bool(task_cfg.get("use_scheduler", True)),
                scheduler_min_lr=float(task_cfg.get("scheduler_min_lr", 1e-6)),
                device=device,
                seed=seed + 1009 * fold_id,
                objective_metric=objective_metric,
                train_weight=weights,
                val_polymer_ids=dev_df.iloc[val_idx]["polymer_id"].to_numpy(dtype=int) if task == "regression" else None,
                nrmse_std_floor=float(task_cfg.get("nrmse_std_floor", 0.02)),
                nrmse_clip=float(task_cfg.get("nrmse_clip", 10.0)),
                predict_batch_size=predict_batch_size,
            )
            pred = predict_model(
                model,
                scaler,
                dev_X[val_idx],
                device=device,
                task="classification" if task == "classification" else "regression",
                batch_size=predict_batch_size,
            )
        if task == "classification":
            metrics = classification_metrics(dev_y[val_idx], pred)
            objective_value = metrics["balanced_accuracy"]
        else:
            metrics = regression_metrics(dev_y[val_idx], pred)
            metrics["poly_nrmse"] = polymer_balanced_nrmse(
                dev_y[val_idx],
                pred,
                dev_df.iloc[val_idx]["polymer_id"].to_numpy(dtype=int),
                std_floor=float(task_cfg.get("nrmse_std_floor", 0.02)),
                clip=float(task_cfg.get("nrmse_clip", 10.0)),
            )
            objective_value = metrics["r2"] if objective_metric == "val_r2" else metrics["poly_nrmse"]
        row = {"fold": fold_id, "objective_value": float(objective_value), "epochs_trained": summary["epochs_trained"]}
        row.update(metrics)
        fold_rows.append(row)
    def _safe_nan_stat(values: Sequence[float], reducer) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(reducer(arr)) if arr.size else float("nan")

    metric_cols = [k for k in fold_rows[0].keys() if k not in {"fold"}] if fold_rows else []
    out = {f"cv_mean_{col}": _safe_nan_stat([r.get(col, np.nan) for r in fold_rows], np.mean) for col in metric_cols}
    out.update({f"cv_std_{col}": _safe_nan_stat([r.get(col, np.nan) for r in fold_rows], np.std) for col in metric_cols})
    out["fold_metrics"] = pd.DataFrame(fold_rows)
    return out


def tune_or_select_params(
    *,
    task: str,
    split_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    task_cfg: Dict[str, Any],
    split_mode: str,
    seed: int,
    device: str,
    force_tune: Optional[bool],
    raw_inputs: Optional[Dict[str, np.ndarray]] = None,
    assets: Optional[ViewAssets] = None,
    aux_features: Optional[np.ndarray] = None,
    backbone_num_layers: int = 0,
    trial_log_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    tune_enabled = bool(task_cfg.get("tune", False)) if force_tune is None else bool(force_tune)
    objective_metric = str(task_cfg.get("tuning_objective", "val_r2" if task == "regression" else "balanced_accuracy"))
    direction = "minimize" if objective_metric == "val_poly_nrmse" else "maximize"
    epochs = int(task_cfg.get("tuning_epochs", 50))
    patience = int(task_cfg.get("tuning_patience", 10))
    search = task_cfg.get("optuna_search_space", {})
    trial_rows: List[Dict[str, Any]] = []

    if tune_enabled:
        if optuna is None:
            raise ImportError("optuna is required when tuning is enabled.")
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction=direction, sampler=sampler)

        def objective(trial):
            params: Dict[str, Any] = {}
            bad_value = 1e12 if direction == "minimize" else -1e12
            try:
                params = suggest_hyperparameters(trial, search, backbone_num_layers=backbone_num_layers)
                cv = evaluate_params_cv(
                    task=task,
                    split_df=split_df,
                    X=X,
                    y=y,
                    task_cfg=task_cfg,
                    params=params,
                    split_mode=split_mode,
                    seed=seed + 10007 * (trial.number + 1),
                    device=device,
                    objective_metric=objective_metric,
                    epochs=epochs,
                    patience=patience,
                    raw_inputs=raw_inputs,
                    assets=assets,
                    aux_features=aux_features,
                )
                if task == "classification":
                    value = cv["cv_mean_balanced_accuracy"]
                else:
                    value = cv["cv_mean_r2"] if objective_metric == "val_r2" else cv["cv_mean_poly_nrmse"]
            except Exception as exc:
                row = {
                    "trial": int(trial.number),
                    "task": task,
                    "status": "failed",
                    "objective_metric": objective_metric,
                    "objective_direction": direction,
                    "objective_value": float(bad_value),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **params,
                }
                trial_rows.append(row)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return float(bad_value)
            row = {
                "trial": int(trial.number),
                "task": task,
                "status": "completed",
                "objective_metric": objective_metric,
                "objective_direction": direction,
                "objective_value": float(value),
                **params,
            }
            for key, val in cv.items():
                if key != "fold_metrics":
                    row[key] = val
            trial_rows.append(row)
            return float(value) if np.isfinite(value) else (1e12 if direction == "minimize" else -1e12)

        study.optimize(objective, n_trials=int(task_cfg.get("n_trials", 50)), show_progress_bar=True)
        successful_rows = [row for row in trial_rows if row.get("status") != "failed"]
        if not successful_rows:
            if trial_log_path is not None:
                trial_log_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(trial_rows).to_csv(trial_log_path, index=False)
            raise RuntimeError("All Optuna trials failed; see optuna_trials.csv for errors.")
        best = max(successful_rows, key=lambda row: float(row["objective_value"])) if direction == "maximize" else min(successful_rows, key=lambda row: float(row["objective_value"]))
        params = {
            "num_layers": int(best["num_layers"]),
            "hidden_sizes": [int(x) for x in best["hidden_sizes"]],
            "dropout": float(best["dropout"]),
            "learning_rate": float(best["learning_rate"]),
            "weight_decay": float(best["weight_decay"]),
            "batch_size": int(best["batch_size"]),
            "finetune_last_layers": int(best.get("finetune_last_layers", 0)),
        }
        return params, pd.DataFrame(trial_rows), best

    params = params_from_config(task_cfg)
    params["finetune_last_layers"] = min(max(int(params.get("finetune_last_layers", 0)), 0), max(int(backbone_num_layers), 0))
    cv = evaluate_params_cv(
        task=task,
        split_df=split_df,
        X=X,
        y=y,
        task_cfg=task_cfg,
        params=params,
        split_mode=split_mode,
        seed=seed + 10007,
        device=device,
        objective_metric=objective_metric,
        epochs=epochs,
        patience=patience,
        raw_inputs=raw_inputs,
        assets=assets,
        aux_features=aux_features,
    )
    value_key = "cv_mean_balanced_accuracy" if task == "classification" else ("cv_mean_r2" if objective_metric == "val_r2" else "cv_mean_poly_nrmse")
    row = {"trial": 0, "task": task, "status": "completed", "objective_metric": objective_metric, "objective_direction": direction, "objective_value": cv.get(value_key, np.nan), **params}
    for key, val in cv.items():
        if key != "fold_metrics":
            row[key] = val
    return params, pd.DataFrame([row]), row


def final_epoch_budget(task_cfg: Dict[str, Any], best_payload: Mapping[str, Any]) -> int:
    default_epochs = int(task_cfg.get("num_epochs", 500))
    if not bool(task_cfg.get("final_epochs_from_cv", True)):
        return default_epochs
    value = best_payload.get("cv_mean_epochs_trained", best_payload.get("epochs_trained", np.nan))
    try:
        epochs = int(round(float(value)))
    except (TypeError, ValueError):
        epochs = default_epochs
    if epochs < 1:
        epochs = default_epochs
    return max(1, min(default_epochs, epochs))


def train_final_and_predict(
    *,
    task: str,
    split_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    task_cfg: Dict[str, Any],
    params: Dict[str, Any],
    best_payload: Mapping[str, Any],
    device: str,
    seed: int,
    raw_inputs: Optional[Dict[str, np.ndarray]] = None,
    assets: Optional[ViewAssets] = None,
    aux_features: Optional[np.ndarray] = None,
    backbone_num_layers: int = 0,
) -> Tuple[MLP, StandardScaler, Dict[str, List[float]], pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Any]]:
    final_df = split_df.copy().reset_index(drop=True)
    final_df.loc[final_df["split"] == "val", "split"] = "train"
    train_mask = final_df["split"].to_numpy() == "train"
    train_rows = np.flatnonzero(train_mask)
    weights = None
    if task == "regression" and task_cfg.get("loss_weighting", "uniform") == "polymer_balanced":
        weights = compute_polymer_weights(final_df.loc[train_mask], clip_ratio=float(task_cfg.get("loss_weight_clip_ratio", 10.0)))
    elif task == "classification" and task_cfg.get("loss_weighting", "uniform") == "class_balanced":
        weights = compute_class_weights(y[train_mask])
    final_epochs = final_epoch_budget(task_cfg, best_payload)
    predict_batch_size = int(task_cfg.get("predict_batch_size", 1024))
    finetune_last_layers = int(params.get("finetune_last_layers", 0))
    predictor = None
    checkpoint_extra: Dict[str, Any] = {
        "joint_training": bool(finetune_last_layers > 0),
        "finetune_last_layers": finetune_last_layers,
        "backbone_num_layers": int(backbone_num_layers),
        "final_training_policy": FINAL_TRAINING_POLICY,
        "final_early_stopping_enabled": False,
    }
    if finetune_last_layers > 0:
        if assets is None or raw_inputs is None:
            raise RuntimeError("Backbone fine-tuning requested but view assets/raw inputs are unavailable.")
        predictor, scaler, history, _ = train_eval_joint_model(
            task="classification" if task == "classification" else "regression",
            features=X,
            y=y,
            raw_inputs=raw_inputs,
            train_rows=train_rows,
            val_rows=np.zeros((0,), dtype=np.int64),
            assets=assets,
            aux_features=aux_features,
            hidden_sizes=params["hidden_sizes"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            num_epochs=final_epochs,
            patience=int(task_cfg.get("patience", 20)),
            gradient_clip_norm=float(task_cfg.get("gradient_clip_norm", 1.0)),
            use_scheduler=bool(task_cfg.get("use_scheduler", True)),
            scheduler_min_lr=float(task_cfg.get("scheduler_min_lr", 1e-6)),
            device=device,
            seed=seed,
            objective_metric=str(task_cfg.get("tuning_objective", "balanced_accuracy" if task == "classification" else "val_r2")),
            finetune_last_layers=finetune_last_layers,
            train_weight=weights,
            predict_batch_size=predict_batch_size,
        )
        model = predictor.head
        checkpoint_extra["backbone_state_dict"] = extract_trainable_backbone_state(predictor.backbone)
    else:
        model, scaler, history, _ = train_eval_model(
            task="classification" if task == "classification" else "regression",
            train_X=X[train_mask],
            train_y=y[train_mask],
            val_X=np.zeros((0, X.shape[1]), dtype=np.float32),
            val_y=np.zeros((0,), dtype=np.float32),
            hidden_sizes=params["hidden_sizes"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            num_epochs=final_epochs,
            patience=int(task_cfg.get("patience", 20)),
            gradient_clip_norm=float(task_cfg.get("gradient_clip_norm", 1.0)),
            use_scheduler=bool(task_cfg.get("use_scheduler", True)),
            scheduler_min_lr=float(task_cfg.get("scheduler_min_lr", 1e-6)),
            device=device,
            seed=seed,
            objective_metric=str(task_cfg.get("tuning_objective", "balanced_accuracy" if task == "classification" else "val_r2")),
            train_weight=weights,
            predict_batch_size=predict_batch_size,
        )
        checkpoint_extra["backbone_state_dict"] = {}
    history["final_epoch_budget"] = [final_epochs for _ in history.get("epoch", [])]
    pred_df = final_df.copy()
    metrics_by_split: Dict[str, Dict[str, float]] = {}
    if task == "classification":
        if predictor is not None:
            pred_df["class_prob"] = predict_joint_model(
                predictor,
                raw_inputs or {},
                np.arange(len(final_df), dtype=np.int64),
                view_type=assets.view_type if assets is not None else "sequence",
                device=device,
                task="classification",
                aux_features=aux_features,
                batch_size=predict_batch_size,
            )
        else:
            pred_df["class_prob"] = predict_model(model, scaler, X, device=device, task="classification", batch_size=predict_batch_size)
        pred_df["class_pred"] = (pred_df["class_prob"] >= 0.5).astype(int)
        for split, sub in pred_df.groupby("split"):
            metrics_by_split[split] = classification_metrics(sub["water_miscible"].to_numpy(dtype=int), sub["class_prob"].to_numpy(dtype=float))
    else:
        if predictor is not None:
            pred_df["chi_pred"] = predict_joint_model(
                predictor,
                raw_inputs or {},
                np.arange(len(final_df), dtype=np.int64),
                view_type=assets.view_type if assets is not None else "sequence",
                device=device,
                task="regression",
                aux_features=aux_features,
                batch_size=predict_batch_size,
            )
        else:
            pred_df["chi_pred"] = predict_model(model, scaler, X, device=device, task="regression", batch_size=predict_batch_size)
        pred_df["chi_error"] = pred_df["chi_pred"] - pred_df["chi"]
        for split, sub in pred_df.groupby("split"):
            metrics = regression_metrics(sub["chi"].to_numpy(dtype=float), sub["chi_pred"].to_numpy(dtype=float))
            metrics["poly_nrmse"] = polymer_balanced_nrmse(
                sub["chi"].to_numpy(dtype=float),
                sub["chi_pred"].to_numpy(dtype=float),
                sub["polymer_id"].to_numpy(dtype=int),
                std_floor=float(task_cfg.get("nrmse_std_floor", 0.02)),
                clip=float(task_cfg.get("nrmse_clip", 10.0)),
            )
            metrics_by_split[split] = metrics
    return model, scaler, history, pred_df, metrics_by_split, checkpoint_extra


def save_model_bundle(model: MLP, scaler: StandardScaler, params: Dict[str, Any], path: Path, extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "water_miscible_mlp",
            "model_state_dict": model.to("cpu").state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "params": params,
            **extra,
        },
        path,
    )


def metrics_rows(task_name: str, view: str, metrics_by_split: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    rows = []
    for split, metrics in metrics_by_split.items():
        rows.append({"task": task_name, "view": view, "representation": REPRESENTATION_LABELS.get(view, view), "split": split, **metrics})
    return rows


def sorted_views(values: Iterable[str]) -> List[str]:
    order = {view: idx for idx, view in enumerate(REPRESENTATION_ORDER)}
    return sorted(set(values), key=lambda x: order.get(x, 999))


def read_run_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {"status": "unreadable"}


def collect_existing_metrics(results_dir: Path, tasks: Sequence[str], views: Sequence[str], strict: bool = False) -> pd.DataFrame:
    rows = []
    warnings = []
    for task in tasks:
        for view in views:
            path = results_dir / task / view / "metrics.csv"
            status_path = results_dir / task / view / "run_status.json"
            status = read_run_status(status_path)
            if path.exists():
                if status and status.get("status") != "completed":
                    warnings.append(f"using metrics for {task}/{view}, but run_status={status.get('status')!r}")
                rows.append(pd.read_csv(path))
                continue
            if status:
                warnings.append(f"missing metrics for {task}/{view}; run_status={status.get('status')!r}")
            else:
                warnings.append(f"missing metrics and run_status for {task}/{view}")
    for message in warnings:
        print(f"Warning: {message}.")
    if strict and warnings:
        raise RuntimeError("Strict postprocess failed:\n- " + "\n- ".join(warnings))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_comparison_outputs(
    metrics_df: pd.DataFrame,
    results_dir: Path,
    tasks: Sequence[str],
    dpi: int,
    font_size: int,
    generate_figures: bool,
) -> Path:
    metrics_dir = results_dir / "comparison"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"metrics_{tasks[0]}.csv" if len(tasks) == 1 else "metrics_all.csv"
    out_path = metrics_dir / out_name
    metrics_df.to_csv(out_path, index=False)
    if generate_figures:
        plot_metric_bars(metrics_df, metrics_dir / "figures", dpi=dpi, font_size=font_size)
    return out_path


def plot_rc(font_size: int) -> Dict[str, Any]:
    size = int(font_size)
    return {
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
    }


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(str(metric), str(metric))


def plot_metric_bars(metrics_df: pd.DataFrame, out_dir: Path, dpi: int, font_size: int) -> None:
    if plt is None or metrics_df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    test = metrics_df[metrics_df["split"] == "test"].copy()
    for task, metric in [("chi_regression", "r2"), ("chi_regression", "rmse"), ("chi_regression", "poly_nrmse"), ("water_classification", "balanced_accuracy"), ("water_classification", "auroc"), ("water_classification", "auprc")]:
        sub = test[(test["task"] == task) & (test[metric].notna())].copy() if metric in test.columns else pd.DataFrame()
        if sub.empty:
            continue
        views = sorted_views(sub["view"].astype(str).tolist())
        sub = sub.set_index("view").loc[views].reset_index()
        labels = [PLOT_VIEW_LABELS.get(v, v) for v in sub["view"]]
        colors = [REPRESENTATION_COLORS.get(v, "#888888") for v in sub["view"]]
        with plt.rc_context(plot_rc(font_size)):
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.bar(np.arange(len(sub)), pd.to_numeric(sub[metric], errors="coerce"), color=colors)
            ax.set_xticks(np.arange(len(sub)))
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylabel(metric_label(metric))
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"{task}_test_{metric}_by_view.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def plot_parity(pred_df: pd.DataFrame, out_path: Path, dpi: int, font_size: int) -> None:
    if plt is None or pred_df.empty:
        return
    test = pred_df[pred_df["split"] == "test"]
    if test.empty:
        return
    with plt.rc_context(plot_rc(font_size)):
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.scatter(test["chi"], test["chi_pred"], s=18, alpha=0.75, color="#4E79A7")
        lo = float(min(test["chi"].min(), test["chi_pred"].min()))
        hi = float(max(test["chi"].max(), test["chi_pred"].max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        ax.set_xlabel("True chi")
        ax.set_ylabel("Pred chi")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_class_scores(pred_df: pd.DataFrame, out_path: Path, dpi: int, font_size: int) -> None:
    if plt is None or pred_df.empty:
        return
    test = pred_df[pred_df["split"] == "test"].copy()
    if test.empty:
        return
    with plt.rc_context(plot_rc(font_size)):
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        rng = np.random.default_rng(0)
        x = test["water_miscible"].to_numpy(dtype=float) + rng.normal(0.0, 0.03, len(test))
        ax.scatter(x, test["class_prob"], s=24, alpha=0.75, color="#59A14F")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Immisc.", "Misc."])
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Class")
        ax.set_ylabel("P(miscible)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_hpo_progress(trials: pd.DataFrame, out_path: Path, dpi: int, font_size: int) -> None:
    if plt is None or trials.empty or "objective_value" not in trials.columns:
        return
    df = trials.copy()
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
    df["objective_value"] = pd.to_numeric(df["objective_value"], errors="coerce")
    df = df.dropna(subset=["trial", "objective_value"]).sort_values("trial")
    if df.empty:
        return
    direction = str(df["objective_direction"].iloc[0])
    trial_x = df["trial"].to_numpy(dtype=float)
    vals = df["objective_value"].to_numpy(dtype=float)
    best = np.minimum.accumulate(vals) if direction == "minimize" else np.maximum.accumulate(vals)
    with plt.rc_context(plot_rc(font_size)):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(trial_x, vals, "o", alpha=0.65, label="trial")
        ax.plot(trial_x, best, "-", linewidth=2.0, label="best")
        ax.set_xlabel("Trial")
        ax.set_ylabel(metric_label(str(df["objective_metric"].iloc[0])))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def run_task_for_view(
    *,
    task_name: str,
    task_kind: str,
    view: str,
    split_df: pd.DataFrame,
    embeddings: np.ndarray,
    raw_inputs: Dict[str, np.ndarray],
    assets: Optional[ViewAssets],
    view_info: ViewInfo,
    task_cfg: Dict[str, Any],
    split_mode: str,
    results_dir: Path,
    device: str,
    seed: int,
    force_tune: Optional[bool],
    dpi: int,
    font_size: int,
    run_context: Optional[Mapping[str, Any]] = None,
    config_snapshot: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if task_kind == "regression":
        X = chi_feature_matrix(split_df, embeddings)
        aux_features = split_df[["temperature", "phi"]].to_numpy(dtype=np.float32)
        y = split_df["chi"].to_numpy(dtype=np.float32)
    else:
        X = embeddings.astype(np.float32, copy=False)
        aux_features = None
        y = split_df["water_miscible"].to_numpy(dtype=np.float32)
    backbone_num_layers = get_backbone_num_layers(assets.backbone) if assets is not None else int(view_info.backbone_num_layers)
    checkpoint_path = assets.checkpoint_path if assets is not None else view_info.checkpoint_path

    task_dir = results_dir / task_name / view
    task_dir.mkdir(parents=True, exist_ok=True)
    if config_snapshot is not None:
        save_yaml(config_snapshot, task_dir / "config_used.yaml")
    save_json(
        {
            "task": task_name,
            "task_kind": task_kind,
            "view": view,
            "run_context": dict(run_context or {}),
            "config_used_path": repo_relative(task_dir / "config_used.yaml") if config_snapshot is not None else None,
        },
        task_dir / "run_metadata.json",
    )
    task_started_perf = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_started_at = utc_now()
    save_json(
        {
            "status": "running",
            "task": task_name,
            "task_kind": task_kind,
            "view": view,
            "started_at_utc": run_started_at,
            "run_context": dict(run_context or {}),
            "note": "During this state, checkpoint.pt may still be from an earlier completed run.",
        },
        task_dir / "run_status.json",
    )
    hpo_started_perf = time.perf_counter()
    params, trials, best_payload = tune_or_select_params(
        task=task_kind,
        split_df=split_df,
        X=X,
        y=y,
        task_cfg=task_cfg,
        split_mode=split_mode,
        seed=seed,
        device=device,
        force_tune=force_tune,
        raw_inputs=raw_inputs,
        assets=assets,
        aux_features=aux_features,
        backbone_num_layers=backbone_num_layers,
        trial_log_path=task_dir / "optuna_trials.csv",
    )
    hpo_seconds = time.perf_counter() - hpo_started_perf
    trials.to_csv(task_dir / "optuna_trials.csv", index=False)
    save_json(best_payload, task_dir / "optuna_best.json")
    final_started_perf = time.perf_counter()
    model, scaler, history, pred_df, metrics_by_split, checkpoint_extra = train_final_and_predict(
        task=task_kind,
        split_df=split_df,
        X=X,
        y=y,
        task_cfg=task_cfg,
        params=params,
        best_payload=best_payload,
        device=device,
        seed=seed + 77,
        raw_inputs=raw_inputs,
        assets=assets,
        aux_features=aux_features,
        backbone_num_layers=backbone_num_layers,
    )
    final_training_seconds = time.perf_counter() - final_started_perf
    artifact_started_perf = time.perf_counter()
    timings = {
        "hpo_seconds": float(hpo_seconds),
        "final_training_seconds": float(final_training_seconds),
    }
    pd.DataFrame(history).to_csv(task_dir / "training_history.csv", index=False)
    pred_df.to_csv(task_dir / "predictions.csv", index=False)
    pd.DataFrame(metrics_rows(task_name, view, metrics_by_split)).to_csv(task_dir / "metrics.csv", index=False)
    save_json(
        {
            "chosen_hyperparameters": params,
            "final_epoch_budget": int(history["final_epoch_budget"][0]) if history.get("final_epoch_budget") else None,
            "final_training_policy": FINAL_TRAINING_POLICY,
            "final_early_stopping_enabled": False,
            "metrics_by_split": metrics_by_split,
            "backbone_num_layers": int(backbone_num_layers),
            "timings": timings,
            "resources": runtime_resources(),
            "run_context": dict(run_context or {}),
        },
        task_dir / "summary.json",
    )
    completed_at_utc = utc_now()
    save_model_bundle(
        model,
        scaler,
        params,
        task_dir / "checkpoint.pt",
        {
            "task": task_name,
            "task_kind": task_kind,
            "view": view,
            "input_dim": int(X.shape[1]),
            "backbone_num_layers": int(backbone_num_layers),
            "backbone_checkpoint_path": repo_relative(checkpoint_path),
            "started_at_utc": run_started_at,
            "completed_at_utc": completed_at_utc,
            "timings": timings,
            "resources": runtime_resources(),
            **checkpoint_extra,
        },
    )
    artifact_seconds = time.perf_counter() - artifact_started_perf
    timings["artifact_seconds"] = float(artifact_seconds)
    timings["total_seconds"] = float(time.perf_counter() - task_started_perf)
    save_json(
        {
            "status": "completed",
            "task": task_name,
            "task_kind": task_kind,
            "view": view,
            "started_at_utc": run_started_at,
            "completed_at_utc": completed_at_utc,
            "checkpoint_path": repo_relative(task_dir / "checkpoint.pt"),
            "timings": timings,
            "resources": runtime_resources(),
            "run_context": dict(run_context or {}),
        },
        task_dir / "run_status.json",
    )
    if task_kind == "regression":
        plot_parity(pred_df, task_dir / "figures" / "test_parity.png", dpi=dpi, font_size=font_size)
    else:
        plot_class_scores(pred_df, task_dir / "figures" / "test_class_scores.png", dpi=dpi, font_size=font_size)
    plot_hpo_progress(trials, task_dir / "figures" / "hpo_progress.png", dpi=dpi, font_size=font_size)
    return metrics_rows(task_name, view, metrics_by_split)


def write_failed_run_status(
    *,
    task_dir: Path,
    task_name: str,
    task_kind: str,
    view: str,
    exc: BaseException,
    run_context: Optional[Mapping[str, Any]] = None,
) -> None:
    previous = read_run_status(task_dir / "run_status.json")
    payload = {
        "status": "failed",
        "task": task_name,
        "task_kind": task_kind,
        "view": view,
        "started_at_utc": previous.get("started_at_utc"),
        "completed_at_utc": utc_now(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "run_context": dict(run_context or previous.get("run_context") or {}),
        "resources": runtime_resources(),
    }
    save_json(payload, task_dir / "run_status.json")


def run_task_for_view_checked(**kwargs) -> List[Dict[str, Any]]:
    task_name = str(kwargs["task_name"])
    task_kind = str(kwargs["task_kind"])
    view = str(kwargs["view"])
    task_dir = Path(kwargs["results_dir"]) / task_name / view
    try:
        return run_task_for_view(**kwargs)
    except Exception as exc:
        write_failed_run_status(
            task_dir=task_dir,
            task_name=task_name,
            task_kind=task_kind,
            view=view,
            exc=exc,
            run_context=kwargs.get("run_context"),
        )
        raise


def run(args) -> None:
    config_path = resolve_path(args.config)
    config = apply_model_size_override(load_yaml(config_path), args.model_size)
    if args.no_tune and args.tune:
        raise ValueError("--tune and --no_tune cannot both be set.")
    force_tune = True if args.tune else False if args.no_tune else None
    selected_tasks = normalize_task_names(args.tasks)
    views = [str(v).strip() for v in (args.views.split(",") if args.views else config.get("views", {}).get("enabled", [])) if str(v).strip()]
    seed = int(config.get("data", {}).get("random_seed", 42))
    set_seed(seed)
    results_dir = resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_context = make_run_context(args, config_path, results_dir, selected_tasks, views)
    if not args.skip_postprocess and not args.postprocess_only and not args.precompute_embeddings_only:
        save_yaml(config, results_dir / "config_used.yaml")
    dpi = int(config.get("plotting", {}).get("dpi", 600))
    font_size = int(config.get("plotting", {}).get("font_size", 16))

    if args.postprocess_only:
        metrics_df = collect_existing_metrics(results_dir, selected_tasks, views, strict=bool(args.strict_postprocess))
        out_path = write_comparison_outputs(
            metrics_df,
            results_dir,
            selected_tasks,
            dpi=dpi,
            font_size=font_size,
            generate_figures=bool(config.get("plotting", {}).get("generate_figures", True)),
        )
        print(f"Saved postprocess metrics to {out_path}")
        return

    device = device_from_config(config)
    print(f"Using device={device}")
    print(f"Writing results to {results_dir}")

    max_rows = int(args.max_rows) if args.max_rows is not None else None
    data_cfg = config.get("data", {})
    run_chi = "chi_regression" in selected_tasks and config.get("chi_regression", {}).get("enabled", True)
    run_water = "water_classification" in selected_tasks and config.get("water_classification", {}).get("enabled", True)
    chi_split = None
    water_split = None
    if run_chi:
        chi_df = load_chi_dataset(config, max_rows=max_rows)
        chi_split = make_splits(
            chi_df,
            split_mode=str(data_cfg.get("split_mode", "polymer")),
            test_ratio=float(data_cfg.get("holdout_test_ratio", 0.1)),
            cv_folds=int(config.get("chi_regression", {}).get("tuning_cv_folds", 5)),
            seed=seed,
            stratify_target="chi",
        )
        if not args.skip_postprocess and not args.precompute_embeddings_only:
            chi_split.to_csv(results_dir / "chi_dataset_with_split.csv", index=False)
    if run_water:
        water_df = load_water_dataset(config, max_rows=max_rows)
        water_split = make_splits(
            water_df,
            split_mode=str(data_cfg.get("classification_split_mode", "random")),
            test_ratio=float(data_cfg.get("holdout_test_ratio", 0.1)),
            cv_folds=int(config.get("water_classification", {}).get("tuning_cv_folds", 5)),
            seed=seed,
            stratify_target="water_miscible",
        )
        if not args.skip_postprocess and not args.precompute_embeddings_only:
            water_split.to_csv(results_dir / "water_dataset_with_split.csv", index=False)

    all_metrics: List[Dict[str, Any]] = []
    cache_dir = results_dir / "embedding_cache"
    cache_tag = f"_{args.cache_tag.strip()}" if args.cache_tag and args.cache_tag.strip() else ""

    for view in views:
        print(f"Preparing view={view}")
        view_info = get_view_info(config, view)
        smiles_sources = []
        if run_chi and chi_split is not None:
            smiles_sources.extend(chi_split["SMILES"].astype(str).tolist())
        if run_water and water_split is not None:
            smiles_sources.extend(water_split["SMILES"].astype(str).tolist())
        all_smiles = sorted(set(smiles_sources))
        cache_path = cache_dir / f"{view}{cache_tag}_embeddings.npz"
        raw_cache_path = cache_dir / f"{view}{cache_tag}_raw_inputs.npz"
        requested = {str(smi).strip() for smi in all_smiles if str(smi).strip()}
        cache_metadata = embedding_cache_metadata_from_info(view_info)
        raw_cache_metadata = raw_input_cache_metadata_from_info(view_info)
        embedding_map = load_embedding_cache(cache_path, requested, expected_metadata=cache_metadata) if not args.fresh_embeddings else None
        raw_map = load_raw_input_cache(raw_cache_path, requested, expected_metadata=raw_cache_metadata) if not args.fresh_embeddings else None

        joint_possible = False
        if run_chi:
            joint_possible = joint_possible or task_allows_backbone_finetune(config["chi_regression"], view_info.backbone_num_layers, force_tune)
        if run_water:
            joint_possible = joint_possible or task_allows_backbone_finetune(config["water_classification"], view_info.backbone_num_layers, force_tune)

        assets: Optional[ViewAssets] = None
        if embedding_map is None or raw_map is None:
            print(f"Loading view assets for cache build: {view}")
            assets = load_view_assets(config, view, device=device)
            embedding_map, raw_map = build_embedding_and_raw_caches(
                smiles_values=all_smiles,
                assets=assets,
                embedding_cache_path=cache_path,
                raw_cache_path=raw_cache_path,
                device=device,
                force_rebuild=bool(args.fresh_embeddings),
            )
        else:
            print(f"Loaded cached embeddings for view={view}: {cache_path}")
            print(f"Loaded cached raw inputs for view={view}: {raw_cache_path}")
        if args.precompute_embeddings_only:
            print(f"Precomputed embeddings for view={view}: {cache_path}")
            print(f"Precomputed raw inputs for view={view}: {raw_cache_path}")
            if assets is not None:
                del assets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        if assets is None and joint_possible:
            print(f"Loading view assets for backbone fine-tuning: {view}")
            assets = load_view_assets(config, view, device=device)
        chi_view_df = water_view_df = None
        chi_embeddings = water_embeddings = None
        chi_raw_inputs = water_raw_inputs = None
        if run_chi and chi_split is not None:
            chi_view_df, chi_embeddings, chi_raw_inputs = align_embeddings_and_raw(chi_split, embedding_map, raw_map, assets or view_info)
        if run_water and water_split is not None:
            water_view_df, water_embeddings, water_raw_inputs = align_embeddings_and_raw(water_split, embedding_map, raw_map, assets or view_info)
        print_rows = []
        if chi_view_df is not None:
            print_rows.append(f"chi_rows={len(chi_view_df)}/{len(chi_split)}")
        if water_view_df is not None:
            print_rows.append(f"water_rows={len(water_view_df)}/{len(water_split)}")
        print(f"view={view}: " + ", ".join(print_rows))
        if run_chi and chi_view_df is not None and chi_embeddings is not None and chi_raw_inputs is not None and len(chi_view_df) > 0:
            all_metrics.extend(
                run_task_for_view_checked(
                    task_name="chi_regression",
                    task_kind="regression",
                    view=view,
                    split_df=chi_view_df,
                    embeddings=chi_embeddings,
                    raw_inputs=chi_raw_inputs,
                    assets=assets,
                    view_info=view_info,
                    task_cfg=config["chi_regression"],
                    split_mode=str(data_cfg.get("split_mode", "polymer")),
                    results_dir=results_dir,
                    device=device,
                    seed=stable_seed(seed, view, "chi"),
                    force_tune=force_tune,
                    dpi=dpi,
                    font_size=font_size,
                    run_context=run_context,
                    config_snapshot=config,
                )
            )
        if run_water and water_view_df is not None and water_embeddings is not None and water_raw_inputs is not None and len(water_view_df) > 0:
            all_metrics.extend(
                run_task_for_view_checked(
                    task_name="water_classification",
                    task_kind="classification",
                    view=view,
                    split_df=water_view_df,
                    embeddings=water_embeddings,
                    raw_inputs=water_raw_inputs,
                    assets=assets,
                    view_info=view_info,
                    task_cfg=config["water_classification"],
                    split_mode=str(data_cfg.get("classification_split_mode", "random")),
                    results_dir=results_dir,
                    device=device,
                    seed=stable_seed(seed, view, "water"),
                    force_tune=force_tune,
                    dpi=dpi,
                    font_size=font_size,
                    run_context=run_context,
                    config_snapshot=config,
                )
            )
        if assets is not None:
            del assets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.precompute_embeddings_only:
        print("Embedding precompute complete.")
        return
    if args.skip_postprocess:
        print("Skipped aggregate postprocess.")
        return
    metrics_df = pd.DataFrame(all_metrics)
    out_path = write_comparison_outputs(
        metrics_df,
        results_dir,
        selected_tasks,
        dpi=dpi,
        font_size=font_size,
        generate_figures=bool(config.get("plotting", {}).get("generate_figures", True)),
    )
    print(f"Saved aggregate metrics to {out_path}")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="water_miscible/configs/config_water.yaml")
    parser.add_argument("--model_size", choices=sorted(MODEL_SIZES), default=None, help="Override backbone checkpoint size and write to results_<size>.")
    parser.add_argument("--views", default=None, help="Comma-separated subset of views.")
    parser.add_argument("--tasks", default=None, help="Comma-separated subset: chi_regression,water_classification.")
    parser.add_argument("--tune", action="store_true", help="Force Optuna tuning on.")
    parser.add_argument("--no_tune", action="store_true", help="Force Optuna tuning off.")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional small-row debug cap for each dataset.")
    parser.add_argument("--fresh_embeddings", action="store_true", help="Rebuild embedding caches even if cache files exist.")
    parser.add_argument("--cache_tag", default="", help="Optional suffix for embedding cache files.")
    parser.add_argument("--precompute_embeddings_only", action="store_true", help="Build embedding caches and exit before head training.")
    parser.add_argument("--skip_postprocess", action="store_true", help="Do not aggregate comparison metrics after training.")
    parser.add_argument("--postprocess_only", action="store_true", help="Aggregate existing per-view metrics without training.")
    parser.add_argument("--strict_postprocess", action="store_true", help="Fail postprocess if an expected task/view is missing or not completed.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
