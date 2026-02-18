#!/usr/bin/env python
"""F2: Retrieval evaluation across views."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.utils.config import load_config, save_config
from src.evaluation.retrieval_metrics import compute_recall_at_k
from typing import Optional

from src.model.multi_view_model import MultiViewModel


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _to_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid integer sample cap.")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _load_view_embeddings(results_dir: Path, view: str, dataset: str):
    emb_path = results_dir / f"embeddings_{view}_{dataset}.npy"
    if not emb_path.exists() and view == "smiles":
        legacy = results_dir / f"embeddings_{dataset}.npy"
        emb_path = legacy if legacy.exists() else emb_path
    if not emb_path.exists():
        return None
    return np.load(emb_path)


def _load_view_index(results_dir: Path, view: str):
    idx_path = results_dir / f"embedding_index_{view}.csv"
    if not idx_path.exists():
        return None
    return pd.read_csv(idx_path)


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


def main(args):
    config = load_config(args.config)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, results_dir / "config_used.yaml")

    views = config.get("alignment_views", ["smiles"])
    ks = config.get("evaluation", {}).get("recall_ks", [1, 5, 10])
    max_eval_samples = _to_int_or_none(config.get("evaluation", {}).get("max_samples_per_dataset"))

    view_data = {}
    view_dims = {}
    for view in views:
        idx_df = _load_view_index(results_dir, view)
        if idx_df is None:
            continue
        view_data[view] = {}
        for dataset in ["d1", "d2"]:
            emb = _load_view_embeddings(results_dir, view, dataset)
            if emb is None:
                continue
            subset = idx_df[idx_df["dataset"] == dataset]
            ids = subset.sort_values("row_index")["polymer_id"].astype(str).tolist()
            if max_eval_samples is not None and len(ids) > max_eval_samples:
                ids = ids[:max_eval_samples]
                emb = emb[:max_eval_samples]
            view_data[view][dataset] = {"embeddings": emb, "ids": ids}
            view_dims[view] = emb.shape[1]

    alignment_model = None
    if args.use_alignment:
        alignment_model = _load_alignment_model(results_dir, view_dims, config, args.alignment_checkpoint)
        if alignment_model is None:
            raise FileNotFoundError("Alignment checkpoint not found for --use_alignment")

    if alignment_model is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for view, datasets in view_data.items():
            for dataset in datasets:
                datasets[dataset]["embeddings"] = _project_embeddings(
                    alignment_model,
                    view,
                    datasets[dataset]["embeddings"],
                    device=device,
                )

    rows = []
    for dataset in ["d1", "d2"]:
        available_views = [v for v in views if dataset in view_data.get(v, {})]
        if available_views:
            sample_sizes = {v: len(view_data[v][dataset]["ids"]) for v in available_views}
            print(f"Retrieval dataset={dataset} views={available_views} sample_sizes={sample_sizes}")
        for src_view in available_views:
            for tgt_view in available_views:
                src = view_data[src_view][dataset]
                tgt = view_data[tgt_view][dataset]
                metrics = compute_recall_at_k(
                    src["embeddings"],
                    tgt["embeddings"],
                    src["ids"],
                    tgt["ids"],
                    ks,
                )
                rows.append({
                    "view_pair": f"{dataset}_{src_view}->{dataset}_{tgt_view}",
                    "view_dropout_mode": "aligned" if alignment_model is not None else "none",
                    **metrics,
                })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(results_dir / "metrics_alignment.csv", index=False)
    print(f"Saved metrics_alignment.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--use_alignment", action="store_true")
    parser.add_argument("--alignment_checkpoint", type=str, default=None)
    main(parser.parse_args())
