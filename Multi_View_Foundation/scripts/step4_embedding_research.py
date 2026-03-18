#!/usr/bin/env python
"""F4: Embedding research for multi-view polymer representations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.view_compare import _match_polymer_class
from src.utils.config import load_config, save_config
from src.utils.embedding_artifacts import load_view_embeddings, load_view_index, load_view_meta
from src.utils.output_layout import ensure_step_dirs, save_csv, save_json
from src.utils.property_names import ordered_properties, property_column_candidates
from src.utils.runtime import (
    load_module as _shared_load_module,
    resolve_path as _shared_resolve_path,
    resolve_with_base as _shared_resolve_with_base,
    to_bool as _to_bool,
)
from src.utils.visualization import (
    ordered_views,
    save_figure_png,
    set_publication_style,
    view_color,
    view_label,
)

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


SEQUENCE_VIEW_SPECS = {
    "smiles": {
        "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SMILES" / "src" / "data" / "tokenizer.py",
        "tokenizer_class": "PSmilesTokenizer",
        "tokenizer_file": "tokenizer.json",
    },
    "smiles_bpe": {
        "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SMILES_BPE" / "src" / "data" / "tokenizer.py",
        "tokenizer_class": "PSmilesTokenizer",
        "tokenizer_file": "tokenizer.json",
    },
    "selfies": {
        "tokenizer_module": REPO_ROOT / "Bi_Diffusion_SELFIES" / "src" / "data" / "selfies_tokenizer.py",
        "tokenizer_class": "SelfiesTokenizer",
        "tokenizer_file": "tokenizer.json",
    },
    "group_selfies": {
        "tokenizer_module": REPO_ROOT / "Bi_Diffusion_Group_SELFIES" / "src" / "data" / "tokenizer.py",
        "tokenizer_class": "GroupSELFIESTokenizer",
        "tokenizer_file": "tokenizer.pkl",
    },
}


def _resolve_path(path_str: str) -> Path:
    return _shared_resolve_path(path_str, BASE_DIR)


def _resolve_with_base(path_str: str, base_dir: Path) -> Path:
    return _shared_resolve_with_base(path_str, base_dir)


def _load_module(module_name: str, path: Path):
    return _shared_load_module(module_name, path, REPO_ROOT)


def _load_step_csv(results_dir: Path, step_name: str, filename: str) -> pd.DataFrame:
    candidates = [
        results_dir / step_name / "metrics" / filename,
        results_dir / step_name / "files" / filename,
        results_dir / step_name / filename,
        results_dir / filename,
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _load_paired_index(results_dir: Path, config: dict) -> pd.DataFrame:
    candidates = [
        results_dir / "step0_paired_dataset" / "files" / "paired_index.csv",
        _resolve_path(config.get("paths", {}).get("paired_index", str(results_dir / "paired_index.csv"))),
        results_dir / "paired_index.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("paired_index.csv not found. Run F0 first.")


def _load_method_config(encoder_cfg: dict) -> tuple[Path, dict]:
    method_dir = _resolve_path(encoder_cfg.get("method_dir"))
    config_path = encoder_cfg.get("config_path")
    if config_path:
        config_path = _resolve_path(config_path)
    else:
        config_path = method_dir / "configs" / "config.yaml"
    method_cfg = load_config(str(config_path))
    return method_dir, method_cfg


def _load_sequence_tokenizer(view: str, encoder_cfg: dict):
    spec = SEQUENCE_VIEW_SPECS[view]
    method_dir, method_cfg = _load_method_config(encoder_cfg)
    scales_mod = _load_module(f"scales_{method_dir.name}_f4_{view}", method_dir / "src" / "utils" / "model_scales.py")
    tokenizer_mod = _load_module(f"tokenizer_{method_dir.name}_f4_{view}", spec["tokenizer_module"])
    get_results_dir = scales_mod.get_results_dir
    TokenizerCls = getattr(tokenizer_mod, spec["tokenizer_class"])

    model_size = encoder_cfg.get("model_size")
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
        tokenizer_path = results_dir / spec["tokenizer_file"]
        if not tokenizer_path.exists():
            tokenizer_path = base_results_dir / spec["tokenizer_file"]
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found for view={view}: {tokenizer_path}")
    return TokenizerCls.load(str(tokenizer_path))


def _load_graph_tokenizer(encoder_cfg: dict):
    method_dir, method_cfg = _load_method_config(encoder_cfg)
    scales_mod = _load_module(f"graph_scales_{method_dir.name}_f4", method_dir / "src" / "utils" / "model_scales.py")
    tokenizer_mod = _load_module(f"graph_tokenizer_{method_dir.name}_f4", method_dir / "src" / "data" / "graph_tokenizer.py")
    get_results_dir = scales_mod.get_results_dir
    GraphTokenizer = tokenizer_mod.GraphTokenizer

    model_size = encoder_cfg.get("model_size")
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
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Graph tokenizer not found: {tokenizer_path}")
    return GraphTokenizer.load(str(tokenizer_path))


def _active_views(config: dict) -> list[str]:
    raw = config.get("alignment_views", []) or []
    enabled_cfg = config.get("views", {}) or {}
    views = []
    for view in raw:
        token = str(view).strip()
        if not token:
            continue
        if not _to_bool((enabled_cfg.get(token) or {}).get("enabled", True), True):
            continue
        views.append(token)
    return ordered_views(views)


def _align_ids_and_embeddings(idx_df: pd.DataFrame, embeddings: np.ndarray, dataset: str) -> tuple[list[str], np.ndarray]:
    subset = idx_df[idx_df["dataset"].astype(str).str.lower() == str(dataset).strip().lower()].copy()
    if subset.empty:
        return [], np.zeros((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32)
    subset["_row_index"] = pd.to_numeric(subset["row_index"], errors="coerce").astype(np.int64)
    subset = subset.sort_values("_row_index")
    row_idx = subset["_row_index"].to_numpy(dtype=np.int64)
    if len(row_idx) != int(embeddings.shape[0]):
        raise ValueError(
            f"Embedding/index length mismatch for dataset={dataset}: index_rows={len(row_idx)} embedding_rows={embeddings.shape[0]}"
        )
    ids = subset["polymer_id"].astype(str).tolist()
    return ids, embeddings[row_idx]


def _subsample_indices(n_items: int, cap: Optional[int], seed: int) -> np.ndarray:
    if cap is None or cap <= 0 or n_items <= cap:
        return np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n_items, size=int(cap), replace=False))
    return chosen.astype(np.int64)


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, a_min=1e-9, a_max=None)


def _effective_rank_and_participation(embeddings: np.ndarray) -> tuple[float, float]:
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        return np.nan, np.nan
    x = x - x.mean(axis=0, keepdims=True)
    try:
        singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    except Exception:
        return np.nan, np.nan
    power = np.square(singular.astype(np.float64))
    power = power[np.isfinite(power) & (power > 0)]
    if power.size == 0:
        return np.nan, np.nan
    probs = power / power.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    effective_rank = float(np.exp(entropy))
    dim = float(x.shape[1])
    participation = float((power.sum() ** 2) / max(np.square(power).sum(), 1e-12))
    return effective_rank / max(dim, 1.0), participation / max(dim, 1.0)


def _hubness_score(embeddings: np.ndarray, k: int) -> float:
    x = np.asarray(embeddings, dtype=np.float32)
    n = int(x.shape[0])
    if n <= 3:
        return np.nan
    neigh_k = max(2, min(int(k) + 1, n))
    try:
        knn = NearestNeighbors(n_neighbors=neigh_k, metric="cosine")
        knn.fit(x)
        indices = knn.kneighbors(return_distance=False)[:, 1:]
    except Exception:
        return np.nan
    if indices.size == 0:
        return np.nan
    counts = np.bincount(indices.reshape(-1), minlength=n).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.nan
    top_n = max(1, int(math.ceil(0.01 * n)))
    return float(np.sort(counts)[-top_n:].sum() / total)


def _class_separation(embeddings: np.ndarray, smiles_list: Iterable[str], min_class_samples: int) -> float:
    labels = pd.Series(list(smiles_list), dtype=str).map(_match_polymer_class)
    counts = labels.value_counts()
    keep = counts[counts >= int(min_class_samples)].index.tolist()
    if len(keep) < 2:
        return np.nan
    mask = labels.isin(keep).to_numpy()
    if int(mask.sum()) < max(10, int(min_class_samples) * 2):
        return np.nan
    try:
        return float(silhouette_score(np.asarray(embeddings)[mask], labels[mask].astype(str).to_numpy(), metric="cosine"))
    except Exception:
        return np.nan


def _resolve_property_columns(df: pd.DataFrame, property_name: str) -> tuple[str, str]:
    smiles_col = "SMILES" if "SMILES" in df.columns else "p_smiles" if "p_smiles" in df.columns else ""
    if not smiles_col:
        return "", ""
    for col in property_column_candidates(property_name):
        if col in df.columns:
            return smiles_col, col
    return "", ""


def _load_property_table(prop_dir: Path, property_name: str) -> pd.DataFrame:
    for candidate in [prop_dir / f"{property_name}.csv"]:
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.DataFrame()


def _property_smoothness(
    embeddings: np.ndarray,
    smiles_lookup: pd.Series,
    property_values: pd.Series,
    k: int,
) -> float:
    joined = pd.DataFrame({"smiles": smiles_lookup.astype(str), "value": smiles_lookup.astype(str).map(property_values.to_dict())})
    mask = joined["value"].notna().to_numpy()
    if int(mask.sum()) <= max(10, int(k) + 2):
        return np.nan
    x = np.asarray(embeddings)[mask]
    y = pd.to_numeric(joined.loc[mask, "value"], errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(y)
    if int(valid.sum()) <= max(10, int(k) + 2):
        return np.nan
    x = x[valid]
    y = y[valid]
    neigh_k = max(2, min(int(k) + 1, len(y)))
    try:
        knn = NearestNeighbors(n_neighbors=neigh_k, metric="cosine")
        knn.fit(x)
        indices = knn.kneighbors(return_distance=False)[:, 1:]
    except Exception:
        return np.nan
    if indices.size == 0:
        return np.nan
    preds = np.asarray([float(np.mean(y[row])) if len(row) else np.nan for row in indices], dtype=np.float32)
    finite = np.isfinite(preds) & np.isfinite(y)
    if int(finite.sum()) <= max(5, int(k)):
        return np.nan
    try:
        return float(r2_score(y[finite], preds[finite]))
    except Exception:
        return np.nan


def _representation_from_view(view: str) -> str:
    mapping = {
        "smiles": "SMILES",
        "smiles_bpe": "SMILES_BPE",
        "selfies": "SELFIES",
        "group_selfies": "Group_SELFIES",
        "graph": "Graph",
        "all": "MultiViewMean",
    }
    return mapping.get(str(view).strip().lower(), str(view))


def _view_from_representation(value: object) -> str:
    token = str(value).strip().lower().replace("-", "_")
    mapping = {
        "smiles": "smiles",
        "smiles_bpe": "smiles_bpe",
        "selfies": "selfies",
        "group_selfies": "group_selfies",
        "group_selfie": "group_selfies",
        "graph": "graph",
        "multiviewmean": "all",
        "multiview_mean": "all",
    }
    return mapping.get(token, token)


def _summarize_retrieval(metrics_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["view", "mean_recall_at_1", "mean_recall_at_5", "mean_recall_at_10", "mean_match_rate", "num_pairs"])
    rows = []
    parsed_rows = []
    for _, row in metrics_df.iterrows():
        pair = str(row.get("view_pair", ""))
        if "->" not in pair:
            continue
        left, right = pair.split("->", 1)
        if "_" not in left or "_" not in right:
            continue
        src_dataset, src_view = left.split("_", 1)
        tgt_dataset, tgt_view = right.split("_", 1)
        if src_dataset != tgt_dataset or src_dataset != str(dataset).strip().lower():
            continue
        if src_view == tgt_view:
            continue
        parsed_rows.append(
            {
                "src_view": src_view,
                "tgt_view": tgt_view,
                "recall_at_1": pd.to_numeric(pd.Series([row.get("recall_at_1")]), errors="coerce").iloc[0],
                "recall_at_5": pd.to_numeric(pd.Series([row.get("recall_at_5")]), errors="coerce").iloc[0],
                "recall_at_10": pd.to_numeric(pd.Series([row.get("recall_at_10")]), errors="coerce").iloc[0],
                "match_rate": pd.to_numeric(pd.Series([row.get("match_rate")]), errors="coerce").iloc[0],
            }
        )
    parsed_df = pd.DataFrame(parsed_rows)
    if parsed_df.empty:
        return pd.DataFrame(columns=["view", "mean_recall_at_1", "mean_recall_at_5", "mean_recall_at_10", "mean_match_rate", "num_pairs"])
    for view in ordered_views(list(parsed_df["src_view"]) + list(parsed_df["tgt_view"])):
        mask = (parsed_df["src_view"] == view) | (parsed_df["tgt_view"] == view)
        sub = parsed_df.loc[mask]
        rows.append(
            {
                "view": view,
                "mean_recall_at_1": float(pd.to_numeric(sub["recall_at_1"], errors="coerce").mean()),
                "mean_recall_at_5": float(pd.to_numeric(sub["recall_at_5"], errors="coerce").mean()),
                "mean_recall_at_10": float(pd.to_numeric(sub["recall_at_10"], errors="coerce").mean()),
                "mean_match_rate": float(pd.to_numeric(sub["match_rate"], errors="coerce").mean()),
                "num_pairs": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_property_heads(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if metrics_df.empty:
        return (
            pd.DataFrame(columns=["view", "mean_test_r2", "mean_test_rmse", "win_rate"]),
            {"fusion_gain_mean": np.nan},
        )
    df = metrics_df.copy()
    if "split" not in df.columns or "representation" not in df.columns:
        return (
            pd.DataFrame(columns=["view", "mean_test_r2", "mean_test_rmse", "win_rate"]),
            {"fusion_gain_mean": np.nan},
        )
    df = df[df["split"].astype(str).str.lower() == "test"].copy()
    if df.empty:
        return (
            pd.DataFrame(columns=["view", "mean_test_r2", "mean_test_rmse", "win_rate"]),
            {"fusion_gain_mean": np.nan},
        )
    df["view"] = df["representation"].map(_view_from_representation)
    rows = []
    single_df = df[df["view"] != "all"].copy()
    if not single_df.empty:
        grouped = single_df.groupby("view", dropna=False)
        for view, sub in grouped:
            rows.append(
                {
                    "view": view,
                    "mean_test_r2": float(pd.to_numeric(sub["r2"], errors="coerce").mean()),
                    "mean_test_rmse": float(pd.to_numeric(sub["rmse"], errors="coerce").mean()),
                    "win_rate": 0.0,
                }
            )
        wins = {}
        for prop, sub in single_df.groupby("property", dropna=False):
            scores = pd.to_numeric(sub["r2"], errors="coerce")
            if not scores.notna().any():
                continue
            best = float(scores.max())
            winners = sub.loc[np.isclose(scores.to_numpy(dtype=float), best, equal_nan=False), "view"].astype(str).tolist()
            if not winners:
                continue
            share = 1.0 / max(len(winners), 1)
            for winner in winners:
                wins[winner] = wins.get(winner, 0.0) + share
        denom = max(len(single_df["property"].dropna().unique()), 1)
        for row in rows:
            row["win_rate"] = float(wins.get(row["view"], 0.0) / denom)
    summary = pd.DataFrame(rows)
    fusion_gain = np.nan
    fused_df = df[df["view"] == "all"].copy()
    if not fused_df.empty and not single_df.empty:
        gains = []
        for prop, fused_sub in fused_df.groupby("property", dropna=False):
            fused_r2 = pd.to_numeric(fused_sub["r2"], errors="coerce").mean()
            best_single = pd.to_numeric(single_df[single_df["property"] == prop]["r2"], errors="coerce").max()
            if np.isfinite(fused_r2) and np.isfinite(best_single):
                gains.append(float(fused_r2 - best_single))
        if gains:
            fusion_gain = float(np.mean(gains))
    return summary, {"fusion_gain_mean": fusion_gain}


def _get_representation_series(paired_subset: pd.DataFrame, view: str) -> pd.Series:
    if view in {"smiles", "smiles_bpe", "graph"}:
        return paired_subset.get("p_smiles", pd.Series(dtype=str)).astype(str)
    if view == "selfies":
        return paired_subset.get("selfies", pd.Series(dtype=str)).astype(str)
    if view == "group_selfies":
        return paired_subset.get("group_selfies", pd.Series(dtype=str)).astype(str)
    return pd.Series(dtype=str)


def _tokenizer_efficiency(
    *,
    view: str,
    paired_subset: pd.DataFrame,
    encoder_cfg: dict,
    sample_cap: Optional[int],
    seed: int,
) -> dict:
    if paired_subset.empty:
        return {
            "view": view,
            "representation_length_mean": np.nan,
            "representation_length_p95": np.nan,
            "truncation_rate": np.nan,
        }
    sample_idx = _subsample_indices(len(paired_subset), sample_cap, seed)
    sample_df = paired_subset.iloc[sample_idx].copy()
    lengths: list[float] = []
    truncated = 0
    total = 0

    try:
        if view == "graph":
            tokenizer = _load_graph_tokenizer(encoder_cfg)
            max_len = int(getattr(tokenizer, "Nmax", 0) or 0)
            for text in sample_df.get("p_smiles", pd.Series(dtype=str)).astype(str):
                if not text:
                    continue
                total += 1
                try:
                    graph = tokenizer.encode(text)
                    length = float(np.asarray(graph["M"]).sum())
                    lengths.append(length)
                    if max_len > 0 and length >= max_len:
                        truncated += 1
                except Exception:
                    continue
        else:
            tokenizer = _load_sequence_tokenizer(view, encoder_cfg)
            max_len = int(getattr(tokenizer, "max_length", 0) or 0)
            rep = _get_representation_series(sample_df, view)
            for text in rep.astype(str):
                if not text or text.lower() in {"nan", "none", "null"}:
                    continue
                total += 1
                try:
                    tokens = tokenizer.tokenize(text)
                except Exception:
                    continue
                length = float(len(tokens) + 2)
                lengths.append(length)
                if max_len > 0 and length > max_len:
                    truncated += 1
    except Exception:
        return {
            "view": view,
            "representation_length_mean": np.nan,
            "representation_length_p95": np.nan,
            "truncation_rate": np.nan,
        }

    if not lengths:
        return {
            "view": view,
            "representation_length_mean": np.nan,
            "representation_length_p95": np.nan,
            "truncation_rate": np.nan,
        }
    arr = np.asarray(lengths, dtype=np.float32)
    return {
        "view": view,
        "representation_length_mean": float(np.mean(arr)),
        "representation_length_p95": float(np.percentile(arr, 95)),
        "truncation_rate": float(truncated / max(total, 1)),
    }


def _plot_embedding_research(metrics_df: pd.DataFrame, figures_dir: Path, dataset: str) -> None:
    if plt is None or metrics_df.empty:
        return
    df = metrics_df.copy()
    df = df[df["view"].astype(str) != "all"].copy()
    if df.empty:
        return
    views = ordered_views(df["view"].tolist())
    x = np.arange(len(views), dtype=np.float32)
    colors = [view_color(view) for view in views]
    labels = [view_label(view) for view in views]

    def _vals(column: str) -> np.ndarray:
        arr = []
        for view in views:
            match = df[df["view"] == view]
            val = pd.to_numeric(match[column], errors="coerce").mean() if not match.empty else np.nan
            arr.append(float(val) if np.isfinite(val) else np.nan)
        return np.asarray(arr, dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax0, ax1, ax2, ax3 = axes.reshape(-1)
    panels = [
        (ax0, "mean_recall_at_10", "Mean Recall@10"),
        (ax1, "effective_rank_norm", "Effective Rank / Dim"),
        (ax2, "representation_length_mean", "Mean Representation Length"),
        (ax3, "property_smoothness_mean", "Property Smoothness (kNN R2)"),
    ]
    for idx, (ax, col, ylabel) in enumerate(panels):
        y = _vals(col)
        ax.bar(x, y, color=colors, alpha=0.92, width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.text(0.01, 0.98, f"({chr(ord('a') + idx)})", transform=ax.transAxes, ha="left", va="top")
        if col in {"mean_recall_at_10"}:
            ax.set_ylim(bottom=0.0, top=max(1.0, float(np.nanmax(y)) * 1.08 if np.isfinite(y).any() else 1.0))
    fig.text(0.995, 0.005, str(dataset).upper(), ha="right", va="bottom", fontsize=16, alpha=0.6)
    save_figure_png(fig, figures_dir / "figure_f4_embedding_research")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    return parser


def main(args) -> None:
    config = load_config(args.config)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step4_embedding_research")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")

    f4_cfg = config.get("embedding_research", {}) or {}
    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(f4_cfg.get("generate_figures", True), True)
    if generate_figures and plt is None:
        print("Warning: matplotlib unavailable; skipping F4 figures.")
        generate_figures = False
    if plt is not None:
        set_publication_style()

    dataset = str(f4_cfg.get("analysis_dataset", "d2")).strip().lower() or "d2"
    max_samples_per_view = f4_cfg.get("max_samples_per_view", 3000)
    tokenizer_sample_cap = f4_cfg.get("tokenizer_sample_cap", 3000)
    knn_k = int(f4_cfg.get("knn_k", 10))
    property_knn_k = int(f4_cfg.get("property_knn_k", 5))
    min_class_samples = int(f4_cfg.get("min_class_samples", 20))
    seed = int((config.get("data", {}) or {}).get("random_seed", 42))

    paired_df = _load_paired_index(results_dir, config)
    paired_df["polymer_id"] = paired_df["polymer_id"].astype(str)
    paired_df["dataset"] = paired_df["dataset"].astype(str).str.lower()
    paired_df = paired_df[paired_df["dataset"] == dataset].copy()
    paired_lookup = paired_df.set_index("polymer_id", drop=False)
    total_dataset_count = int(len(paired_df))

    retrieval_df = _load_step_csv(results_dir, "step2_retrieval", "metrics_alignment.csv")
    retrieval_summary_df = _summarize_retrieval(retrieval_df, dataset)

    property_metrics_df = _load_step_csv(results_dir, "step3_property", "metrics_property.csv")
    complementarity_df, complementarity_meta = _summarize_property_heads(property_metrics_df)

    prop_dir = _resolve_path(config["paths"]["property_dir"])
    prop_cfg = config.get("property", {}) or {}
    properties = []
    files = prop_cfg.get("files", []) or []
    if isinstance(files, str):
        files = [files]
    for item in files:
        name = str(Path(str(item)).stem).strip()
        if name:
            properties.append(name)
    properties = ordered_properties(properties)

    views = _active_views(config)
    encoder_cfgs = {
        "smiles": config.get("smiles_encoder", {}) or {},
        "smiles_bpe": config.get("smiles_bpe_encoder", {}) or {},
        "selfies": config.get("selfies_encoder", {}) or {},
        "group_selfies": config.get("group_selfies_encoder", {}) or {},
        "graph": config.get("graph_encoder", {}) or {},
    }

    geometry_rows = []
    tokenizer_rows = []
    semantic_rows = []
    metric_rows = []

    for view in views:
        embeddings = load_view_embeddings(results_dir, view, dataset)
        idx_df = load_view_index(results_dir, view)
        if embeddings is None or idx_df is None:
            print(f"Warning: missing embeddings for view={view} dataset={dataset}; skipping in F4.")
            continue
        ids, aligned = _align_ids_and_embeddings(idx_df, embeddings, dataset)
        if not ids or aligned.size == 0:
            continue
        valid_ids = [pid for pid in ids if pid in paired_lookup.index]
        if not valid_ids:
            continue
        id_mask = [pid in paired_lookup.index for pid in ids]
        aligned = aligned[np.asarray(id_mask, dtype=bool)]
        paired_subset = paired_lookup.loc[valid_ids].copy()
        paired_subset = paired_subset.reset_index(drop=True)
        if len(paired_subset) != int(aligned.shape[0]):
            continue

        sample_idx = _subsample_indices(len(paired_subset), max_samples_per_view, seed + len(view))
        emb_sample = aligned[sample_idx]
        paired_sample = paired_subset.iloc[sample_idx].reset_index(drop=True)
        emb_sample = _normalize_rows(emb_sample)

        effective_rank, participation_ratio = _effective_rank_and_participation(emb_sample)
        hubness_top1pct = _hubness_score(emb_sample, knn_k)
        class_separation = _class_separation(emb_sample, paired_sample.get("p_smiles", pd.Series(dtype=str)).astype(str).tolist(), min_class_samples)

        geometry_rows.append(
            {
                "view": view,
                "dataset": dataset,
                "n_samples": int(len(paired_subset)),
                "embedding_dim": int(aligned.shape[1]),
                "effective_rank_norm": effective_rank,
                "participation_ratio_norm": participation_ratio,
                "hubness_top1pct": hubness_top1pct,
                "class_separation": class_separation,
            }
        )

        token_stats = _tokenizer_efficiency(
            view=view,
            paired_subset=paired_subset,
            encoder_cfg=encoder_cfgs.get(view, {}),
            sample_cap=tokenizer_sample_cap,
            seed=seed + 100 + len(view),
        )
        tokenizer_rows.append(token_stats)

        prop_scores = []
        for prop in properties:
            prop_df = _load_property_table(prop_dir, prop)
            if prop_df.empty:
                continue
            smiles_col, value_col = _resolve_property_columns(prop_df, prop)
            if not smiles_col or not value_col:
                continue
            series = (
                prop_df[[smiles_col, value_col]]
                .dropna(subset=[smiles_col, value_col])
                .groupby(smiles_col, dropna=False)[value_col]
                .mean()
            )
            score = _property_smoothness(
                embeddings=emb_sample,
                smiles_lookup=paired_sample.get("p_smiles", pd.Series(dtype=str)),
                property_values=series,
                k=property_knn_k,
            )
            semantic_rows.append(
                {
                    "view": view,
                    "dataset": dataset,
                    "property": prop,
                    "property_smoothness": score,
                }
            )
            if np.isfinite(score):
                prop_scores.append(float(score))

        retrieval_match = retrieval_summary_df[retrieval_summary_df["view"] == view]
        retrieval_row = retrieval_match.iloc[0].to_dict() if not retrieval_match.empty else {}
        comp_match = complementarity_df[complementarity_df["view"] == view]
        comp_row = comp_match.iloc[0].to_dict() if not comp_match.empty else {}
        meta = load_view_meta(results_dir, view) or {}
        coverage = float(len(paired_subset) / max(total_dataset_count, 1))

        metric_rows.append(
            {
                "view": view,
                "representation": _representation_from_view(view),
                "dataset": dataset,
                "model_size": str(meta.get("model_size", "")),
                "n_samples": int(len(paired_subset)),
                "embedding_dim": int(aligned.shape[1]),
                "coverage": coverage,
                "mean_recall_at_1": retrieval_row.get("mean_recall_at_1", np.nan),
                "mean_recall_at_5": retrieval_row.get("mean_recall_at_5", np.nan),
                "mean_recall_at_10": retrieval_row.get("mean_recall_at_10", np.nan),
                "mean_match_rate": retrieval_row.get("mean_match_rate", np.nan),
                "effective_rank_norm": effective_rank,
                "participation_ratio_norm": participation_ratio,
                "hubness_top1pct": hubness_top1pct,
                "class_separation": class_separation,
                "representation_length_mean": token_stats.get("representation_length_mean", np.nan),
                "representation_length_p95": token_stats.get("representation_length_p95", np.nan),
                "truncation_rate": token_stats.get("truncation_rate", np.nan),
                "tokenizer_failure_rate": 1.0 - coverage,
                "property_smoothness_mean": float(np.mean(prop_scores)) if prop_scores else np.nan,
                "mean_test_r2": comp_row.get("mean_test_r2", np.nan),
                "mean_test_rmse": comp_row.get("mean_test_rmse", np.nan),
                "win_rate": comp_row.get("win_rate", np.nan),
                "fusion_gain_mean": complementarity_meta.get("fusion_gain_mean", np.nan),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        aggregate = {
            "view": "all",
            "representation": "MultiViewSummary",
            "dataset": dataset,
            "model_size": str(next((x for x in metrics_df["model_size"].astype(str) if x), "")),
            "n_samples": int(metrics_df["n_samples"].sum()),
            "embedding_dim": round(float(pd.to_numeric(metrics_df["embedding_dim"], errors="coerce").mean())),
            "coverage": float(pd.to_numeric(metrics_df["coverage"], errors="coerce").mean()),
            "mean_recall_at_1": float(pd.to_numeric(metrics_df["mean_recall_at_1"], errors="coerce").mean()),
            "mean_recall_at_5": float(pd.to_numeric(metrics_df["mean_recall_at_5"], errors="coerce").mean()),
            "mean_recall_at_10": float(pd.to_numeric(metrics_df["mean_recall_at_10"], errors="coerce").mean()),
            "mean_match_rate": float(pd.to_numeric(metrics_df["mean_match_rate"], errors="coerce").mean()),
            "effective_rank_norm": float(pd.to_numeric(metrics_df["effective_rank_norm"], errors="coerce").mean()),
            "participation_ratio_norm": float(pd.to_numeric(metrics_df["participation_ratio_norm"], errors="coerce").mean()),
            "hubness_top1pct": float(pd.to_numeric(metrics_df["hubness_top1pct"], errors="coerce").mean()),
            "class_separation": float(pd.to_numeric(metrics_df["class_separation"], errors="coerce").mean()),
            "representation_length_mean": float(pd.to_numeric(metrics_df["representation_length_mean"], errors="coerce").mean()),
            "representation_length_p95": float(pd.to_numeric(metrics_df["representation_length_p95"], errors="coerce").mean()),
            "truncation_rate": float(pd.to_numeric(metrics_df["truncation_rate"], errors="coerce").mean()),
            "tokenizer_failure_rate": float(pd.to_numeric(metrics_df["tokenizer_failure_rate"], errors="coerce").mean()),
            "property_smoothness_mean": float(pd.to_numeric(metrics_df["property_smoothness_mean"], errors="coerce").mean()),
            "mean_test_r2": float(pd.to_numeric(metrics_df["mean_test_r2"], errors="coerce").mean()),
            "mean_test_rmse": float(pd.to_numeric(metrics_df["mean_test_rmse"], errors="coerce").mean()),
            "win_rate": float(pd.to_numeric(metrics_df["win_rate"], errors="coerce").mean()),
            "fusion_gain_mean": float(pd.to_numeric(metrics_df["fusion_gain_mean"], errors="coerce").mean()),
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([aggregate])], ignore_index=True)

    save_csv(
        metrics_df,
        step_dirs["metrics_dir"] / "metrics_embedding_research.csv",
        legacy_paths=[results_dir / "metrics_embedding_research.csv"],
        index=False,
    )
    save_csv(
        pd.DataFrame(geometry_rows),
        step_dirs["files_dir"] / "view_geometry_summary.csv",
        legacy_paths=[results_dir / "step4_embedding_research" / "view_geometry_summary.csv"],
        index=False,
    )
    save_csv(
        retrieval_summary_df,
        step_dirs["files_dir"] / "view_retrieval_summary.csv",
        legacy_paths=[results_dir / "step4_embedding_research" / "view_retrieval_summary.csv"],
        index=False,
    )
    save_csv(
        pd.DataFrame(tokenizer_rows),
        step_dirs["files_dir"] / "view_tokenizer_efficiency.csv",
        legacy_paths=[results_dir / "step4_embedding_research" / "view_tokenizer_efficiency.csv"],
        index=False,
    )
    save_csv(
        pd.DataFrame(semantic_rows),
        step_dirs["files_dir"] / "view_semantic_structure.csv",
        legacy_paths=[results_dir / "step4_embedding_research" / "view_semantic_structure.csv"],
        index=False,
    )
    save_csv(
        complementarity_df,
        step_dirs["files_dir"] / "view_complementarity_summary.csv",
        legacy_paths=[results_dir / "step4_embedding_research" / "view_complementarity_summary.csv"],
        index=False,
    )

    run_meta = {
        "dataset": dataset,
        "views": [str(v) for v in metrics_df.get("view", pd.Series(dtype=str)).astype(str).tolist() if str(v) != "all"],
        "max_samples_per_view": max_samples_per_view,
        "tokenizer_sample_cap": tokenizer_sample_cap,
        "knn_k": knn_k,
        "property_knn_k": property_knn_k,
        "min_class_samples": min_class_samples,
        "properties": properties,
        "fusion_gain_mean": complementarity_meta.get("fusion_gain_mean", np.nan),
    }
    save_json(
        run_meta,
        step_dirs["files_dir"] / "run_meta.json",
        legacy_paths=[results_dir / "step4_embedding_research" / "run_meta.json"],
        indent=2,
    )

    if generate_figures and not metrics_df.empty:
        _plot_embedding_research(metrics_df, step_dirs["figures_dir"], dataset)

    print(f"Saved metrics_embedding_research.csv to {results_dir}")


if __name__ == "__main__":
    main(build_parser().parse_args())
