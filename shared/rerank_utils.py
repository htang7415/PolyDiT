"""Utilities for reranking inverse design candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def load_rerank_scores(
    score_path: Optional[str],
    keys: List[str],
    key_column: Optional[str] = None,
) -> Optional[np.ndarray]:
    if not score_path:
        return None

    path = Path(score_path)
    if not path.exists():
        return None

    if path.suffix == ".npy":
        arr = np.load(path)
        if len(arr) != len(keys):
            return None
        return np.asarray(arr, dtype=float)

    df = pd.read_csv(path)
    if df.empty:
        return None

    if key_column is None:
        for candidate in ["smiles", "selfies", "p_smiles", "psmiles"]:
            if candidate in df.columns:
                key_column = candidate
                break
    if key_column is None or key_column not in df.columns:
        return None

    score_col = None
    for candidate in ["score", "distance", "metric"]:
        if candidate in df.columns:
            score_col = candidate
            break
    if score_col is None:
        return None

    mapping = dict(zip(df[key_column].astype(str), df[score_col].astype(float)))
    scores = [mapping.get(str(k), float("inf")) for k in keys]
    return np.asarray(scores, dtype=float)


def compute_rerank_metrics(
    predictions: np.ndarray,
    target_value: float,
    epsilon: float,
    keys: List[str],
    strategy: str,
    score_path: Optional[str],
    key_column: Optional[str],
    top_k: int,
) -> Dict:
    """Compute rerank metrics using an external score or property error.

    Returns a dict with rerank_applied and associated metrics.
    """
    if predictions is None or len(predictions) == 0:
        return {"rerank_applied": False}

    scores = None
    if strategy == "property_error":
        scores = np.abs(predictions - target_value)
    elif strategy in {"external", "d2_distance", "consistency", "retrieval"}:
        scores = load_rerank_scores(score_path, keys, key_column)
        if scores is None:
            return {
                "rerank_applied": False,
                "rerank_reason": "missing_scores",
            }
    else:
        return {"rerank_applied": False}

    order = np.argsort(scores)
    k = min(int(top_k), len(order))
    if k <= 0:
        return {"rerank_applied": False}

    top_idx = order[:k]
    hits = np.abs(predictions[top_idx] - target_value) <= epsilon
    hits_count = int(hits.sum())
    success_rate = hits_count / k if k > 0 else 0.0

    return {
        "rerank_applied": True,
        "rerank_strategy": strategy,
        "rerank_top_k": k,
        "rerank_hits": hits_count,
        "rerank_success_rate": round(success_rate, 4),
    }
