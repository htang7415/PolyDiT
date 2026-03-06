"""Foundation-enhanced inverse design helpers."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from shared.ood_metrics import knn_distances


def compute_property_hits(
    predictions: np.ndarray,
    target_value: float,
    epsilon: float,
    target_mode: str = "window",
) -> np.ndarray:
    """Return boolean property-hit mask for window/ge/le targets."""
    preds = np.asarray(predictions, dtype=np.float32).reshape(-1)
    mode = str(target_mode).strip().lower()
    if mode == "window":
        return np.abs(preds - float(target_value)) <= float(epsilon)
    if mode == "ge":
        return preds >= float(target_value)
    if mode == "le":
        return preds <= float(target_value)
    raise ValueError(f"Unsupported target_mode={target_mode}. Use window|ge|le.")


def compute_property_error(
    predictions: np.ndarray,
    target_value: float,
    target_mode: str = "window",
    error_scale: Optional[float] = None,
    min_scale: float = 1e-3,
) -> np.ndarray:
    """Compute non-negative normalized property violation/error.

    - window: absolute error |pred-target|
    - ge: shortfall max(0, target-pred)
    - le: exceedance max(0, pred-target)
    Values are normalized by a robust scale:
      max(user_scale, robust_prediction_scale, 0.1*|target|, min_scale).
    This avoids unstable blow-ups when target is near zero.
    """
    preds = np.asarray(predictions, dtype=np.float32).reshape(-1)
    target = float(target_value)

    scale_candidates = [float(min_scale), abs(target) * 0.1]
    if error_scale is not None:
        try:
            scale_candidates.append(abs(float(error_scale)))
        except Exception:
            pass

    finite = preds[np.isfinite(preds)]
    if finite.size > 1:
        std = float(np.std(finite))
        if std > 0:
            scale_candidates.append(std)
        q75, q25 = np.percentile(finite, [75.0, 25.0])
        iqr = float(q75 - q25)
        if iqr > 0:
            # Consistent with sigma estimate for normal distributions.
            scale_candidates.append(iqr / 1.349)
        q95, q05 = np.percentile(finite, [95.0, 5.0])
        span = float(q95 - q05)
        if span > 0:
            scale_candidates.append(span / 3.29)

    denom = max(scale_candidates)

    mode = str(target_mode).strip().lower()
    if mode == "window":
        err = np.abs(preds - target)
    elif mode == "ge":
        err = np.maximum(0.0, target - preds)
    elif mode == "le":
        err = np.maximum(0.0, preds - target)
    else:
        raise ValueError(f"Unsupported target_mode={target_mode}. Use window|ge|le.")
    return (err / max(denom, float(min_scale))).astype(np.float32, copy=False)


def normalize_scores(values: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x

    norm_mode = str(mode).strip().lower()
    if norm_mode in {"none", ""}:
        return x

    if x.size <= 1:
        return np.zeros_like(x, dtype=np.float32)

    if norm_mode == "rank":
        if float(np.max(x) - np.min(x)) <= 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(x.size, dtype=np.float32)
        denom = max(float(x.size - 1), 1.0)
        return ranks / denom

    if norm_mode != "minmax":
        raise ValueError(f"Unsupported normalization={mode}. Use minmax|rank|none.")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    span = x_max - x_min
    if span <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / span


def _normalize_scores(values: np.ndarray, mode: str) -> np.ndarray:
    """Backward-compatible alias for existing imports."""
    return normalize_scores(values, mode)


def compute_ood_aware_objective(
    predictions: np.ndarray,
    d2_distances: np.ndarray,
    target_value: float,
    target_mode: str = "window",
    property_weight: float = 0.7,
    ood_weight: float = 0.3,
    normalization: str = "minmax",
) -> Dict[str, np.ndarray]:
    """Compute OOD-aware objective scores for inverse design candidates.

    Objective is a weighted sum of normalized terms:
      property_weight * property_error_norm + ood_weight * d2_distance_norm
    Lower is better.
    """
    preds = np.asarray(predictions, dtype=np.float32).reshape(-1)
    d2 = np.asarray(d2_distances, dtype=np.float32).reshape(-1)
    if preds.shape[0] != d2.shape[0]:
        raise ValueError("predictions and d2_distances must have the same length.")

    p_w = float(property_weight)
    o_w = float(ood_weight)
    if p_w < 0 or o_w < 0:
        raise ValueError("property_weight and ood_weight must be non-negative.")
    total_w = p_w + o_w
    if total_w <= 0:
        raise ValueError("property_weight + ood_weight must be > 0.")
    p_w /= total_w
    o_w /= total_w

    property_error = compute_property_error(preds, target_value=float(target_value), target_mode=target_mode)
    property_error_norm = normalize_scores(property_error, normalization)
    d2_distance_norm = normalize_scores(d2, normalization)
    objective = p_w * property_error_norm + o_w * d2_distance_norm

    order = np.argsort(objective)
    rank = np.empty_like(order, dtype=np.int64)
    rank[order] = np.arange(objective.shape[0], dtype=np.int64)

    return {
        "property_error": property_error.astype(np.float32, copy=False),
        "property_error_norm": property_error_norm.astype(np.float32, copy=False),
        "d2_distance_norm": d2_distance_norm.astype(np.float32, copy=False),
        "objective": objective.astype(np.float32, copy=False),
        "objective_rank": rank,
        "property_weight_norm": np.asarray([p_w], dtype=np.float32),
        "ood_weight_norm": np.asarray([o_w], dtype=np.float32),
    }


def rerank_candidates(
    predictions: np.ndarray,
    target_value: float,
    epsilon: float,
    embeddings: Optional[np.ndarray] = None,
    d2_embeddings: Optional[np.ndarray] = None,
    strategy: str = "d2_distance",
    top_k: int = 100,
    k_nn: int = 5,
) -> Dict:
    if predictions is None or len(predictions) == 0:
        return {"rerank_applied": False}

    if strategy == "d2_distance":
        if embeddings is None or d2_embeddings is None:
            return {"rerank_applied": False, "rerank_reason": "missing_embeddings"}
        distances = knn_distances(embeddings, d2_embeddings, k=k_nn)
        scores = distances.mean(axis=1)
    else:
        scores = np.abs(predictions - target_value)

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
