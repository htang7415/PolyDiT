"""Foundation-enhanced inverse design helpers."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from shared.ood_metrics import knn_distances


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
