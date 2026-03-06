"""Retrieval metrics utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.shape[1] <= k:
        return np.argsort(-scores, axis=1)
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row_indices = np.arange(scores.shape[0])[:, None]
    sorted_part = np.argsort(-scores[row_indices, part], axis=1)
    return part[row_indices, sorted_part]


def _build_self_ref_indices(query_ids: List[str], ref_ids: List[str]) -> np.ndarray | None:
    """Infer one-to-one query->reference self mapping by ID occurrence order."""
    if len(query_ids) != len(ref_ids):
        return None

    ref_positions: Dict[str, List[int]] = {}
    for idx, rid in enumerate(ref_ids):
        ref_positions.setdefault(str(rid), []).append(idx)

    query_seen: Dict[str, int] = {}
    mapping: List[int] = []
    for qid in query_ids:
        qid_str = str(qid)
        occ = query_seen.get(qid_str, 0)
        query_seen[qid_str] = occ + 1
        candidates = ref_positions.get(qid_str, [])
        if occ >= len(candidates):
            return None
        mapping.append(int(candidates[occ]))

    if len(set(mapping)) != len(mapping):
        return None
    return np.asarray(mapping, dtype=np.int64)


def compute_recall_at_k(
    query_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    query_ids: Iterable[str],
    ref_ids: Iterable[str],
    ks: List[int],
    batch_size: int = 512,
    exclude_self: bool = False,
) -> Dict[str, float]:
    """Compute recall@K for retrieval with cosine similarity.

    Recall is computed over queries that have at least one positive in the
    reference set (based on matching IDs).
    """
    query_ids = list(query_ids)
    ref_ids = list(ref_ids)
    if len(query_ids) != query_embeddings.shape[0]:
        raise ValueError("query_ids length must match query_embeddings rows")
    if len(ref_ids) != ref_embeddings.shape[0]:
        raise ValueError("ref_ids length must match ref_embeddings rows")

    if query_embeddings.size == 0 or ref_embeddings.size == 0:
        return {
            **{f"recall_at_{k}": 0.0 for k in ks},
            "mean_cosine_sim_matched": 0.0,
            "matched_queries": 0,
            "total_queries": len(query_ids),
            "match_rate": 0.0,
        }

    ref_index: Dict[str, List[int]] = {}
    for idx, rid in enumerate(ref_ids):
        ref_index.setdefault(str(rid), []).append(idx)
    self_ref_indices = _build_self_ref_indices(query_ids, ref_ids) if exclude_self else None

    query_embeddings = _normalize(query_embeddings.astype(np.float32))
    ref_embeddings = _normalize(ref_embeddings.astype(np.float32))

    total_queries = len(query_ids)
    matched_queries = 0
    hits = {k: 0 for k in ks}
    max_k = max(ks)
    cosine_sims: List[float] = []

    for start in range(0, total_queries, batch_size):
        end = min(start + batch_size, total_queries)
        batch_emb = query_embeddings[start:end]
        batch_ids = query_ids[start:end]

        scores = batch_emb @ ref_embeddings.T
        if self_ref_indices is not None:
            local_rows = np.arange(end - start, dtype=np.int64)
            scores[local_rows, self_ref_indices[start:end]] = -np.inf
        topk = _topk_indices(scores, max_k)

        for row, qid in enumerate(batch_ids):
            qid_str = str(qid)
            pos_indices = ref_index.get(qid_str)
            if self_ref_indices is not None:
                self_idx = int(self_ref_indices[start + row])
                pos_indices = [idx for idx in pos_indices or [] if idx != self_idx]
            if not pos_indices:
                continue
            matched_queries += 1
            for k in ks:
                if any(idx in pos_indices for idx in topk[row, :k]):
                    hits[k] += 1
            cosine_sims.append(float(scores[row, pos_indices].max()))

    if matched_queries == 0:
        return {
            **{f"recall_at_{k}": np.nan for k in ks},
            "mean_cosine_sim_matched": np.nan,
            "matched_queries": 0,
            "total_queries": total_queries,
            "match_rate": 0.0,
        }

    denom = matched_queries
    return {
        **{f"recall_at_{k}": round(hits[k] / denom, 4) for k in ks},
        "mean_cosine_sim_matched": round(float(np.mean(cosine_sims)), 4) if cosine_sims else 0.0,
        "matched_queries": matched_queries,
        "total_queries": total_queries,
        "match_rate": round(matched_queries / max(total_queries, 1), 4),
    }
