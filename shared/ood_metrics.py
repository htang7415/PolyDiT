"""OOD metrics utilities.

Compute nearest-neighbor distances between embedding sets. Uses FAISS if available,
otherwise falls back to a batched numpy implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _try_faiss_knn(queries: np.ndarray, refs: np.ndarray, k: int) -> Optional[np.ndarray]:
    try:
        import faiss  # type: ignore
    except Exception:
        return None

    queries_f = queries.astype("float32")
    refs_f = refs.astype("float32")
    dim = refs_f.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(refs_f)
    distances, _ = index.search(queries_f, k)
    # distances are squared L2
    return np.sqrt(distances)


def _numpy_knn(queries: np.ndarray, refs: np.ndarray, k: int, batch_size: int = 512) -> np.ndarray:
    n = queries.shape[0]
    distances = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        q = queries[start:end]
        # Compute squared L2 distances
        diff = q[:, None, :] - refs[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        # Take k smallest
        part = np.partition(dist2, kth=min(k - 1, dist2.shape[1] - 1), axis=1)[:, :k]
        distances.append(np.sqrt(part))
    return np.vstack(distances)


def knn_distances(queries: np.ndarray, refs: np.ndarray, k: int = 1) -> np.ndarray:
    if queries.size == 0 or refs.size == 0:
        return np.zeros((queries.shape[0], k), dtype=np.float32)
    distances = _try_faiss_knn(queries, refs, k)
    if distances is not None:
        return distances
    return _numpy_knn(queries, refs, k)


def compute_ood_summary(
    d1_embeddings: np.ndarray,
    d2_embeddings: np.ndarray,
    generated_embeddings: Optional[np.ndarray] = None,
    k: int = 1,
    near_threshold: Optional[float] = None,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Compute OOD distances.

    Returns:
        (d1_to_d2_mean_dist, generated_to_d2_mean_dist, frac_generated_near_d2)
    """
    d1_to_d2 = knn_distances(d1_embeddings, d2_embeddings, k)
    d1_to_d2_mean = float(np.mean(d1_to_d2)) if d1_to_d2.size > 0 else 0.0

    gen_mean = None
    frac_near = None
    if generated_embeddings is not None and generated_embeddings.size > 0:
        gen_to_d2 = knn_distances(generated_embeddings, d2_embeddings, k)
        gen_mean = float(np.mean(gen_to_d2)) if gen_to_d2.size > 0 else 0.0
        threshold = near_threshold if near_threshold is not None else d1_to_d2_mean
        frac_near = float(np.mean(gen_to_d2 <= threshold)) if gen_to_d2.size > 0 else 0.0

    return d1_to_d2_mean, gen_mean, frac_near


def compute_ood_metrics_from_files(
    d1_path: Path,
    d2_path: Path,
    generated_path: Optional[Path] = None,
    k: int = 1,
) -> dict:
    d1 = np.load(d1_path)
    d2 = np.load(d2_path)
    gen = np.load(generated_path) if generated_path and generated_path.exists() else None

    d1_to_d2_mean, gen_mean, frac_near = compute_ood_summary(
        d1_embeddings=d1,
        d2_embeddings=d2,
        generated_embeddings=gen,
        k=k,
        near_threshold=None,
    )

    return {
        "d1_to_d2_mean_dist": round(d1_to_d2_mean, 4),
        "generated_to_d2_mean_dist": round(gen_mean, 4) if gen_mean is not None else None,
        "frac_generated_near_d2": round(frac_near, 4) if frac_near is not None else None,
    }
