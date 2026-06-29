"""OOD metrics utilities.

Compute nearest-neighbor distances between embedding sets. Uses FAISS if available,
otherwise falls back to a batched numpy implementation.

Distance metric: cosine distance in [0, 1].
- FAISS path: L2-normalize both sets -> IndexFlatL2 -> sq_dist / 2.0
- NumPy path: L2-normalize both sets -> 1 - (q @ refs.T), clipped to [0, 1]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize; zero-norm rows get denominator 1.0 (remain zero-vectors)."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return x / norms


def _try_faiss_knn(queries: np.ndarray, refs: np.ndarray, k: int, use_faiss: bool = True) -> Optional[np.ndarray]:
    if not use_faiss:
        return None
    try:
        import faiss  # type: ignore
    except Exception:
        return None

    queries_f = _l2_normalize(queries).astype("float32")
    refs_f = _l2_normalize(refs).astype("float32")
    dim = refs_f.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(refs_f)
    sq_distances, _ = index.search(queries_f, k)
    # On unit sphere: cosine_dist = sq_L2 / 2, range [0, 1]
    return (sq_distances / 2.0).astype(np.float32, copy=False)


def _numpy_knn(queries: np.ndarray, refs: np.ndarray, k: int, batch_size: int = 512) -> np.ndarray:
    queries_f = _l2_normalize(queries)
    refs_f = _l2_normalize(refs)
    n = queries_f.shape[0]
    distances = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        q = queries_f[start:end]
        # cosine distance = 1 - cosine_similarity; clip to [0, 1]
        cos_sim = q @ refs_f.T
        cos_dist = 1.0 - cos_sim
        np.clip(cos_dist, 0.0, 1.0, out=cos_dist)
        # Take k smallest cosine distances
        part = np.partition(cos_dist, kth=min(k - 1, cos_dist.shape[1] - 1), axis=1)[:, :k]
        distances.append(part.astype(np.float32, copy=False))
    return np.vstack(distances)


def knn_distances(queries: np.ndarray, refs: np.ndarray, k: int = 1, use_faiss: bool = True) -> np.ndarray:
    """Return cosine distances to k nearest neighbours. Shape: (n_queries, k), range [0, 1]."""
    if queries.size == 0 or refs.size == 0:
        return np.zeros((queries.shape[0], k), dtype=np.float32)
    distances = _try_faiss_knn(queries, refs, k, use_faiss=use_faiss)
    if distances is not None:
        return distances
    return _numpy_knn(queries, refs, k)


def compute_ood_summary(
    d1_embeddings: np.ndarray,
    d2_embeddings: np.ndarray,
    generated_embeddings: Optional[np.ndarray] = None,
    k: int = 1,
    near_threshold: Optional[float] = None,
    use_faiss: bool = True,
) -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute OOD cosine distances.

    Returns:
        (d1_to_d2_mean, gen_ood_prop_mean, frac_near_d2, gen_ood_gen_mean, frac_near_d1)

        - d1_to_d2_mean:   mean cosine dist from D1 to D2 (baseline)
        - gen_ood_prop_mean: mean cosine dist from generated to D2 (ood_prop)
        - frac_near_d2:    fraction of generated within threshold of D2
        - gen_ood_gen_mean: mean cosine dist from generated to D1 (ood_gen)
        - frac_near_d1:    fraction of generated within d1-to-d2 threshold of D1
    """
    d1_to_d2 = knn_distances(d1_embeddings, d2_embeddings, k, use_faiss=use_faiss)
    d1_to_d2_mean = float(np.mean(d1_to_d2)) if d1_to_d2.size > 0 else 0.0

    gen_ood_prop_mean = None
    frac_near_d2 = None
    gen_ood_gen_mean = None
    frac_near_d1 = None

    if generated_embeddings is not None and generated_embeddings.size > 0:
        gen_to_d2 = knn_distances(generated_embeddings, d2_embeddings, k, use_faiss=use_faiss)
        gen_ood_prop_mean = float(np.mean(gen_to_d2)) if gen_to_d2.size > 0 else 0.0
        threshold = near_threshold if near_threshold is not None else d1_to_d2_mean
        frac_near_d2 = float(np.mean(gen_to_d2 <= threshold)) if gen_to_d2.size > 0 else 0.0

        gen_to_d1 = knn_distances(generated_embeddings, d1_embeddings, k, use_faiss=use_faiss)
        gen_ood_gen_mean = float(np.mean(gen_to_d1)) if gen_to_d1.size > 0 else 0.0
        frac_near_d1 = float(np.mean(gen_to_d1 <= d1_to_d2_mean)) if gen_to_d1.size > 0 else 0.0

    return d1_to_d2_mean, gen_ood_prop_mean, frac_near_d2, gen_ood_gen_mean, frac_near_d1


def compute_ood_metrics_from_files(
    d1_path: Path,
    d2_path: Path,
    generated_path: Optional[Path] = None,
    k: int = 1,
    use_faiss: bool = True,
) -> dict:
    d1 = np.load(d1_path)
    d2 = np.load(d2_path)
    gen = np.load(generated_path) if generated_path and generated_path.exists() else None

    d1_to_d2_mean, gen_ood_prop_mean, frac_near_d2, gen_ood_gen_mean, frac_near_d1 = compute_ood_summary(
        d1_embeddings=d1,
        d2_embeddings=d2,
        generated_embeddings=gen,
        k=k,
        near_threshold=None,
        use_faiss=use_faiss,
    )

    return {
        "d1_to_d2_mean_dist": round(d1_to_d2_mean, 4),
        # New canonical names
        "ood_prop_mean": round(gen_ood_prop_mean, 4) if gen_ood_prop_mean is not None else None,
        "ood_gen_mean": round(gen_ood_gen_mean, 4) if gen_ood_gen_mean is not None else None,
        "frac_generated_near_d2": round(frac_near_d2, 4) if frac_near_d2 is not None else None,
        "frac_generated_near_d1": round(frac_near_d1, 4) if frac_near_d1 is not None else None,
        # Backward-compat alias
        "generated_to_d2_mean_dist": round(gen_ood_prop_mean, 4) if gen_ood_prop_mean is not None else None,
    }
