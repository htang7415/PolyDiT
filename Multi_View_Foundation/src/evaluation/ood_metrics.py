"""OOD metrics wrapper for foundation embeddings."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from shared.ood_metrics import compute_ood_summary


def compute_ood_metrics(
    d1_embeddings: np.ndarray,
    d2_embeddings: np.ndarray,
    generated_embeddings: Optional[np.ndarray] = None,
    k: int = 1,
) -> Dict[str, float]:
    d1_to_d2_mean, gen_mean, frac_near = compute_ood_summary(
        d1_embeddings=d1_embeddings,
        d2_embeddings=d2_embeddings,
        generated_embeddings=generated_embeddings,
        k=k,
        near_threshold=None,
    )
    return {
        "d1_to_d2_mean_dist": round(float(d1_to_d2_mean), 4),
        "generated_to_d2_mean_dist": round(float(gen_mean), 4) if gen_mean is not None else None,
        "frac_generated_near_d2": round(float(frac_near), 4) if frac_near is not None else None,
    }
