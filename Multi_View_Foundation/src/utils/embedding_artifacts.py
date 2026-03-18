"""Helpers for reading MVF embedding artifacts from canonical or legacy paths."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Optional

import numpy as np
import pandas as pd


def view_embedding_candidates(results_dir: Path, view: str, dataset: str) -> list[Path]:
    view_token = str(view).strip()
    dataset_token = str(dataset).strip().lower()
    candidates = [
        Path(results_dir) / "step1_alignment_embeddings" / "files" / f"embeddings_{view_token}_{dataset_token}.npy",
        Path(results_dir) / f"embeddings_{view_token}_{dataset_token}.npy",
    ]
    if view_token == "smiles":
        candidates.extend(
            [
                Path(results_dir) / "step1_alignment_embeddings" / "files" / f"embeddings_{dataset_token}.npy",
                Path(results_dir) / f"embeddings_{dataset_token}.npy",
            ]
        )
    return candidates


def view_index_candidates(results_dir: Path, view: str) -> list[Path]:
    view_token = str(view).strip()
    return [
        Path(results_dir) / "step1_alignment_embeddings" / "files" / f"embedding_index_{view_token}.csv",
        Path(results_dir) / f"embedding_index_{view_token}.csv",
    ]


def view_meta_candidates(results_dir: Path, view: str) -> list[Path]:
    view_token = str(view).strip()
    candidates = [
        Path(results_dir) / "step1_alignment_embeddings" / "files" / f"embedding_meta_{view_token}.json",
        Path(results_dir) / f"embedding_meta_{view_token}.json",
    ]
    if view_token == "smiles":
        candidates.extend(
            [
                Path(results_dir) / "step1_alignment_embeddings" / "files" / "embedding_meta.json",
                Path(results_dir) / "embedding_meta.json",
            ]
        )
    return candidates


def resolve_view_embedding_path(results_dir: Path, view: str, dataset: str) -> Path:
    candidates = view_embedding_candidates(results_dir, view, dataset)
    for path in candidates:
        if path.exists():
            return path
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Embedding file not found for view={view}, dataset={dataset}. Searched: {searched}"
    )


def load_view_embeddings(results_dir: Path, view: str, dataset: str) -> Optional[np.ndarray]:
    for path in view_embedding_candidates(results_dir, view, dataset):
        if path.exists():
            return np.load(path)
    return None


def load_view_index(results_dir: Path, view: str) -> Optional[pd.DataFrame]:
    for path in view_index_candidates(results_dir, view):
        if path.exists():
            return pd.read_csv(path)
    return None


def load_view_meta(results_dir: Path, view: str) -> Optional[dict]:
    for path in view_meta_candidates(results_dir, view):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None
