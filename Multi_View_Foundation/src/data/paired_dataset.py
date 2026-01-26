"""Paired dataset helpers for multi-view alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


@dataclass
class PairedRecord:
    polymer_id: str
    p_smiles: str
    selfies: str
    group_selfies: str
    graph: Dict


def load_paired_index(path: Path) -> List[PairedRecord]:
    df = pd.read_csv(path)
    for col in ["p_smiles", "selfies", "group_selfies", "graph"]:
        if col not in df.columns:
            df[col] = ""
    if "polymer_id" not in df.columns:
        df["polymer_id"] = [f"row_{i}" for i in range(len(df))]
    records: List[PairedRecord] = []
    for _, row in df.iterrows():
        records.append(
            PairedRecord(
                polymer_id=str(row["polymer_id"]),
                p_smiles=str(row["p_smiles"]),
                selfies=str(row["selfies"]),
                group_selfies=str(row["group_selfies"]),
                graph=row["graph"] if isinstance(row["graph"], dict) else {},
            )
        )
    return records


def _load_embeddings(results_dir: Path, view: str, dataset: str) -> Optional[np.ndarray]:
    emb_path = results_dir / f"embeddings_{view}_{dataset}.npy"
    if not emb_path.exists() and view == "smiles":
        legacy = results_dir / f"embeddings_{dataset}.npy"
        if legacy.exists():
            emb_path = legacy
    if not emb_path.exists():
        return None
    return np.load(emb_path)


def _load_embedding_index(results_dir: Path, view: str) -> Optional[pd.DataFrame]:
    idx_path = results_dir / f"embedding_index_{view}.csv"
    if not idx_path.exists():
        return None
    return pd.read_csv(idx_path)


def load_view_embeddings(
    results_dir: Path,
    views: List[str],
    dataset: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    view_embeddings: Dict[str, np.ndarray] = {}
    view_ids: Dict[str, List[str]] = {}

    for view in views:
        emb = _load_embeddings(results_dir, view, dataset)
        idx_df = _load_embedding_index(results_dir, view)
        if emb is None or idx_df is None:
            continue

        subset = idx_df[idx_df["dataset"] == dataset].copy()
        subset = subset.sort_values("row_index")
        ids = subset["polymer_id"].astype(str).tolist()

        if len(ids) != emb.shape[0]:
            min_len = min(len(ids), emb.shape[0])
            ids = ids[:min_len]
            emb = emb[:min_len]

        view_embeddings[view] = emb.astype(np.float32)
        view_ids[view] = ids

    return view_embeddings, view_ids


class EmbeddingPairDataset(Dataset):
    """Dataset that serves aligned embedding vectors across multiple views."""

    def __init__(self, view_embeddings: Dict[str, np.ndarray], view_ids: Dict[str, List[str]]):
        self.views = sorted(view_embeddings.keys())
        self.view_embeddings = view_embeddings
        self.view_ids = view_ids

        if not self.views:
            self.common_ids: List[str] = []
            self._indices: List[Dict[str, int]] = []
            return

        id_sets = [set(view_ids[v]) for v in self.views]
        common = set.intersection(*id_sets) if id_sets else set()
        self.common_ids = sorted(common)

        id_maps = {
            view: {pid: idx for idx, pid in enumerate(view_ids[view])}
            for view in self.views
        }
        self._indices = [
            {view: id_maps[view][pid] for view in self.views}
            for pid in self.common_ids
        ]

    def __len__(self) -> int:
        return len(self.common_ids)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        row = self._indices[idx]
        return {view: self.view_embeddings[view][row[view]] for view in self.views}
