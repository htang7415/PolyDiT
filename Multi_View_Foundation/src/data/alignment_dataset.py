"""Datasets for end-to-end multi-view alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .view_converters import smiles_to_selfies


@dataclass
class AlignmentSample:
    view_tensors: Dict[str, Dict[str, np.ndarray]]


def _load_paired_df(path: Path, dataset_filter: Optional[str] = None, max_samples: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "p_smiles" not in df.columns and "SMILES" in df.columns:
        df = df.rename(columns={"SMILES": "p_smiles"})
    if dataset_filter and "dataset" in df.columns:
        df = df[df["dataset"] == dataset_filter].copy()
    if max_samples:
        df = df.head(int(max_samples)).copy()
    if "polymer_id" not in df.columns:
        df["polymer_id"] = [f"row_{i}" for i in range(len(df))]
    return df


class MultiViewAlignmentDataset(Dataset):
    """On-the-fly tokenization for end-to-end alignment training.

    Each item returns per-view token dicts. Samples that fail to encode
    any requested view will be returned with missing views; collate will
    filter to complete-view samples.
    """

    def __init__(
        self,
        paired_index: Path,
        views: List[str],
        tokenizers: Dict[str, object],
        dataset_filter: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.views = views
        self.tokenizers = tokenizers
        self.df = _load_paired_df(paired_index, dataset_filter=dataset_filter, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_view(self, view: str, row: pd.Series) -> Optional[Dict[str, np.ndarray]]:
        if view == "smiles":
            text = row.get("p_smiles", "")
            if not isinstance(text, str) or not text:
                return None
            encoded = self.tokenizers[view].encode(text, add_special_tokens=True, padding=True, return_attention_mask=True)
            return {
                "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
            }
        if view == "selfies":
            text = row.get("selfies")
            if not isinstance(text, str) or not text:
                text = smiles_to_selfies(row.get("p_smiles", ""))
            if not isinstance(text, str) or not text:
                return None
            encoded = self.tokenizers[view].encode(text, add_special_tokens=True, padding=True, return_attention_mask=True)
            return {
                "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
            }
        if view == "group_selfies":
            text = row.get("p_smiles", "")
            if not isinstance(text, str) or not text:
                return None
            encoded = self.tokenizers[view].encode(text, add_special_tokens=True, padding=True, return_attention_mask=True)
            return {
                "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
            }
        if view == "graph":
            text = row.get("p_smiles", "")
            if not isinstance(text, str) or not text:
                return None
            data = self.tokenizers[view].encode(text)
            return {
                "X": np.asarray(data["X"], dtype=np.int64),
                "E": np.asarray(data["E"], dtype=np.int64),
                "M": np.asarray(data["M"], dtype=np.float32),
            }
        return None

    def __getitem__(self, idx: int) -> AlignmentSample:
        row = self.df.iloc[idx]
        view_tensors: Dict[str, Dict[str, np.ndarray]] = {}
        for view in self.views:
            if view not in self.tokenizers:
                continue
            try:
                encoded = self._encode_view(view, row)
            except Exception:
                encoded = None
            if encoded is not None:
                view_tensors[view] = encoded
        return AlignmentSample(view_tensors=view_tensors)


def collate_alignment_batch(batch: List[AlignmentSample], required_views: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collate batch ensuring all required views present per sample."""
    if not batch:
        return {}

    filtered = [item for item in batch if all(v in item.view_tensors for v in required_views)]
    if not filtered:
        return {}

    out: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for view in required_views:
        out[view] = {}

    for item in filtered:
        for view in required_views:
            view_data = item.view_tensors[view]
            for key, arr in view_data.items():
                out[view].setdefault(key, []).append(arr)

    tensor_out: Dict[str, Dict[str, torch.Tensor]] = {}
    for view, view_data in out.items():
        tensor_out[view] = {}
        for key, arr_list in view_data.items():
            stacked = np.stack(arr_list)
            dtype = torch.float32 if key == "M" else torch.long
            tensor_out[view][key] = torch.tensor(stacked, dtype=dtype)

    return tensor_out
