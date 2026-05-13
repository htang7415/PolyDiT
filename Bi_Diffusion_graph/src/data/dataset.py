"""PyTorch Dataset classes for polymer data.

Includes:
- PolymerDataset: Sequence-based dataset for tokenized SMILES
- GraphPolymerDataset: Graph-based dataset for (X, E, M) tensors
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm

from .tokenizer import PSmilesTokenizer
from .graph_tokenizer import GraphTokenizer


class PolymerDataset(Dataset):
    """Dataset for unlabeled polymer SMILES (diffusion training)."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PSmilesTokenizer,
        smiles_col: str = 'p_smiles',
        max_length: Optional[int] = None,
        cache_tokenization: bool = False
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with SMILES data.
            tokenizer: Tokenizer instance.
            smiles_col: Name of SMILES column.
            max_length: Maximum sequence length (overrides tokenizer).
            cache_tokenization: Whether to pre-tokenize and cache all samples.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.smiles_col = smiles_col
        self.cache_tokenization = cache_tokenization
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if max_length:
            self.tokenizer.max_length = max_length

        if cache_tokenization:
            self._pretokenize()

    def _pretokenize(self):
        """Pre-tokenize all samples and cache them."""
        print(f"Pre-tokenizing {len(self)} samples...")
        for idx in tqdm(range(len(self)), desc="Tokenizing"):
            smiles = self.df.iloc[idx][self.smiles_col]
            encoded = self.tokenizer.encode(
                smiles,
                add_special_tokens=True,
                padding=True,
                return_attention_mask=True
            )
            self._cache[idx] = {
                'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long)
            }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return cached if available
        if self.cache_tokenization and idx in self._cache:
            return self._cache[idx]

        smiles = self.df.iloc[idx][self.smiles_col]

        # Encode SMILES
        encoded = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True
        )

        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long)
        }


class GraphPolymerDataset(Dataset):
    """Dataset for graph-based polymer representation (graph diffusion training).

    Returns (X, E, M) tensors where:
    - X: node tokens (atom types)
    - E: edge tokens (bond types)
    - M: node mask
    """

    def __init__(
        self,
        df: pd.DataFrame,
        graph_tokenizer: GraphTokenizer,
        smiles_col: str = 'p_smiles',
        cache_graphs: bool = False
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with SMILES data.
            graph_tokenizer: GraphTokenizer instance.
            smiles_col: Name of SMILES column.
            cache_graphs: Whether to pre-encode and cache all graphs.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = graph_tokenizer
        self.smiles_col = smiles_col
        self.cache_graphs = cache_graphs
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if cache_graphs:
            self._pre_encode()

    def _pre_encode(self):
        """Pre-encode all samples and cache them."""
        print(f"Pre-encoding {len(self)} graphs...")
        for idx in tqdm(range(len(self)), desc="Encoding graphs"):
            smiles = self.df.iloc[idx][self.smiles_col]
            try:
                graph_data = self.tokenizer.encode(smiles)
                self._cache[idx] = {
                    'X': torch.tensor(graph_data['X'], dtype=torch.long),
                    'E': torch.tensor(graph_data['E'], dtype=torch.long),
                    'M': torch.tensor(graph_data['M'], dtype=torch.float)
                }
            except Exception as e:
                # Skip invalid molecules (shouldn't happen with clean data)
                print(f"Warning: Failed to encode molecule {idx}: {e}")
                # Use empty placeholder
                N = self.tokenizer.Nmax
                self._cache[idx] = {
                    'X': torch.full((N,), self.tokenizer.pad_id, dtype=torch.long),
                    'E': torch.zeros((N, N), dtype=torch.long),
                    'M': torch.zeros(N, dtype=torch.float)
                }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return cached if available
        if self.cache_graphs and idx in self._cache:
            return self._cache[idx]

        smiles = self.df.iloc[idx][self.smiles_col]

        try:
            graph_data = self.tokenizer.encode(smiles)
            return {
                'X': torch.tensor(graph_data['X'], dtype=torch.long),
                'E': torch.tensor(graph_data['E'], dtype=torch.long),
                'M': torch.tensor(graph_data['M'], dtype=torch.float)
            }
        except Exception:
            # Return empty graph for invalid molecules
            N = self.tokenizer.Nmax
            return {
                'X': torch.full((N,), self.tokenizer.pad_id, dtype=torch.long),
                'E': torch.zeros((N, N), dtype=torch.long),
                'M': torch.zeros(N, dtype=torch.float)
            }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader (sequence-based).

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary of tensors.
    """
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([item[key] for item in batch])
    return result


def graph_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for graph DataLoader.

    Args:
        batch: List of sample dictionaries with X, E, M tensors.

    Returns:
        Batched dictionary of tensors.
    """
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([item[key] for item in batch])
    return result
