"""Data loading and preprocessing for polymer data."""

import gzip
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split

from ..utils.chemistry import canonicalize_smiles, is_valid_psmiles, compute_sa_score


class PolymerDataLoader:
    """Load and preprocess polymer data."""

    def __init__(self, config: Dict):
        """Initialize data loader.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.random_seed = config['data']['random_seed']
        self.repo_root = Path(__file__).resolve().parents[3]
        self.data_dir = self._resolve_data_path(config['paths']['data_dir'])
        self.results_dir = Path(config['paths']['results_dir'])

    def _resolve_data_path(self, path_str: str) -> Path:
        """Resolve data path against cwd first, then repository root."""
        path = Path(path_str)
        if path.is_absolute():
            return path

        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path

        repo_path = self.repo_root / path
        if repo_path.exists():
            return repo_path

        # Keep a stable fallback for downstream error messages.
        return repo_path

    def load_unlabeled_data(self) -> pd.DataFrame:
        """Load unlabeled polymer SMILES data.

        Returns:
            DataFrame with cleaned p-SMILES.
        """
        polymer_file = self._resolve_data_path(self.config['paths']['polymer_file'])

        # Load gzipped CSV
        with gzip.open(polymer_file, 'rt') as f:
            df = pd.read_csv(f)

        # Column is named 'SMILES'
        smiles_col = 'SMILES'
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in {polymer_file}")

        return df[[smiles_col]].rename(columns={smiles_col: 'p_smiles'})

    def clean_and_filter(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'p_smiles'
    ) -> pd.DataFrame:
        """Clean and filter p-SMILES data.

        Args:
            df: DataFrame with SMILES column.
            smiles_col: Name of SMILES column.

        Returns:
            Cleaned DataFrame with only valid p-SMILES.
        """
        # Remove duplicates
        df = df.drop_duplicates(subset=[smiles_col]).copy()

        # Remove empty strings
        df = df[df[smiles_col].notna() & (df[smiles_col] != '')]

        # Filter to valid p-SMILES (parseable and exactly 2 stars)
        valid_mask = df[smiles_col].apply(is_valid_psmiles)
        df = df[valid_mask].copy()

        # Canonicalize SMILES
        df[smiles_col] = df[smiles_col].apply(
            lambda x: canonicalize_smiles(x) or x
        )

        # Remove duplicates again after canonicalization
        df = df.drop_duplicates(subset=[smiles_col]).copy()

        return df.reset_index(drop=True)

    def compute_sa_scores(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'p_smiles'
    ) -> pd.DataFrame:
        """Compute SA scores for all molecules.

        Args:
            df: DataFrame with SMILES column.
            smiles_col: Name of SMILES column.

        Returns:
            DataFrame with SA score column added.
        """
        df = df.copy()
        df['sa_score'] = df[smiles_col].apply(compute_sa_score)
        return df

    def split_unlabeled(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split unlabeled data into train/val.

        Args:
            df: DataFrame with unlabeled data.

        Returns:
            Tuple of (train_df, val_df).
        """
        train_ratio = self.config['data']['unlabeled_train_ratio']

        train_df, val_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=self.random_seed
        )

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    def prepare_unlabeled_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare unlabeled data: load, clean, split.

        Returns:
            Dictionary with 'train' and 'val' DataFrames.
        """
        # Load data
        df = self.load_unlabeled_data()
        print(f"Loaded {len(df)} raw polymer SMILES")

        # Clean and filter
        df = self.clean_and_filter(df)
        print(f"After cleaning: {len(df)} valid p-SMILES")

        # Compute SA scores
        df = self.compute_sa_scores(df)

        # Split
        train_df, val_df = self.split_unlabeled(df)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")

        return {'train': train_df, 'val': val_df}

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute statistics for a dataset.

        Args:
            df: DataFrame with data.

        Returns:
            Dictionary of statistics.
        """
        stats = {
            'count': len(df),
            'unique_smiles': df['p_smiles'].nunique()
        }

        # Length statistics (round floats to 4 decimal places)
        lengths = df['p_smiles'].str.len()
        stats['length_mean'] = round(float(lengths.mean()), 4)
        stats['length_std'] = round(float(lengths.std()), 4)
        stats['length_min'] = int(lengths.min())
        stats['length_max'] = int(lengths.max())

        # SA score statistics (round floats to 4 decimal places)
        if 'sa_score' in df.columns:
            sa = df['sa_score'].dropna()
            stats['sa_mean'] = round(float(sa.mean()), 4)
            stats['sa_std'] = round(float(sa.std()), 4)
            stats['sa_min'] = round(float(sa.min()), 4)
            stats['sa_max'] = round(float(sa.max()), 4)

        return stats
