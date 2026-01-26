#!/usr/bin/env python
"""F0: Build paired dataset across views (initial implementation).

Currently builds a paired index with p-SMILES for D1 (SMiPoly) and D2 (PolyInfo).
"""

import argparse
import gzip
from pathlib import Path
import sys
import importlib.util

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.data.view_converters import smiles_to_selfies, smiles_to_group_selfies


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_group_tokenizer(tokenizer_path: Path):
    module = _load_module(
        "group_selfies_tokenizer",
        REPO_ROOT / "Bi_Diffusion_Group_SELFIES" / "src" / "data" / "tokenizer.py",
    )
    tokenizer_cls = getattr(module, "GroupSELFIESTokenizer")
    return tokenizer_cls.load(str(tokenizer_path))


def _load_smiles(path: Path) -> pd.Series:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)

    if "SMILES" in df.columns:
        return df["SMILES"].astype(str)
    if "p_smiles" in df.columns:
        return df["p_smiles"].astype(str)
    raise ValueError(f"SMILES column not found in {path}")


def main(args):
    config = load_config(args.config)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, results_dir / "config_used.yaml")

    d1_path = _resolve_path(config["paths"]["polymer_file"])
    d2_path = _resolve_path(config["paths"].get("polymer_file_d2", "../Data/Polymer/PolyInfo_Homopolymer.csv"))

    d1_smiles = _load_smiles(d1_path)
    d2_smiles = _load_smiles(d2_path)

    max_d1 = config.get("data", {}).get("max_samples_d1")
    max_d2 = config.get("data", {}).get("max_samples_d2")
    if max_d1:
        d1_smiles = d1_smiles.head(int(max_d1))
    if max_d2:
        d2_smiles = d2_smiles.head(int(max_d2))

    views_cfg = config.get("views", {})
    selfies_enabled = views_cfg.get("selfies", {}).get("enabled", False)
    group_enabled = views_cfg.get("group_selfies", {}).get("enabled", False)
    group_tokenizer_path = views_cfg.get("group_selfies", {}).get("tokenizer_path")
    group_tokenizer = None
    if group_enabled and group_tokenizer_path:
        group_tokenizer_path = _resolve_path(group_tokenizer_path)
        if group_tokenizer_path.exists():
            group_tokenizer = _load_group_tokenizer(group_tokenizer_path)
        else:
            print(f"Warning: group selfies tokenizer not found at {group_tokenizer_path}")

    records = []
    stats_rows = []

    def _append_stats(dataset_name: str, view_name: str, total: int, success: int) -> None:
        stats_rows.append({
            "dataset": dataset_name,
            "view": view_name,
            "total": int(total),
            "success": int(success),
            "failure": int(total - success),
        })

    def _convert_selfies(smiles_list):
        selfies_list = []
        success = 0
        for smi in smiles_list:
            if not selfies_enabled:
                selfies_list.append("")
                continue
            s = smiles_to_selfies(smi)
            if s:
                success += 1
                selfies_list.append(s)
            else:
                selfies_list.append("")
        return selfies_list, success

    def _convert_group_selfies(smiles_list):
        gs_list = []
        success = 0
        for smi in smiles_list:
            if not group_enabled or group_tokenizer is None:
                gs_list.append("")
                continue
            gsf = smiles_to_group_selfies(smi, tokenizer=group_tokenizer)
            if gsf:
                success += 1
                gs_list.append(gsf)
            else:
                gs_list.append("")
        return gs_list, success

    d1_selfies, d1_selfies_success = _convert_selfies(d1_smiles)
    d2_selfies, d2_selfies_success = _convert_selfies(d2_smiles)
    d1_group, d1_group_success = _convert_group_selfies(d1_smiles)
    d2_group, d2_group_success = _convert_group_selfies(d2_smiles)

    if selfies_enabled:
        _append_stats("d1", "selfies", len(d1_smiles), d1_selfies_success)
        _append_stats("d2", "selfies", len(d2_smiles), d2_selfies_success)
    if group_enabled:
        _append_stats("d1", "group_selfies", len(d1_smiles), d1_group_success)
        _append_stats("d2", "group_selfies", len(d2_smiles), d2_group_success)

    for i, smi in enumerate(d1_smiles):
        records.append({
            "polymer_id": f"D1_{i}",
            "dataset": "d1",
            "p_smiles": smi,
            "selfies": d1_selfies[i] if i < len(d1_selfies) else "",
            "group_selfies": d1_group[i] if i < len(d1_group) else "",
            "graph": "",
        })
    for i, smi in enumerate(d2_smiles):
        records.append({
            "polymer_id": f"D2_{i}",
            "dataset": "d2",
            "p_smiles": smi,
            "selfies": d2_selfies[i] if i < len(d2_selfies) else "",
            "group_selfies": d2_group[i] if i < len(d2_group) else "",
            "graph": "",
        })

    paired_df = pd.DataFrame(records)
    paired_index_path = _resolve_path(config["paths"].get("paired_index", str(results_dir / "paired_index.csv")))
    paired_index_path.parent.mkdir(parents=True, exist_ok=True)
    paired_df.to_csv(paired_index_path, index=False)

    stats_rows.insert(0, {"dataset": "d1", "view": "p_smiles", "total": int(len(d1_smiles)), "success": int(len(d1_smiles)), "failure": 0})
    stats_rows.insert(1, {"dataset": "d2", "view": "p_smiles", "total": int(len(d2_smiles)), "success": int(len(d2_smiles)), "failure": 0})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(results_dir / "conversion_stats.csv", index=False)

    print(f"Saved paired_index.csv with {len(paired_df)} records at {paired_index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    main(parser.parse_args())
