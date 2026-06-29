"""Minimal YAML config helpers for Multi_View_Foundation."""

from pathlib import Path
from typing import Dict

import yaml


def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
