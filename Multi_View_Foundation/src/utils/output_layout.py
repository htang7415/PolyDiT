"""Output layout helpers for MVF pipeline steps.

Provides a consistent per-step structure:
  <results_dir>/<step_name>/{files,metrics,figures}
or, for property-scoped artifacts:
  <results_dir>/<step_name>/<property>/{files,metrics,figures}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.utils.property_names import normalize_property_name


def _normalize_optional_property_name(property_name: Optional[str]) -> Optional[str]:
    if property_name is None:
        return None
    text = normalize_property_name(property_name)
    if not text:
        return None
    text = text.replace("/", "_").replace("\\", "_").strip()
    return text or None


def ensure_step_dirs(results_dir: Path, step_name: str, property_name: Optional[str] = None) -> dict:
    root_step_dir = Path(results_dir) / step_name
    property_token = _normalize_optional_property_name(property_name)
    step_dir = root_step_dir / property_token if property_token else root_step_dir
    files_dir = step_dir / "files"
    metrics_dir = step_dir / "metrics"
    figures_dir = step_dir / "figures"
    for directory in (root_step_dir, step_dir, files_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "root_step_dir": root_step_dir,
        "step_dir": step_dir,
        "files_dir": files_dir,
        "metrics_dir": metrics_dir,
        "figures_dir": figures_dir,
        "property_name": property_token,
    }


def save_csv(df, primary_path: Path, index: bool = False) -> None:
    path = Path(primary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_json(payload, primary_path: Path, indent: int = 2) -> None:
    path = Path(primary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
