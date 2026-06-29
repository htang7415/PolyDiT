"""Shared runtime helpers used across MVF step scripts."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Optional


def resolve_path(path_str: str | Path, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path)


def resolve_with_base(path_str: str | Path, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path)


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid integer value.")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def load_module(module_name: str, path: Path, repo_root: Path | None = None):
    module_path = Path(path).resolve()

    import_name = None
    if repo_root is not None:
        try:
            rel = module_path.relative_to(Path(repo_root).resolve())
            if rel.suffix == ".py":
                parts = list(rel.with_suffix("").parts)
                if parts and parts[-1] == "__init__":
                    parts = parts[:-1]
                if parts and all(part.isidentifier() for part in parts):
                    import_name = ".".join(parts)
        except Exception:
            import_name = None

    if import_name:
        try:
            return importlib.import_module(import_name)
        except Exception:
            pass

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
