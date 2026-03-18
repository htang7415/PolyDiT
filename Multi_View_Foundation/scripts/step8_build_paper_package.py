#!/usr/bin/env python
"""F8: Build a paper-ready package from F1-F7 outputs and 5-method baselines."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

import pandas as pd
import numpy as np

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils.config import load_config
from src.utils.property_names import (
    normalize_property_name as shared_normalize_property_name,
    ordered_properties,
)
from src.utils.runtime import resolve_path as _shared_resolve_path, to_bool as _to_bool
from src.utils.visualization import (
    COLOR_MUTED,
    COLOR_TEXT,
    NATURE_PALETTE,
    normalize_view_name,
    ordered_views,
    set_legend_location,
    view_color,
    view_label,
)


DEFAULT_METHOD_DIRS = [
    "Bi_Diffusion_SMILES",
    "Bi_Diffusion_SMILES_BPE",
    "Bi_Diffusion_SELFIES",
    "Bi_Diffusion_Group_SELFIES",
    "Bi_Diffusion_graph",
]
MANUSCRIPT_FIGURE_COUNT = 6
SUPPORTING_INFORMATION_FIGURE_COUNT = 9

FIGURE_FALLBACK_ORDER = [
    "base_step0",
    "base_step12",
    "mvf_f1",
    "mvf_f2",
    "mvf_f3",
    "mvf_f4",
    "mvf_f5",
    "mvf_f6",
    "mvf_f7",
    "misc",
]

MODEL_SIZE_ORDER = ["small", "medium", "large", "xl", "base", "unknown"]
STEP_LEAF_DIR_NAMES = {"files", "metrics", "figures"}
PROPERTY_PRIORITY = [
    "Tg",
    "Tm",
    "Td",
    "Eg",
    "Ced",
    "Ea",
    "Eib",
    "In",
]
STEP2_METRIC_COLUMNS = ["step2_validity", "step2_uniqueness", "step2_novelty", "step2_diversity"]
METHOD_PLOT_COLORS = {
    "Bi_Diffusion_SMILES": "#3C5488",
    "Bi_Diffusion_SMILES_BPE": "#4DBBD5",
    "Bi_Diffusion_SELFIES": "#00A087",
    "Bi_Diffusion_Group_SELFIES": "#E64B35",
    "Bi_Diffusion_graph": "#8491B4",
}
MANUSCRIPT_CAPTIONS = [
    "Figure 1. All five representation-specific generators produce valid polymers, but their diversity and novelty profiles differ, motivating a multi-view comparison. (a) Valid polymer fraction for each representation at the best-performing model size. (b) Unique fraction among valid generated polymers. (c) Validity-uniqueness trade-off colored by novelty.",
    "Figure 2. Cross-view alignment is strong enough to compare the same polymer across representation spaces. (a) Recall@1 heatmap: fraction of queries for which the correct paired molecule ranks first under cosine similarity. (b) Recall@10 heatmap: fraction of correct pairs recovered within the top-10 retrieved candidates.",
    "Figure 3. Different polymer views provide complementary embedding structure rather than a single uniformly best representation. (a) Mean cross-view Recall@10 across available model sizes or, when only one size is available, across views directly. (b) Embedding-space property smoothness across available model sizes or across views directly.",
    "Figure 4. Fusing aligned views improves downstream property prediction over the strongest single-view alternative. (a) Test-set R^2 for the best single-view reference and the fused multi-view representation across configured target properties or model sizes, depending on available checkpoints. (b) Fusion gain in R^2, computed as fused multi-view performance minus the best single-view reference.",
    "Figure 5. Inverse design success generalizes across target properties rather than a single polymer objective. (a) Fair hit rate for each proposal view across available properties, aggregating over model sizes when multiple sizes are present. (b) Top-k fair hit rate under the same multi-property comparison.",
    "Figure 6. Accepted inverse designs remain chemically plausible under descriptor, physics-rule, motif, and nearest-neighbor checks. (a) Normalized descriptor shifts relative to the reference set. (b) Physics-rule consistency rate across properties. (c) Motif enrichment of accepted candidates relative to the reference distribution. (d) Nearest-neighbor similarity to known polymers.",
]

SI_CAPTIONS = [
    "Figure S1. Distribution of sequence length and synthetic accessibility (SA) score in training and validation sets for each polymer representation.",
    "Figure S2. Training convergence and generation quality across representations and model sizes.",
    "Figure S3. Multi-view embedding extraction summary: embedding dimensionalities, model sizes, and sample counts per view.",
    "Figure S4. Cross-view retrieval evaluation: full Recall@K heatmaps across all view pairs.",
    "Figure S5. Property-head training diagnostics supporting Figure 4: per-split metrics, head leaderboard, and coverage across target properties.",
    "Figure S6. Embedding-research diagnostics supporting Figure 3: tokenizer efficiency, geometry summaries, and semantic-structure metrics across views.",
    "Figure S7. Foundation-guided inverse-design diagnostics supporting Figure 5: candidate score distributions and accepted-candidate profiles by view.",
    "Figure S8. F6 DiT interpretability diagnostics complementing Figures 3-5: integrated gradients, gradient-times-hidden, and attention-rollout analyses of the shared SMILES scorer, with faithfulness summaries across proposal views and outcome groups.",
    "Figure S9. Chemistry/physics diagnostics supporting Figure 6: per-property descriptor distributions, physics consistency checks, and nearest-neighbor explanations.",
]

STEP_EXPORT_SPECS = {
    "F1": {
        "step_name": "alignment_embeddings",
        "step_subdir": "step1_alignment_embeddings",
        "theme": "mvf_f1",
        "core_patterns": ["embedding_meta*.json", "embedding_index_*.csv", "embeddings_*.npy", "figure_f1*.png"],
        "expected_outputs": ["embedding_meta_*.json", "embedding_index_*.csv", "embeddings_*.npy", "figure_f1_*.png"],
        "label": "MVF F1 embedding extraction",
    },
    "F2": {
        "step_name": "retrieval",
        "step_subdir": "step2_retrieval",
        "theme": "mvf_f2",
        "core_patterns": ["metrics_alignment.csv", "figure_f2*.png"],
        "expected_outputs": ["metrics_alignment.csv", "figure_f2_*.png"],
        "label": "MVF F2 retrieval evaluation",
    },
    "F3": {
        "step_name": "property_heads",
        "step_subdir": "step3_property",
        "theme": "mvf_f3",
        "core_patterns": ["metrics_property.csv", "figure_f3*.png"],
        "expected_outputs": ["metrics_property.csv", "figure_f3_*.png"],
        "label": "MVF F3 property-head training",
    },
    "F4": {
        "step_name": "embedding_research",
        "step_subdir": "step4_embedding_research",
        "theme": "mvf_f4",
        "core_patterns": ["metrics_embedding_research.csv", "view_*summary.csv", "figure_f4*.png"],
        "expected_outputs": ["metrics_embedding_research.csv", "view_*summary.csv", "figure_f4_*.png"],
        "label": "MVF F4 embedding research",
    },
    "F5": {
        "step_name": "foundation_inverse",
        "step_subdir": "step5_foundation_inverse",
        "theme": "mvf_f5",
        "core_patterns": [
            "metrics_inverse.csv",
            "metrics_view_compare*.csv",
            "accepted_candidates*.csv",
            "candidate_scores*.csv",
            "figure_f5*.png",
        ],
        "expected_outputs": ["metrics_inverse.csv", "metrics_view_compare*.csv", "accepted_candidates*.csv", "figure_f5_*.png"],
        "label": "MVF F5 inverse design",
    },
    "F6": {
        "step_name": "dit_interpretability",
        "step_subdir": "step6_dit_interpretability",
        "theme": "mvf_f6",
        "core_patterns": ["metrics_dit_interpretability*.csv", "dit_*summary*.csv", "figure_f6*.png"],
        "expected_outputs": ["metrics_dit_interpretability*.csv", "dit_*summary*.csv", "figure_f6_*.png"],
        "label": "MVF F6 interpretability",
    },
    "F7": {
        "step_name": "chem_physics_analysis",
        "step_subdir": "step7_chem_physics_analysis",
        "theme": "mvf_f7",
        "core_patterns": ["metrics_chem_physics.csv", "descriptor_shifts*.csv", "motif_enrichment*.csv", "figure_f7*.png"],
        "expected_outputs": ["metrics_chem_physics.csv", "descriptor_shifts*.csv", "motif_enrichment*.csv", "figure_f7_*.png"],
        "label": "MVF F7 chemistry/physics analysis",
    },
}


def _resolve_path(path_str: str) -> Path:
    return _shared_resolve_path(path_str, BASE_DIR)


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _to_int(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _normalize_property_name(value) -> str:
    return shared_normalize_property_name(value)


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _existing_dirs(paths: Iterable[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def _copy_file(
    src: Path,
    dst: Path,
    *,
    category: str,
    copied: list[dict],
    results_dir: Path,
    output_dir: Path,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(
        {
            "category": category,
            "source": _safe_rel(src, results_dir),
            "destination": _safe_rel(dst, output_dir),
        }
    )


def _copy_first(
    candidates: Iterable[Path],
    dst: Path,
    *,
    category: str,
    copied: list[dict],
    missing: list[dict],
    results_dir: Path,
    output_dir: Path,
    missing_label: str,
) -> Optional[Path]:
    src = _first_existing(candidates)
    if src is None:
        missing.append({"category": category, "name": missing_label})
        return None
    _copy_file(src, dst, category=category, copied=copied, results_dir=results_dir, output_dir=output_dir)
    return src


def _collect_glob_unique(source_dirs: Iterable[Path], patterns: Iterable[str]) -> list[Path]:
    collected: list[Path] = []
    seen_paths: set[str] = set()
    for directory in source_dirs:
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                key = str(path.resolve())
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                collected.append(path)
    return collected


def _iter_property_scoped_dirs(step_root: Path) -> list[Path]:
    if not step_root.exists() or not step_root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(step_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.lower() in STEP_LEAF_DIR_NAMES:
            continue
        if p.name.startswith("."):
            continue
        out.append(p)
    return out


def _collect_step_artifact_dirs(
    *,
    mvf_results_dir: Path,
    step_subdir: str,
    include_property_scopes: bool = False,
    include_root_when_property_scopes: bool = False,
) -> list[Path]:
    step_root = mvf_results_dir / step_subdir
    dirs: list[Path] = []
    property_scopes = _iter_property_scoped_dirs(step_root) if include_property_scopes else []
    if property_scopes:
        for prop_dir in property_scopes:
            dirs.extend(_existing_dirs([prop_dir / "files", prop_dir]))
        if include_root_when_property_scopes:
            dirs.extend(_existing_dirs([step_root / "files", step_root]))
    else:
        dirs.extend(_existing_dirs([step_root / "files", step_root]))
    return dirs


def _load_step_scope_csv(scope_dir: Path, filename: str) -> pd.DataFrame:
    candidates = [
        scope_dir / "metrics" / filename,
        scope_dir / "files" / filename,
        scope_dir / filename,
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return pd.read_csv(path)
        except Exception:
            continue
    return pd.DataFrame()


def _tag_mvf_frame(df: pd.DataFrame, res_dir: Path, property_hint: Optional[str] = None) -> pd.DataFrame:
    tagged = df.copy()
    inferred = _normalize_model_size(_infer_model_size_from_results_dir(res_dir))
    if "model_size" in tagged.columns:
        tagged["model_size"] = tagged["model_size"].apply(_normalize_model_size)
        tagged.loc[tagged["model_size"].isin(["", "unknown", "nan", "none"]), "model_size"] = inferred
    else:
        tagged["model_size"] = inferred
    tagged["source_results_dir"] = res_dir.name

    prop_hint = _normalize_property_name(property_hint) if property_hint is not None else ""
    if prop_hint:
        if "property" not in tagged.columns:
            tagged["property"] = prop_hint
        else:
            prop_series = tagged["property"].astype(str).str.strip()
            missing_mask = (
                tagged["property"].isna()
                | prop_series.eq("")
                | prop_series.str.lower().isin({"nan", "none", "null"})
            )
            tagged.loc[missing_mask, "property"] = prop_hint
        tagged["source_property_scope"] = prop_hint
    return tagged


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _f1_embedding_summary(results_dir: Path) -> pd.DataFrame:
    meta_candidates = _collect_glob_unique(
        _existing_dirs(
            [
                results_dir / "step1_alignment_embeddings" / "files",
                results_dir,
            ]
        ),
        ["embedding_meta_*.json"],
    )

    rows = []
    for path in meta_candidates:
        try:
            meta = _read_json(path)
        except Exception:
            continue
        rows.append(
            {
                "view": str(meta.get("view", "")).strip(),
                "model_size": str(meta.get("model_size", "")).strip(),
                "embedding_dim": meta.get("embedding_dim", None),
                "d1_samples": meta.get("d1_samples", None),
                "d2_samples": meta.get("d2_samples", None),
                "pooling": str(meta.get("pooling", "")).strip(),
                "timestep": meta.get("timestep", None),
                "device": str(meta.get("device", "")).strip(),
                "d1_time_sec": meta.get("d1_time_sec", None),
                "d2_time_sec": meta.get("d2_time_sec", None),
                "checkpoint_path": str(meta.get("checkpoint_path", "")).strip(),
                "tokenizer_path": str(meta.get("tokenizer_path", "")).strip(),
                "source_meta": str(path),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "view",
                "model_size",
                "embedding_dim",
                "d1_samples",
                "d2_samples",
                "pooling",
                "timestep",
                "device",
                "d1_time_sec",
                "d2_time_sec",
                "checkpoint_path",
                "tokenizer_path",
                "source_meta",
            ]
        )

    df = pd.DataFrame(rows)
    if "view" in df.columns:
        df["view"] = df["view"].astype(str)
        df = df.sort_values("view").reset_index(drop=True)
    return df


def _f1_embedding_summary_multi(results_dirs: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for res_dir in _unique_paths(list(results_dirs)):
        df = _f1_embedding_summary(res_dir)
        if df.empty:
            continue
        out = df.copy()
        inferred = _normalize_model_size(_infer_model_size_from_results_dir(res_dir))
        if "model_size" in out.columns:
            out["model_size"] = out["model_size"].apply(_normalize_model_size)
            out.loc[out["model_size"].isin(["", "unknown", "nan", "none"]), "model_size"] = inferred
        else:
            out["model_size"] = inferred
        out["source_results_dir"] = res_dir.name
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["model_size", "view", "source_results_dir"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    return merged


def _configured_properties(config: dict) -> list[str]:
    prop_cfg = config.get("property", {}) or {}
    files = prop_cfg.get("files")
    if files is None:
        return []
    if isinstance(files, str):
        files = [files]
    props = []
    for item in files:
        name = _normalize_property_name(item)
        if name and name not in props:
            props.append(name)
    return props


def _ordered_properties(values: Iterable[object]) -> list[str]:
    """Return stable property ordering with configured priority first."""
    by_lower: dict[str, str] = {}
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key not in by_lower:
            by_lower[key] = text

    ordered: list[str] = []
    for prop in PROPERTY_PRIORITY:
        key = prop.lower()
        if key in by_lower:
            ordered.append(by_lower.pop(key))

    for key in sorted(by_lower):
        ordered.append(by_lower[key])
    return ordered


def _discover_properties(
    config: dict,
    step5_dirs: list[Path],
    step6_dirs: list[Path],
    step7_metric_path: Optional[Path],
) -> list[str]:
    props = _configured_properties(config)
    prop_set = set(props)

    regexes = [
        re.compile(r"^candidate_scores_(.+)\.csv$"),
        re.compile(r"^accepted_candidates_(.+)\.csv$"),
        re.compile(r"^view_compare_topk_(.+)\.csv$"),
        re.compile(r"^view_compare_scores_(.+)\.csv$"),
        re.compile(r"^metrics_view_compare_(.+)\.csv$"),
        re.compile(r"^metrics_dit_interpretability_(.+)\.csv$"),
        re.compile(r"^ood_objective_topk_(.+)\.csv$"),
        re.compile(r"^ood_objective_scores_(.+)\.csv$"),
        re.compile(r"^metrics_inverse_ood_objective_(.+)\.csv$"),
    ]
    for directory in step5_dirs + step6_dirs:
        for path in directory.glob("*.csv"):
            name = path.name
            for regex in regexes:
                m = regex.match(name)
                if not m:
                    continue
                prop = _normalize_property_name(m.group(1))
                if prop and prop not in prop_set:
                    prop_set.add(prop)
                    props.append(prop)

    if step7_metric_path is not None and step7_metric_path.exists():
        try:
            df = pd.read_csv(step7_metric_path)
            if "property" in df.columns:
                for prop in df["property"].dropna().astype(str).tolist():
                    name = _normalize_property_name(prop)
                    if name and name not in prop_set:
                        prop_set.add(name)
                        props.append(name)
        except Exception:
            pass

    return props


def _iter_method_results_dirs(method_root: Path) -> list[Path]:
    if not method_root.exists() or not method_root.is_dir():
        return []
    dirs = [p for p in sorted(method_root.glob("results*")) if p.is_dir()]
    unique: list[Path] = []
    seen: set[str] = set()
    for p in dirs:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _normalize_model_size(value: object) -> str:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return "unknown"
    for size in MODEL_SIZE_ORDER:
        if size == "unknown":
            continue
        if text == size:
            return size
    for size in MODEL_SIZE_ORDER:
        if size == "unknown":
            continue
        if re.search(rf"(^|[_-]){size}($|[_-])", text):
            return size
    return text


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        if not p.exists() or not p.is_dir():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _parse_results_dirs(value) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, str):
        tokens = [v.strip() for v in value.split(",") if v.strip()]
        return [_resolve_path(token) for token in tokens]
    if isinstance(value, (list, tuple, set)):
        paths: list[Path] = []
        for v in value:
            token = str(v).strip()
            if not token:
                continue
            paths.append(_resolve_path(token))
        return paths
    return []


def _has_mvf_step_artifacts(results_dir: Path) -> bool:
    for step in [
        "step1_alignment_embeddings",
        "step2_retrieval",
        "step3_property",
        "step4_embedding_research",
        "step5_foundation_inverse",
        "step6_dit_interpretability",
        "step6_view_compare_analysis",
        "step6_ood_aware_inverse",
        "step7_chem_physics_analysis",
    ]:
        step_dir = results_dir / step
        if not step_dir.exists() or not step_dir.is_dir():
            continue
        if any((step_dir / leaf).exists() for leaf in STEP_LEAF_DIR_NAMES):
            return True
        if any(step_dir.glob("*.csv")):
            return True
    return False


def _discover_mvf_results_dirs(
    primary_results_dir: Path,
    *,
    explicit_results_dirs: Sequence[Path],
    auto_discover: bool,
) -> list[Path]:
    candidates: list[Path] = [primary_results_dir]
    candidates.extend(explicit_results_dirs)

    if auto_discover:
        for p in sorted(BASE_DIR.glob("results*")):
            if not p.is_dir():
                continue
            name = p.name.lower()
            if "paper_package" in name:
                continue
            if not _has_mvf_step_artifacts(p):
                continue
            candidates.append(p)

    dirs = _unique_paths(candidates)
    if not dirs:
        return [primary_results_dir]

    def _rank(path: Path) -> tuple[int, float, str]:
        size = _normalize_model_size(_infer_model_size_from_results_dir(path))
        size_rank = MODEL_SIZE_ORDER.index(size) if size in MODEL_SIZE_ORDER else len(MODEL_SIZE_ORDER)
        try:
            mtime = -float(path.stat().st_mtime)
        except Exception:
            mtime = 0.0
        return (size_rank, mtime, path.name)

    dirs.sort(key=_rank)
    return dirs


def _copy_base_method_csvs(
    *,
    method_dirs: Sequence[str],
    tables_supp_dir: Path,
    copied: list[dict],
    missing: list[dict],
    results_dir: Path,
    output_dir: Path,
) -> list[Path]:
    copied_sources: list[Path] = []

    aggregate_root = REPO_ROOT / "results"
    if aggregate_root.exists():
        for csv_path in sorted(aggregate_root.glob("aggregate*/*.csv")):
            dst = (
                tables_supp_dir
                / "base_methods"
                / "aggregates"
                / csv_path.parent.name
                / csv_path.name
            )
            _copy_file(
                csv_path,
                dst,
                category="table_supplementary",
                copied=copied,
                results_dir=results_dir,
                output_dir=output_dir,
            )
            copied_sources.append(csv_path)

    if not method_dirs:
        method_dirs = list(DEFAULT_METHOD_DIRS)

    for method_name in method_dirs:
        method_root = _resolve_repo_path(method_name)
        result_roots = _iter_method_results_dirs(method_root)
        if not result_roots:
            missing.append(
                {
                    "category": "table_supplementary",
                    "name": f"missing_method_results:{method_name}",
                }
            )
            continue

        for res_dir in result_roots:
            candidates: list[Path] = []
            candidates.extend(sorted((res_dir / "step0_data_prep" / "metrics").glob("*.csv")))
            candidates.extend(sorted((res_dir / "step1_backbone" / "metrics").glob("*.csv")))
            candidates.extend(sorted((res_dir / "step2_sampling" / "metrics").glob("*.csv")))
            candidates.extend(sorted((res_dir / "step2_sampling" / "files").glob("*.csv")))
            seen: set[str] = set()
            unique_candidates: list[Path] = []
            for c in candidates:
                key = str(c.resolve())
                if key in seen:
                    continue
                seen.add(key)
                unique_candidates.append(c)

            for csv_path in unique_candidates:
                rel = csv_path.relative_to(res_dir)
                dst = tables_supp_dir / "base_methods" / method_name / res_dir.name / rel
                _copy_file(
                    csv_path,
                    dst,
                    category="table_supplementary",
                    copied=copied,
                    results_dir=results_dir,
                    output_dir=output_dir,
                )
                copied_sources.append(csv_path)

    return copied_sources


def _theme_bucket_for_mvf_figure(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("figure_f1_"):
        return "mvf_f1"
    if name.startswith("figure_f2_"):
        return "mvf_f2"
    if name.startswith("figure_f3_"):
        return "mvf_f3"
    if name.startswith("figure_f4_"):
        return "mvf_f4"
    if name.startswith("figure_f5_"):
        return "mvf_f5"
    if name.startswith("figure_f6_"):
        return "mvf_f6"
    if name.startswith("figure_f7_"):
        return "mvf_f7"
    return "misc"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _collect_source_panels(
    *,
    results_dirs: Sequence[Path],
    output_dir: Path,
    method_dirs: Sequence[str],
) -> dict[str, list[Path]]:
    themed: dict[str, list[Path]] = {
        "base_step0": [],
        "base_step12": [],
        "mvf_f1": [],
        "mvf_f2": [],
        "mvf_f3": [],
        "mvf_f4": [],
        "mvf_f5": [],
        "mvf_f6": [],
        "mvf_f7": [],
        "misc": [],
    }

    seen: set[str] = set()

    def add(theme: str, path: Path) -> None:
        if not path.exists() or path.suffix.lower() != ".png":
            return
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        themed.setdefault(theme, []).append(path)

    for method_name in (method_dirs or list(DEFAULT_METHOD_DIRS)):
        method_root = _resolve_repo_path(method_name)
        for res_dir in _iter_method_results_dirs(method_root):
            step0_fig_dir = res_dir / "step0_data_prep" / "figures"
            if step0_fig_dir.exists():
                for png in sorted(step0_fig_dir.glob("*.png")):
                    add("base_step0", png)

            for step12_name in ["step1_backbone", "step2_sampling"]:
                fig_dir = res_dir / step12_name / "figures"
                if not fig_dir.exists():
                    continue
                for png in sorted(fig_dir.glob("*.png")):
                    add("base_step12", png)

    aggregate_root = REPO_ROOT / "results"
    if aggregate_root.exists():
        for png in sorted(aggregate_root.glob("aggregate*/figures/*.png")):
            add("base_step12", png)

    for mvf_results_dir in _unique_paths(list(results_dirs)):
        for png in sorted(mvf_results_dir.rglob("*.png")):
            if _is_relative_to(png, output_dir):
                continue
            parts_lower = {part.lower() for part in png.parts}
            if any(part.startswith("paper_package") for part in parts_lower):
                continue
            bucket = _theme_bucket_for_mvf_figure(png)
            add(bucket, png)

    return themed


def _panel_label(index: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    if index < len(letters):
        return f"({letters[index]})"
    return f"({index + 1})"


def _format_source_name(path: Path) -> str:
    text = path.stem
    text = re.sub(r"^figure_", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _describe_panel(path: Path) -> str:
    rel = _safe_rel(path, REPO_ROOT)
    source = _format_source_name(path)

    step_hint = ""
    lower_rel = rel.lower()
    if "step0_data_prep" in lower_rel:
        step_hint = "Step0 data-prep"
    elif "step1_backbone" in lower_rel:
        step_hint = "Step1 backbone"
    elif "step2_sampling" in lower_rel:
        step_hint = "Step2 sampling"
    elif "step1_alignment_embeddings" in lower_rel:
        step_hint = "MVF F1"
    elif "step2_retrieval" in lower_rel:
        step_hint = "MVF F2"
    elif "step3_property" in lower_rel:
        step_hint = "MVF F3"
    elif "step4_embedding_research" in lower_rel or "step4_ood" in lower_rel:
        step_hint = "MVF F4"
    elif "step5_foundation_inverse" in lower_rel:
        step_hint = "MVF F5"
    elif "step6_dit_interpretability" in lower_rel or "step6_view_compare_analysis" in lower_rel or "step6_ood_aware_inverse" in lower_rel:
        step_hint = "MVF F6"
    elif "step7_chem_physics_analysis" in lower_rel:
        step_hint = "MVF F7"

    method_hint = ""
    rel_norm = "/" + lower_rel.replace("\\", "/").strip("/") + "/"
    for method_name in sorted(DEFAULT_METHOD_DIRS, key=len, reverse=True):
        token = "/" + method_name.lower().strip("/") + "/"
        if token in rel_norm:
            method_hint = method_name.replace("Bi_Diffusion_", "").replace("_", " ")
            break

    parts = [x for x in [step_hint, method_hint, source] if x]
    if not parts:
        return source
    return ": ".join(parts)


def _set_publication_style(font_size: int) -> None:
    if plt is None or matplotlib is None:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "figure.constrained_layout.use": True,
            "legend.frameon": False,
            "axes.prop_cycle": matplotlib.cycler(color=NATURE_PALETTE),
        }
    )


def _nature_sequential_cmap():
    if plt is None:
        return None
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "paper_nature_seq",
        ["#F7FBFF", NATURE_PALETTE[1], NATURE_PALETTE[3]],
    )


def _nature_diverging_cmap():
    if plt is None:
        return None
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "paper_nature_div",
        [NATURE_PALETTE[3], "#F7F7F7", NATURE_PALETTE[0]],
    )


def _wrap_ticklabels(ax, axis: str = "x", width: int = 16, rotation: int = 32) -> None:
    ticks = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    updated = []
    needs_rotation = False
    for tick in ticks:
        text = str(tick.get_text())
        if not text:
            updated.append(text)
            continue
        words = text.replace("_", " ").split()
        lines: list[str] = []
        current = ""
        for word in words:
            proposal = word if not current else f"{current} {word}"
            if len(proposal) <= width:
                current = proposal
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        wrapped = "\n".join(lines) if lines else text
        if "\n" in wrapped or len(text) > width:
            needs_rotation = True
        updated.append(wrapped)
    if axis == "x":
        ax.set_xticklabels(updated, rotation=rotation if needs_rotation else 0, ha="right" if needs_rotation else "center")
    else:
        ax.set_yticklabels(updated)


def _compose_multi_panel_figure(
    *,
    panel_paths: Sequence[Path],
    output_path: Path,
    font_size: int,
    dpi: int,
) -> bool:
    if plt is None:
        return False

    panels = list(panel_paths)
    n = len(panels)
    ncols = 2 if n > 1 else 1
    nrows = max(1, int(math.ceil(max(n, 1) / ncols)))

    panel_aspects: list[float] = []
    for panel_path in panels:
        try:
            img = plt.imread(panel_path)
            h, w = img.shape[0], img.shape[1]
            if h > 0 and w > 0:
                panel_aspects.append(float(w) / float(h))
        except Exception:
            continue
    median_aspect = float(np.median(np.asarray(panel_aspects, dtype=float))) if panel_aspects else 1.44
    cell_h = 5.2
    cell_w = max(5.8, min(7.6, cell_h * median_aspect))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(cell_w * ncols, cell_h * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    if hasattr(axes, "ravel"):
        ax_list = list(axes.ravel())
    else:
        ax_list = [axes]

    for i, ax in enumerate(ax_list):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        if i >= n:
            continue

        panel_path = panels[i]
        try:
            img = plt.imread(panel_path)
            ax.imshow(img)
            ax.set_aspect("equal")
        except Exception:
            ax.text(
                0.5,
                0.5,
                f"Unable to load panel:\n{panel_path.name}",
                ha="center",
                va="center",
                fontsize=font_size,
                transform=ax.transAxes,
            )

        ax.text(
            0.01,
            0.99,
            _panel_label(i),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=font_size,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
        )

    if n == 0:
        ax = ax_list[0]
        ax.text(
            0.5,
            0.5,
            "No source panels available for this figure.",
            ha="center",
            va="center",
            fontsize=font_size,
            transform=ax.transAxes,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_caption_file(*, caption_lines: Sequence[str], caption_path: Path) -> None:
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(line.strip() for line in caption_lines if str(line).strip())
    if not text:
        text = "No figure captions were generated."
    caption_path.write_text(text + "\n", encoding="utf-8")


def _flatten_themed_panels(themed: dict[str, list[Path]], theme_order: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for theme in theme_order:
        for p in themed.get(theme, []):
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def _rolling_panel_window(panels: Sequence[Path], start: int, count: int) -> list[Path]:
    items = list(panels)
    if not items:
        return []
    n = len(items)
    k = min(max(1, int(count)), n)
    return [items[(start + i) % n] for i in range(k)]


def _pick_panels(
    themed: dict[str, list[Path]],
    theme_group: Sequence[str],
    used: set[str],
    max_panels: int,
    *,
    allow_reuse: bool = False,
    fallback_order: Optional[Sequence[str]] = None,
) -> list[Path]:
    out: list[Path] = []
    order = list(fallback_order) if fallback_order is not None else list(FIGURE_FALLBACK_ORDER)

    def _collect_from(themes: Sequence[str], *, enforce_used: bool) -> bool:
        for theme in themes:
            for panel in themed.get(theme, []):
                key = str(panel.resolve())
                if enforce_used and key in used:
                    continue
                if panel in out:
                    continue
                out.append(panel)
                if len(out) >= max_panels:
                    return True
        return len(out) >= max_panels

    if _collect_from(theme_group, enforce_used=True):
        return out
    if _collect_from(order, enforce_used=True):
        return out

    if allow_reuse:
        if _collect_from(theme_group, enforce_used=False):
            return out
        _collect_from(order, enforce_used=False)
    return out


def _copy_tree_files(
    *,
    source_dir: Path,
    destination_dir: Path,
    category: str,
    copied: list[dict],
    results_dir: Path,
    output_dir: Path,
) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        return
    for src in sorted(source_dir.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(source_dir)
        dst = destination_dir / rel
        _copy_file(
            src,
            dst,
            category=category,
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )


def _path_matches_patterns(path: Path, patterns: Sequence[str]) -> bool:
    path_name = path.name
    path_text = str(path).replace("\\", "/")
    for pattern in patterns:
        norm = str(pattern).replace("\\", "/")
        if fnmatch.fnmatch(path_name, norm):
            return True
        if fnmatch.fnmatch(path_text, f"*{norm}"):
            return True
    return False


def _file_has_substance(path: Path) -> bool:
    try:
        size = int(path.stat().st_size)
    except Exception:
        return False
    if size <= 0:
        return False

    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            df = pd.read_csv(path)
        except Exception:
            return size > 1
        return not df.empty
    if suffix == ".json":
        try:
            payload = _read_json(path)
        except Exception:
            return size > 2
        if isinstance(payload, dict):
            return bool(payload)
        if isinstance(payload, list):
            return bool(payload)
        return True
    return True


def _inventory_step_artifacts(
    *,
    results_dirs: Sequence[Path],
    step_id: str,
    output_dir: Path,
    results_dir: Path,
) -> dict:
    spec = STEP_EXPORT_SPECS[step_id]
    step_subdir = spec["step_subdir"]
    step_roots: list[Path] = []
    all_files: list[Path] = []
    for res_dir in _unique_paths(list(results_dirs)):
        step_root = res_dir / step_subdir
        if not step_root.exists() or not step_root.is_dir():
            step_root = None
        else:
            step_roots.append(step_root)
            for path in sorted(step_root.rglob("*")):
                if not path.is_file():
                    continue
                if _is_relative_to(path, output_dir):
                    continue
                all_files.append(path)

        for path in sorted(res_dir.glob("*")):
            if not path.is_file():
                continue
            if _is_relative_to(path, output_dir):
                continue
            if path.name == "config_used.yaml" or _path_matches_patterns(path, spec["core_patterns"]):
                all_files.append(path)

    config_files = [p for p in all_files if p.name == "config_used.yaml"]
    non_config_files = [p for p in all_files if p.name != "config_used.yaml"]
    figure_files = [p for p in non_config_files if p.suffix.lower() == ".png"]
    core_files = [p for p in non_config_files if _path_matches_patterns(p, spec["core_patterns"])]
    nonempty_core_files = [p for p in core_files if _file_has_substance(p)]

    if nonempty_core_files or figure_files:
        status = "completed"
    elif step_roots and (config_files or non_config_files):
        status = "partial"
    else:
        status = "missing"

    sample_paths = [
        _safe_rel(p, results_dir)
        for p in sorted(non_config_files, key=lambda x: str(x))[:6]
    ]
    primary_artifact = _safe_rel(nonempty_core_files[0], results_dir) if nonempty_core_files else ""
    return {
        "step_id": step_id,
        "step_name": spec["step_name"],
        "step_subdir": step_subdir,
        "label": spec["label"],
        "theme": spec["theme"],
        "status": status,
        "artifact_roots": [_safe_rel(p, results_dir) for p in step_roots],
        "config_file_count": len(config_files),
        "non_config_file_count": len(non_config_files),
        "figure_file_count": len(figure_files),
        "core_file_count": len(core_files),
        "nonempty_core_file_count": len(nonempty_core_files),
        "sample_artifacts": sample_paths,
        "source_metric": primary_artifact,
        "expected_outputs": list(spec["expected_outputs"]),
    }


def _build_step_status_rows(
    *,
    results_dirs: Sequence[Path],
    output_dir: Path,
    results_dir: Path,
    paper_table_map: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    for step_id in ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]:
        inventory = _inventory_step_artifacts(
            results_dirs=results_dirs,
            step_id=step_id,
            output_dir=output_dir,
            results_dir=results_dir,
        )
        inventory["paper_table"] = paper_table_map.get(step_id, "")
        rows.append(inventory)
    return rows


def _placeholder_lines_for_step_ids(step_ids: Sequence[str], step_status_rows: dict[str, dict]) -> list[str]:
    if not step_ids:
        return ["No source panels or non-empty metrics were available during F8 export."]
    if len(step_ids) == 1:
        step_id = step_ids[0]
        summary = step_status_rows.get(step_id, {})
        spec = STEP_EXPORT_SPECS.get(step_id, {})
        return [
            f"No non-empty {spec.get('label', step_id)} outputs were found.",
            f"Observed status: {summary.get('status', 'missing')}.",
            "Expected one of: " + ", ".join(spec.get("expected_outputs", [])),
            (
                "Observed artifacts: "
                f"non-config={summary.get('non_config_file_count', 0)}, "
                f"png={summary.get('figure_file_count', 0)}, "
                f"nonempty_core={summary.get('nonempty_core_file_count', 0)}."
            ),
        ]

    lines = ["Required MVF artifacts were incomplete for this figure."]
    for step_id in step_ids:
        summary = step_status_rows.get(step_id, {})
        spec = STEP_EXPORT_SPECS.get(step_id, {})
        lines.append(
            f"{step_id}: {spec.get('label', step_id)} status={summary.get('status', 'missing')}, "
            f"nonempty_core={summary.get('nonempty_core_file_count', 0)}, png={summary.get('figure_file_count', 0)}."
        )
    return lines


def _render_placeholder_figure(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    title: str,
    lines: Sequence[str],
) -> bool:
    if plt is None:
        return False

    wrapped_lines: list[str] = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(str(line), width=62) or [""])

    fig, ax = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(0.03, 0.93, title, ha="left", va="top", fontsize=font_size + 2, fontweight="bold", color=COLOR_TEXT)
    y = 0.82
    for line in wrapped_lines:
        ax.text(0.05, y, line, ha="left", va="top", fontsize=max(10, font_size - 2), color=COLOR_TEXT)
        y -= 0.075
        if y < 0.10:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _render_panels_or_placeholder(
    *,
    output_path: Path,
    fallback_panels: Sequence[Path],
    font_size: int,
    dpi: int,
    empty_title: str,
    empty_lines: Sequence[str],
) -> tuple[bool, str]:
    panels = [Path(p) for p in fallback_panels if Path(p).exists()]
    if panels:
        ok = _compose_multi_panel_figure(
            panel_paths=panels,
            output_path=output_path,
            font_size=font_size,
            dpi=dpi,
        )
        return ok, "panels"
    ok = _render_placeholder_figure(
        output_path=output_path,
        font_size=font_size,
        dpi=dpi,
        title=empty_title,
        lines=empty_lines,
    )
    return ok, "placeholder"


def _caption_with_render_note(caption: str, render_mode: str, render_note: str) -> str:
    if render_mode != "placeholder" or not render_note:
        return caption
    return f"{caption} Placeholder export note: {render_note}"


def _build_f1_summary_panel(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    f1_df: pd.DataFrame,
) -> bool:
    if plt is None or f1_df is None or f1_df.empty:
        return False

    df = f1_df.copy()
    if "view" not in df.columns:
        return False
    df["view"] = df["view"].astype(str).map(normalize_view_name)
    df["embedding_dim"] = pd.to_numeric(df.get("embedding_dim"), errors="coerce")
    df["d1_samples"] = pd.to_numeric(df.get("d1_samples"), errors="coerce")
    df["d2_samples"] = pd.to_numeric(df.get("d2_samples"), errors="coerce")
    agg = (
        df.groupby("view", as_index=False)[["embedding_dim", "d1_samples", "d2_samples"]]
        .mean(numeric_only=True)
        .dropna(subset=["embedding_dim", "d1_samples", "d2_samples"], how="all")
    )
    if agg.empty:
        return False

    views = ordered_views(agg["view"].tolist())
    agg = agg.set_index("view").reindex(views).reset_index()
    labels = [view_label(v) for v in agg["view"].tolist()]
    x = np.arange(len(agg))

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.0), squeeze=False)
    ax0, ax1 = axes[0]

    ax0.bar(
        x,
        agg["embedding_dim"].to_numpy(dtype=float),
        color=[view_color(v) for v in agg["view"].tolist()],
        edgecolor=COLOR_TEXT,
        linewidth=0.7,
    )
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=30, ha="right")
    ax0.set_ylabel("Embedding Dimension")
    ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
    _wrap_ticklabels(ax0, axis="x", width=12, rotation=30)
    _panel_mark(ax0, "(a)", font_size)

    width = 0.38
    ax1.bar(
        x - width / 2.0,
        agg["d1_samples"].to_numpy(dtype=float),
        width=width,
        color=NATURE_PALETTE[3],
        edgecolor=COLOR_TEXT,
        linewidth=0.7,
        label="D1",
    )
    ax1.bar(
        x + width / 2.0,
        agg["d2_samples"].to_numpy(dtype=float),
        width=width,
        color=NATURE_PALETTE[0],
        edgecolor=COLOR_TEXT,
        linewidth=0.7,
        label="D2",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Embedded Samples")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="best", frameon=False)
    _wrap_ticklabels(ax1, axis="x", width=12, rotation=30)
    _panel_mark(ax1, "(b)", font_size)

    return _save_plot_figure(fig, output_path, dpi)


def _property_from_suffixed_filename(filename: str) -> str:
    m = re.match(
        r"^(?:candidate_scores|accepted_candidates|accepted_polymer_report|accepted_polymer_summary|"
        r"view_compare_scores|view_compare_topk|metrics_view_compare|metrics_dit_interpretability|"
        r"ood_objective_scores|ood_objective_topk|metrics_inverse_ood_objective|"
        r"descriptor_shifts|motif_enrichment|physics_consistency|nearest_neighbor_explanations|design_filter_audit|"
        r"property_input_files|run_meta)_(.+)\.(?:csv|json)$",
        filename,
    )
    if not m:
        return ""
    return _normalize_property_name(m.group(1))


def _extract_property_scope_from_path(path: Path) -> str:
    parts = list(path.parts)
    for step_name in [
        "step5_foundation_inverse",
        "step6_dit_interpretability",
        "step6_view_compare_analysis",
        "step6_ood_aware_inverse",
        "step7_chem_physics_analysis",
    ]:
        if step_name not in parts:
            continue
        idx = parts.index(step_name)
        if idx + 1 >= len(parts):
            continue
        candidate = parts[idx + 1]
        if candidate.lower() in STEP_LEAF_DIR_NAMES:
            continue
        prop = _normalize_property_name(candidate)
        if prop:
            return prop
    return _property_from_suffixed_filename(path.name)


def _stable_hashed_copy_name(path: Path, *, root: Path, default_stem: str = "artifact") -> str:
    rel = _safe_rel(path, root).replace("\\", "/")
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    if not stem:
        stem = default_stem
    stem = stem[:90]
    suffix = path.suffix.lower() if path.suffix else ".bin"
    return f"{stem}__{digest}{suffix}"


def _supplementary_dst_name(path: Path, size_tag: str) -> Path:
    property_scope = _extract_property_scope_from_path(path) or "global"
    safe_scope = re.sub(r"[^A-Za-z0-9._-]+", "_", property_scope).strip("._") or "global"
    hashed = _stable_hashed_copy_name(path, root=REPO_ROOT, default_stem="supp")
    return Path(f"{size_tag}__{safe_scope}__{hashed}")


def _collapse_copied_artifacts(copied: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for item in copied:
        category = str(item.get("category", ""))
        source = str(item.get("source", ""))
        destination = str(item.get("destination", ""))
        key = (category, source)
        if key not in grouped:
            grouped[key] = {
                "category": category,
                "source": source,
                "destinations": [],
            }
        if destination and destination not in grouped[key]["destinations"]:
            grouped[key]["destinations"].append(destination)

    collapsed = list(grouped.values())
    collapsed.sort(key=lambda row: (str(row.get("category", "")), str(row.get("source", ""))))
    return collapsed


def _parse_method_dirs(value) -> list[str]:
    if value is None:
        return list(DEFAULT_METHOD_DIRS)
    if isinstance(value, str):
        tokens = [v.strip() for v in value.split(",") if v.strip()]
        return tokens if tokens else list(DEFAULT_METHOD_DIRS)
    if isinstance(value, (list, tuple, set)):
        tokens = [str(v).strip() for v in value if str(v).strip()]
        return tokens if tokens else list(DEFAULT_METHOD_DIRS)
    return list(DEFAULT_METHOD_DIRS)


def _aggregate_dirs_sorted(repo_results_root: Path) -> list[Path]:
    if not repo_results_root.exists() or not repo_results_root.is_dir():
        return []
    dirs = [p for p in repo_results_root.glob("aggregate*") if p.is_dir()]
    dirs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return dirs


def _load_aggregate_csv(repo_results_root: Path, stem: str) -> pd.DataFrame:
    # Deterministic policy: pick the newest non-empty aggregate run.
    # If all are empty, fall back to the newest readable one.
    newest_nonempty: Optional[pd.DataFrame] = None
    newest_any: Optional[pd.DataFrame] = None
    for agg_dir in _aggregate_dirs_sorted(repo_results_root):
        path = agg_dir / stem
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df is None:
            continue
        tagged = df.copy()
        tagged["source_path"] = str(path)
        tagged["aggregate_dir"] = agg_dir.name

        if newest_any is None:
            newest_any = tagged
        if not tagged.empty:
            newest_nonempty = tagged
            return newest_nonempty

    if newest_any is not None:
        return newest_any

    # Legacy fallback in case directory glob misses anything unusual.
    legacy_candidates = sorted(repo_results_root.glob(f"aggregate*/{stem}"), reverse=True)
    for path in legacy_candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        tagged = df.copy()
        tagged["source_path"] = str(path)
        tagged["aggregate_dir"] = path.parent.name
        return tagged
    return pd.DataFrame()


def _figure1_primary_panels(repo_results_root: Path) -> list[Path]:
    # Priority source explicitly requested: aggregate_step12 figure trilogy.
    ordered: list[Path] = []
    prioritized = repo_results_root / "aggregate_step12" / "figures"
    if prioritized.exists():
        for name in [
            "fig_01_step1_bpb_heatmap.png",
            "fig_02_step2_metrics_heatmap.png",
            "fig_03_step2_tradeoff_scatter.png",
        ]:
            p = prioritized / name
            if p.exists():
                ordered.append(p)
    if ordered:
        return ordered

    # Fallback: newest aggregate run with matching fig_01..03 names.
    for agg_dir in _aggregate_dirs_sorted(repo_results_root):
        fig_dir = agg_dir / "figures"
        if not fig_dir.exists():
            continue
        local: list[Path] = []
        for pattern in [
            "fig_01*.png",
            "fig_02*.png",
            "fig_03*.png",
        ]:
            for p in sorted(fig_dir.glob(pattern)):
                local.append(p)
                break
        if local:
            return local
    return []


def _load_mvf_csv(results_dir: Path, step_subdir: str, filename: str) -> pd.DataFrame:
    candidates = [
        results_dir / step_subdir / "metrics" / filename,
        results_dir / step_subdir / "files" / filename,
        results_dir / step_subdir / filename,
        results_dir / filename,
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return pd.read_csv(path)
        except Exception:
            continue
    return pd.DataFrame()


def _load_mvf_csv_multi(
    results_dirs: Sequence[Path],
    step_subdir: str,
    filename: str,
    *,
    include_property_scopes: bool = False,
    suffixed_filename_regex: Optional[str] = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    suffixed_pattern = re.compile(suffixed_filename_regex) if suffixed_filename_regex else None

    for res_dir in _unique_paths(list(results_dirs)):
        step_root = res_dir / step_subdir
        property_scopes = _iter_property_scoped_dirs(step_root) if include_property_scopes else []
        scope_dirs: list[Path] = property_scopes if property_scopes else [step_root]
        loaded_for_res_dir = False

        for scope_dir in scope_dirs:
            df = _load_step_scope_csv(scope_dir, filename)
            if df is None or df.empty:
                continue
            prop_hint = scope_dir.name if scope_dir != step_root else None
            frames.append(_tag_mvf_frame(df, res_dir, property_hint=prop_hint))
            loaded_for_res_dir = True

        if loaded_for_res_dir:
            continue

        if include_property_scopes and suffixed_pattern is not None and step_root.exists():
            seen_suffix_paths: set[str] = set()
            for search_dir in [step_root / "metrics", step_root / "files", step_root]:
                if not search_dir.exists() or not search_dir.is_dir():
                    continue
                for path in sorted(search_dir.glob("*.csv")):
                    key = str(path.resolve())
                    if key in seen_suffix_paths:
                        continue
                    seen_suffix_paths.add(key)
                    m = suffixed_pattern.match(path.name)
                    if not m:
                        continue
                    prop_hint = _normalize_property_name(m.group(1))
                    try:
                        df = pd.read_csv(path)
                    except Exception:
                        continue
                    if df.empty:
                        continue
                    frames.append(_tag_mvf_frame(df, res_dir, property_hint=prop_hint))
                    loaded_for_res_dir = True

        if loaded_for_res_dir:
            continue

        # Legacy fallback: non-property-scoped file locations.
        df = _load_mvf_csv(res_dir, step_subdir, filename)
        if df is None or df.empty:
            continue
        frames.append(_tag_mvf_frame(df, res_dir))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _repr_label(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "Unknown"
    lower = text.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "smiles": "SMILES",
        "smiles_bpe": "SMILES-BPE",
        "selfies": "SELFIES",
        "group_selfies": "Group-SELFIES",
        "graph": "Graph",
        "multiview_mean": "MVF Mean",
        "multi_view_mean": "MVF Mean",
        "multiviewmean": "MVF Mean",
        "mvf_mean": "MVF Mean",
    }
    if lower in mapping:
        return mapping[lower]
    if lower.startswith("bi_diffusion_"):
        return _repr_label(lower.replace("bi_diffusion_", ""))
    return text


def _first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _panel_mark(ax, label: str, font_size: int) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=font_size,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 1.5},
    )


def _save_plot_figure(fig, output_path: Path, dpi: int) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for text_obj in fig.findobj(match=lambda artist: hasattr(artist, "set_fontsize")):
        try:
            text_obj.set_fontsize(16)
        except Exception:
            continue
    for ax in fig.axes:
        try:
            ax.xaxis.label.set_fontsize(16)
            ax.yaxis.label.set_fontsize(16)
        except Exception:
            pass
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            try:
                tick.set_fontsize(16)
            except Exception:
                continue
        legend = ax.get_legend()
        if legend is not None:
            set_legend_location(legend, "best")
            for text in legend.get_texts():
                text.set_fontsize(16)
        if ax.get_xlabel():
            ax.xaxis.label.set_color(COLOR_TEXT)
        if ax.get_ylabel():
            ax.yaxis.label.set_color(COLOR_TEXT)
    try:
        suptitle = getattr(fig, "_suptitle", None)
        if suptitle is not None:
            suptitle.set_text("")
    except Exception:
        pass
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _split_view_pair(text: object) -> tuple[Optional[str], Optional[str]]:
    raw = str(text).strip()
    if not raw:
        return None, None
    for sep in ["->", "|", "__", ",", " to ", "/", "=>"]:
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep) if p.strip()]
            if len(parts) >= 2:
                return _repr_label(parts[0]), _repr_label(parts[1])
    return None, None


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _method_display_name(method_dir: str) -> str:
    return str(method_dir).replace("Bi_Diffusion_", "").replace("_", " ").strip() or str(method_dir)


def _representation_from_method_dir(method_dir: str) -> str:
    name = str(method_dir).lower()
    if "group_selfies" in name:
        return "Group SELFIES"
    if "smiles_bpe" in name:
        return "SMILES BPE"
    if "selfies" in name:
        return "SELFIES"
    if "graph" in name:
        return "Graph"
    return "SMILES"


def _infer_model_size_from_results_dir(results_dir: Path) -> str:
    token = results_dir.name.lower()
    for size in ("small", "medium", "large", "xl", "base"):
        if re.search(rf"(^|[_-]){size}($|[_-])", token):
            return size
    return "unknown"


def _safe_numeric(value) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    if np.isnan(x) or np.isinf(x):
        return float("nan")
    return float(x)


def _extract_step1_bpb(results_dir: Path) -> float:
    metrics_dir = results_dir / "step1_backbone" / "metrics"
    if not metrics_dir.exists():
        return float("nan")

    bpb_keys = {"bpb", "best_bpb", "best_val_bpb", "val_bpb", "bits_per_byte", "bits_per_token"}
    loss_keys = {"best_val_loss", "val_loss", "final_val_loss", "validation_loss"}

    bpb_vals: list[float] = []
    loss_vals: list[float] = []

    for csv_path in sorted(metrics_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        lookup = {str(c).lower().strip(): c for c in df.columns}
        for key in bpb_keys:
            col = lookup.get(key)
            if col is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size:
                bpb_vals.append(float(np.min(vals)))
        for key in loss_keys:
            col = lookup.get(key)
            if col is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size:
                loss_vals.append(float(np.min(vals)))

    for json_path in sorted(metrics_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for key in bpb_keys:
            if key in payload:
                val = _safe_numeric(payload.get(key))
                if np.isfinite(val):
                    bpb_vals.append(float(val))
        for key in loss_keys:
            if key in payload:
                val = _safe_numeric(payload.get(key))
                if np.isfinite(val):
                    loss_vals.append(float(val))
        val_losses = payload.get("val_losses")
        if isinstance(val_losses, list) and val_losses:
            vals = np.array([_safe_numeric(v) for v in val_losses], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                loss_vals.append(float(np.min(vals)))

    if bpb_vals:
        return float(np.nanmin(np.array(bpb_vals, dtype=float)))
    if loss_vals:
        return float(np.nanmin(np.array(loss_vals, dtype=float)) / math.log(2.0))
    return float("nan")


def _extract_step2_metrics(results_dir: Path) -> dict[str, float]:
    out = {
        "step2_validity": float("nan"),
        "step2_uniqueness": float("nan"),
        "step2_novelty": float("nan"),
        "step2_diversity": float("nan"),
        "step2_mean_sa": float("nan"),
    }
    metrics_path = results_dir / "step2_sampling" / "metrics" / "sampling_generative_metrics.csv"
    if not metrics_path.exists():
        return out
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return out
    if df.empty:
        return out

    aliases = {
        "step2_validity": ["validity"],
        "step2_uniqueness": ["uniqueness"],
        "step2_novelty": ["novelty"],
        "step2_diversity": ["avg_diversity", "diversity"],
        "step2_mean_sa": ["mean_sa", "avg_sa"],
    }
    for key, cols in aliases.items():
        for col in cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
                if vals.size:
                    out[key] = float(np.mean(vals))
                    break
    return out


def _collect_step12_scale_summary(method_dirs: Sequence[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for method_dir in (method_dirs or list(DEFAULT_METHOD_DIRS)):
        method_root = _resolve_repo_path(method_dir)
        for res_dir in _iter_method_results_dirs(method_root):
            model_size = _infer_model_size_from_results_dir(res_dir)
            step1_bpb = _extract_step1_bpb(res_dir)
            step2 = _extract_step2_metrics(res_dir)
            has_step2 = any(np.isfinite(_safe_numeric(step2.get(c))) for c in STEP2_METRIC_COLUMNS + ["step2_mean_sa"])
            if not np.isfinite(step1_bpb) and not has_step2:
                continue
            rows.append(
                {
                    "method_dir": method_dir,
                    "method": _method_display_name(method_dir),
                    "representation": _representation_from_method_dir(method_dir),
                    "model_size": model_size,
                    "results_dir": str(res_dir),
                    "step1_bpb": step1_bpb,
                    **step2,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "method_dir",
                "method",
                "representation",
                "model_size",
                "results_dir",
                "step1_bpb",
                "step2_validity",
                "step2_uniqueness",
                "step2_novelty",
                "step2_diversity",
                "step2_mean_sa",
                "step2_quality_score",
            ]
        )

    df = pd.DataFrame(rows)
    group_cols = ["method_dir", "method", "representation", "model_size"]
    agg_cols = [
        "step1_bpb",
        "step2_validity",
        "step2_uniqueness",
        "step2_novelty",
        "step2_diversity",
        "step2_mean_sa",
    ]
    summary = df.groupby(group_cols, as_index=False)[agg_cols].mean(numeric_only=True)

    sa = pd.to_numeric(summary["step2_mean_sa"], errors="coerce")
    sa_component = pd.Series(np.nan, index=summary.index, dtype=float)
    if sa.notna().any():
        sa_min = float(sa.min())
        sa_max = float(sa.max())
        if abs(sa_max - sa_min) > 1e-12:
            sa_component = (sa_max - sa) / (sa_max - sa_min)
        else:
            sa_component = pd.Series(np.where(sa.notna(), 1.0, np.nan), index=summary.index, dtype=float)
    summary["step2_sa_component"] = sa_component
    score_cols = STEP2_METRIC_COLUMNS + ["step2_sa_component"]
    summary["step2_quality_score"] = (
        summary[score_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=False)
    )

    summary["size_rank"] = summary["model_size"].apply(
        lambda x: MODEL_SIZE_ORDER.index(x) if x in MODEL_SIZE_ORDER else len(MODEL_SIZE_ORDER)
    )
    summary["method_rank"] = summary["method_dir"].apply(
        lambda x: DEFAULT_METHOD_DIRS.index(x) if x in DEFAULT_METHOD_DIRS else len(DEFAULT_METHOD_DIRS)
    )
    summary = summary.sort_values(["method_rank", "size_rank"]).reset_index(drop=True)
    summary = summary.drop(columns=["method_rank", "size_rank"])
    return summary


def _generate_step12_scale_panels(
    *,
    summary_df: pd.DataFrame,
    panel_dir: Path,
    font_size: int,
    dpi: int,
    max_panels_per_figure: int,
) -> list[Path]:
    if plt is None or summary_df.empty:
        return []

    panel_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    size_order = [s for s in MODEL_SIZE_ORDER if s in set(summary_df["model_size"].astype(str))]
    if not size_order:
        size_order = list(MODEL_SIZE_ORDER)

    # Panel 1: Step1 BPB vs model size for each method.
    fig1, ax1 = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    for method_dir in (DEFAULT_METHOD_DIRS + [m for m in sorted(summary_df["method_dir"].unique()) if m not in DEFAULT_METHOD_DIRS]):
        sub = summary_df[summary_df["method_dir"] == method_dir].copy()
        if sub.empty:
            continue
        sub["size_idx"] = sub["model_size"].apply(lambda s: size_order.index(s) if s in size_order else np.nan)
        sub = sub.dropna(subset=["size_idx", "step1_bpb"]).sort_values("size_idx")
        if sub.empty:
            continue
        ax1.plot(
            sub["size_idx"].to_numpy(dtype=float),
            sub["step1_bpb"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.2,
            markersize=7.5,
            color=METHOD_PLOT_COLORS.get(method_dir, "#334155"),
            label=_method_display_name(method_dir),
        )
    ax1.set_xticks(np.arange(len(size_order)))
    ax1.set_xticklabels([s.upper() for s in size_order])
    ax1.set_xlabel("Model Size")
    ax1.set_ylabel("Step1 BPB")
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="best", frameon=False)
    p1 = panel_dir / "figure_step12_scale_step1_bpb_vs_size.png"
    fig1.savefig(p1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    generated.append(p1)

    # Panel 2: Step2 quality vs model size for each method.
    fig2, ax2 = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    for method_dir in (DEFAULT_METHOD_DIRS + [m for m in sorted(summary_df["method_dir"].unique()) if m not in DEFAULT_METHOD_DIRS]):
        sub = summary_df[summary_df["method_dir"] == method_dir].copy()
        if sub.empty:
            continue
        sub["size_idx"] = sub["model_size"].apply(lambda s: size_order.index(s) if s in size_order else np.nan)
        sub = sub.dropna(subset=["size_idx", "step2_quality_score"]).sort_values("size_idx")
        if sub.empty:
            continue
        ax2.plot(
            sub["size_idx"].to_numpy(dtype=float),
            sub["step2_quality_score"].to_numpy(dtype=float),
            marker="s",
            linewidth=2.2,
            markersize=7.0,
            color=METHOD_PLOT_COLORS.get(method_dir, "#334155"),
            label=_method_display_name(method_dir),
        )
    ax2.set_xticks(np.arange(len(size_order)))
    ax2.set_xticklabels([s.upper() for s in size_order])
    ax2.set_xlabel("Model Size")
    ax2.set_ylabel("Step2 Composite Quality")
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="best", frameon=False)
    p2 = panel_dir / "figure_step12_scale_step2_quality_vs_size.png"
    fig2.savefig(p2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    generated.append(p2)

    # Panel 3: Step1-Step2 tradeoff frontier across methods and scales.
    fig3, ax3 = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    marker_by_size = {"small": "o", "medium": "s", "large": "^", "xl": "D", "base": "P", "unknown": "X"}
    for method_dir in (DEFAULT_METHOD_DIRS + [m for m in sorted(summary_df["method_dir"].unique()) if m not in DEFAULT_METHOD_DIRS]):
        sub = summary_df[summary_df["method_dir"] == method_dir].copy()
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            x = _safe_numeric(row.get("step1_bpb"))
            y = _safe_numeric(row.get("step2_quality_score"))
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            size = str(row.get("model_size", "unknown"))
            ax3.scatter(
                [x],
                [y],
                s=150,
                marker=marker_by_size.get(size, "o"),
                color=METHOD_PLOT_COLORS.get(method_dir, "#334155"),
                edgecolors=COLOR_TEXT,
                linewidths=0.7,
                alpha=0.92,
            )
            ax3.text(x, y, size.upper(), fontsize=max(10, font_size - 4), ha="left", va="bottom")

    ax3.set_xlabel("Step1 BPB (lower is better)")
    ax3.set_ylabel("Step2 Composite Quality (higher is better)")
    ax3.grid(True, linestyle="--", alpha=0.45)
    p3 = panel_dir / "figure_step12_scale_frontier.png"
    fig3.savefig(p3, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    generated.append(p3)

    # Optional panel 4: metric ribbons for validity/novelty/diversity by size.
    if max_panels_per_figure >= 4:
        metric_plot_cols = ["step2_validity", "step2_novelty", "step2_diversity"]
        fig4, axes = plt.subplots(
            1,
            len(metric_plot_cols),
            figsize=(5.6 * len(metric_plot_cols), 5.6),
            squeeze=False,
            constrained_layout=True,
        )
        for mi, metric_col in enumerate(metric_plot_cols):
            ax = axes[0, mi]
            for method_dir in (DEFAULT_METHOD_DIRS + [m for m in sorted(summary_df["method_dir"].unique()) if m not in DEFAULT_METHOD_DIRS]):
                sub = summary_df[summary_df["method_dir"] == method_dir].copy()
                if sub.empty:
                    continue
                sub["size_idx"] = sub["model_size"].apply(lambda s: size_order.index(s) if s in size_order else np.nan)
                sub = sub.dropna(subset=["size_idx", metric_col]).sort_values("size_idx")
                if sub.empty:
                    continue
                ax.plot(
                    sub["size_idx"].to_numpy(dtype=float),
                    sub[metric_col].to_numpy(dtype=float),
                    marker="o",
                    linewidth=1.8,
                    markersize=5.8,
                    color=METHOD_PLOT_COLORS.get(method_dir, "#334155"),
                )
            ax.set_xticks(np.arange(len(size_order)))
            ax.set_xticklabels([s.upper() for s in size_order], rotation=20, ha="right")
            ax.set_xlabel("Model Size")
            ax.set_ylabel(metric_col.replace("step2_", "").replace("_", " ").title())
            ax.grid(True, linestyle="--", alpha=0.45)
        p4 = panel_dir / "figure_step12_scale_step2_metric_trends.png"
        fig4.savefig(p4, dpi=dpi, bbox_inches="tight")
        plt.close(fig4)
        generated.append(p4)

    return generated


def _fig1_baseline_generation(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_generation: pd.DataFrame,
    df_step12: pd.DataFrame,
    primary_panels: Sequence[Path],
    fallback_panels: Sequence[Path],
    fallback_title: str,
    fallback_lines: Sequence[str],
    fallback_note: str,
) -> tuple[bool, str, str]:
    caption = MANUSCRIPT_CAPTIONS[0]

    # Highest priority for Figure 1 storyline: aggregate_step12 fig_01..03 panels.
    primaries = [p for p in primary_panels if Path(p).exists()]
    if primaries:
        ok = _compose_multi_panel_figure(
            panel_paths=primaries,
            output_path=output_path,
            font_size=font_size,
            dpi=dpi,
        )
        return ok, caption, "panels"

    if plt is None:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    use_df = pd.DataFrame()
    if not df_generation.empty:
        use_df = df_generation.copy()
        use_df = _coerce_numeric(use_df, ["validity", "uniqueness", "novelty", "avg_diversity"])
        if "avg_diversity" not in use_df.columns and "diversity" in use_df.columns:
            use_df["avg_diversity"] = pd.to_numeric(use_df["diversity"], errors="coerce")
        rep_col = _first_existing_col(use_df, ["representation", "method", "method_dir"])
        size_col = _first_existing_col(use_df, ["model_size"])
        if rep_col is None:
            rep_col = _first_existing_col(use_df, ["method"])
        if rep_col is not None:
            use_df["representation_label"] = use_df[rep_col].astype(str).map(_repr_label)
        else:
            use_df["representation_label"] = "Unknown"
        if size_col is None:
            use_df["model_size"] = "unknown"
        else:
            use_df["model_size"] = use_df[size_col].astype(str).str.lower().str.strip()
    elif not df_step12.empty:
        tmp = df_step12.copy()
        rename_map = {
            "step2_validity": "validity",
            "step2_uniqueness": "uniqueness",
            "step2_novelty": "novelty",
            "step2_diversity": "avg_diversity",
        }
        tmp = tmp.rename(columns=rename_map)
        tmp = _coerce_numeric(tmp, ["validity", "uniqueness", "novelty", "avg_diversity"])
        rep_col = _first_existing_col(tmp, ["representation", "method", "method_dir"])
        if rep_col is not None:
            tmp["representation_label"] = tmp[rep_col].astype(str).map(_repr_label)
        else:
            tmp["representation_label"] = "Unknown"
        if "model_size" not in tmp.columns:
            tmp["model_size"] = "unknown"
        use_df = tmp

    metric_cols = ["validity", "uniqueness", "novelty", "avg_diversity"]
    if use_df.empty or not any(col in use_df.columns for col in metric_cols):
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    agg = (
        use_df.groupby(["representation_label", "model_size"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )
    agg = agg.dropna(subset=["validity", "uniqueness"], how="all")
    if agg.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    best_idx = agg.groupby("representation_label")["validity"].idxmax()
    best_idx = pd.to_numeric(best_idx, errors="coerce").dropna().astype(int)
    if best_idx.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode
    best = agg.loc[best_idx].copy()
    best = best.dropna(subset=["validity"], how="any")
    if best.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode
    best = best.sort_values("validity", ascending=False).reset_index(drop=True)
    scatter_df = agg.copy()

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.2), squeeze=False)
    ax0, ax1, ax2 = axes[0]

    x = np.arange(len(best))
    ax0.bar(x, best["validity"].to_numpy(dtype=float), color=NATURE_PALETTE[3], edgecolor=COLOR_TEXT, linewidth=0.8)
    ax0.set_ylim(0.0, 1.02)
    ax0.set_ylabel("Validity")
    ax0.set_xticks(x)
    ax0.set_xticklabels(best["representation_label"].tolist(), rotation=25, ha="right")
    ax0.grid(True, axis="y", linestyle="--", alpha=0.4)
    _wrap_ticklabels(ax0, axis="x", width=14, rotation=32)
    _panel_mark(ax0, "(a)", font_size)

    ax1.bar(x, best["uniqueness"].to_numpy(dtype=float), color=NATURE_PALETTE[0], edgecolor=COLOR_TEXT, linewidth=0.8)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_ylabel("Uniqueness")
    ax1.set_xticks(x)
    ax1.set_xticklabels(best["representation_label"].tolist(), rotation=25, ha="right")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    _wrap_ticklabels(ax1, axis="x", width=14, rotation=32)
    _panel_mark(ax1, "(b)", font_size)

    novelty_vals = pd.to_numeric(scatter_df.get("novelty"), errors="coerce")
    scatter = ax2.scatter(
        pd.to_numeric(scatter_df.get("validity"), errors="coerce"),
        pd.to_numeric(scatter_df.get("uniqueness"), errors="coerce"),
        c=novelty_vals,
        cmap=_nature_sequential_cmap(),
        s=140,
        edgecolors=COLOR_TEXT,
        linewidths=0.6,
        alpha=0.9,
    )
    ax2.set_xlim(0.0, 1.02)
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xlabel("Validity")
    ax2.set_ylabel("Uniqueness")
    ax2.grid(True, linestyle="--", alpha=0.4)
    cbar = fig.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Novelty")
    _panel_mark(ax2, "(c)", font_size)

    return _save_plot_figure(fig, output_path, dpi), caption, "data"


def _fig2_mvf_alignment(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_alignment: pd.DataFrame,
    fallback_panels: Sequence[Path],
    fallback_title: str,
    fallback_lines: Sequence[str],
    fallback_note: str,
) -> tuple[bool, str, str]:
    caption = MANUSCRIPT_CAPTIONS[1]
    if plt is None or df_alignment.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    df = df_alignment.copy()
    r1_col = _first_existing_col(df, ["recall_at_1", "r1"])
    r10_col = _first_existing_col(df, ["recall_at_10", "r10"])
    if r1_col is None or r10_col is None:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    df = _coerce_numeric(df, [r1_col, r10_col])
    if {"source_view", "target_view"}.issubset(df.columns):
        df["source_label"] = df["source_view"].astype(str).map(_repr_label)
        df["target_label"] = df["target_view"].astype(str).map(_repr_label)
    elif "view_pair" in df.columns:
        pairs = df["view_pair"].apply(_split_view_pair)
        df["source_label"] = [p[0] for p in pairs]
        df["target_label"] = [p[1] for p in pairs]
    else:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    df = df.dropna(subset=["source_label", "target_label"])
    if df.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    views = sorted(set(df["source_label"]).union(set(df["target_label"])))
    p1 = df.pivot_table(index="source_label", columns="target_label", values=r1_col, aggfunc="mean").reindex(index=views, columns=views)
    p10 = df.pivot_table(index="source_label", columns="target_label", values=r10_col, aggfunc="mean").reindex(index=views, columns=views)
    if p1.isna().all().all() and p10.isna().all().all():
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), squeeze=False)
    ax0, ax1 = axes[0]
    mats = [p1.to_numpy(dtype=float), p10.to_numpy(dtype=float)]
    cmap = _nature_sequential_cmap()
    cmap.set_bad(color="#D1D5DB")
    for ax, mat, panel_tag in [(ax0, mats[0], "(a)"), (ax1, mats[1], "(b)")]:
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(views)))
        ax.set_yticks(np.arange(len(views)))
        ax.set_xticklabels(views, rotation=35, ha="right")
        ax.set_yticklabels(views)
        ax.set_xlabel("Target View")
        ax.set_ylabel("Source View")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=max(10, font_size - 5), color=COLOR_TEXT)
        _panel_mark(ax, panel_tag, font_size)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Recall")
        _wrap_ticklabels(ax, axis="x", width=12, rotation=35)
        _wrap_ticklabels(ax, axis="y", width=12, rotation=0)

    return _save_plot_figure(fig, output_path, dpi), caption, "data"


def _fig3_property_prediction(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_property_base: pd.DataFrame,
    df_property_mvf: pd.DataFrame,
    fallback_panels: Sequence[Path],
    fallback_title: str,
    fallback_lines: Sequence[str],
    fallback_note: str,
) -> tuple[bool, str, str]:
    caption = MANUSCRIPT_CAPTIONS[3]
    if plt is None:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    frames: list[pd.DataFrame] = []
    for src_name, src_df in [("baseline", df_property_base), ("mvf", df_property_mvf)]:
        if src_df is None or src_df.empty:
            continue
        df = src_df.copy()
        if "split" in df.columns:
            split_l = df["split"].astype(str).str.lower().str.strip()
            if (split_l == "test").any():
                df = df.loc[split_l == "test"].copy()
        r2_col = _first_existing_col(df, ["r2", "R2", "test_r2"])
        prop_col = _first_existing_col(df, ["property", "Property"])
        rep_col = _first_existing_col(df, ["representation", "view", "method"])
        size_col = _first_existing_col(df, ["model_size"])
        if r2_col is None or prop_col is None or rep_col is None:
            continue
        df["r2"] = pd.to_numeric(df[r2_col], errors="coerce")
        df["property"] = df[prop_col].astype(str).str.strip()
        df["representation_label"] = df[rep_col].astype(str).map(_repr_label)
        if size_col is not None:
            df["model_size"] = df[size_col].apply(_normalize_model_size)
        else:
            df["model_size"] = "unknown"
        df["source_name"] = src_name
        frames.append(df[["property", "representation_label", "model_size", "r2", "source_name"]])

    if not frames:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["property", "representation_label", "model_size", "r2"])
    if data.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    size_order = [s for s in MODEL_SIZE_ORDER if s in set(data["model_size"].astype(str))]
    for s in sorted(set(data["model_size"].astype(str))):
        if s not in size_order:
            size_order.append(s)
    if not size_order:
        size_order = ["unknown"]

    prop_order = _ordered_properties(data["property"].tolist())

    agg = data.groupby(["property", "model_size", "representation_label", "source_name"], as_index=False)["r2"].mean(numeric_only=True)
    if agg.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    fusion_tokens = ("multiview", "multi view", "fusion", "mean")
    comp_rows: list[dict[str, object]] = []
    for prop in prop_order:
        for size in size_order:
            block = agg[(agg["property"] == prop) & (agg["model_size"] == size)]
            if block.empty:
                continue
            mvf_block = block.loc[block["source_name"] == "mvf"].copy()
            if mvf_block.empty:
                continue
            rep_lower = mvf_block["representation_label"].astype(str).str.lower()
            fusion_mask = rep_lower.apply(lambda text: any(tok in text for tok in fusion_tokens))
            fusion_vals = mvf_block.loc[fusion_mask, "r2"].dropna()
            single_vals = mvf_block.loc[~fusion_mask, "r2"].dropna()
            baseline_vals = block.loc[block["source_name"] == "baseline", "r2"].dropna()
            if fusion_vals.empty and mvf_block["r2"].dropna().empty:
                continue
            fusion_best = float(fusion_vals.max()) if not fusion_vals.empty else float(mvf_block["r2"].max())
            if not single_vals.empty:
                reference_best = float(single_vals.max())
                reference_source = "single_view"
            elif not baseline_vals.empty:
                reference_best = float(baseline_vals.max())
                reference_source = "baseline"
            else:
                reference_best = np.nan
                reference_source = "missing"
            gain = fusion_best - reference_best if np.isfinite(fusion_best) and np.isfinite(reference_best) else np.nan
            comp_rows.append(
                {
                    "property": prop,
                    "model_size": size,
                    "reference_r2": reference_best,
                    "fusion_r2": fusion_best,
                    "fusion_gain": gain,
                    "reference_source": reference_source,
                }
            )

    comp = pd.DataFrame(comp_rows)
    if comp.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2), squeeze=False)
    ax0, ax1 = axes[0]

    if len(size_order) <= 1:
        summary = (
            comp.groupby("property", as_index=False)[["reference_r2", "fusion_r2", "fusion_gain"]]
            .mean(numeric_only=True)
        )
        summary = summary.set_index("property").reindex(prop_order).dropna(how="all").reset_index()
        x = np.arange(len(summary))
        width = 0.36
        ax0.bar(
            x - width / 2.0,
            pd.to_numeric(summary["reference_r2"], errors="coerce").to_numpy(dtype=float),
            width=width,
            color=COLOR_MUTED,
            edgecolor=COLOR_TEXT,
            linewidth=0.7,
            label="Best single-view reference",
        )
        ax0.bar(
            x + width / 2.0,
            pd.to_numeric(summary["fusion_r2"], errors="coerce").to_numpy(dtype=float),
            width=width,
            color=NATURE_PALETTE[0],
            edgecolor=COLOR_TEXT,
            linewidth=0.7,
            label="Fused multi-view",
        )
        ax0.set_xticks(x)
        ax0.set_xticklabels(summary["property"].tolist(), rotation=35, ha="right")
        ax0.set_xlabel("Property")
        ax0.set_ylabel("Test R^2")
        ax0.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax0.legend(loc="best", frameon=False, fontsize=max(8, font_size - 4))

        gains = pd.to_numeric(summary["fusion_gain"], errors="coerce").to_numpy(dtype=float)
        gain_colors = [
            NATURE_PALETTE[0] if np.isfinite(val) and val >= 0.0 else NATURE_PALETTE[3]
            for val in gains
        ]
        ax1.bar(
            x,
            gains,
            color=gain_colors,
            edgecolor=COLOR_TEXT,
            linewidth=0.7,
        )
        ax1.axhline(0.0, color=COLOR_TEXT, linewidth=1.0, alpha=0.75)
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary["property"].tolist(), rotation=35, ha="right")
        ax1.set_xlabel("Property")
        ax1.set_ylabel("Fusion Gain in R^2")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    else:
        size_to_x = {s: i for i, s in enumerate(size_order)}
        for idx, prop in enumerate(prop_order):
            sub = comp[comp["property"] == prop].copy()
            if sub.empty:
                continue
            sub["x"] = sub["model_size"].map(size_to_x)
            sub = sub.dropna(subset=["x"]).sort_values("x")
            if sub.empty:
                continue
            color = NATURE_PALETTE[idx % len(NATURE_PALETTE)]
            x_vals = sub["x"].to_numpy(dtype=float)
            ax0.plot(
                x_vals,
                sub["reference_r2"].to_numpy(dtype=float),
                linestyle="--",
                marker="o",
                linewidth=2.0,
                markersize=6.5,
                color=color,
                alpha=0.65,
                label=f"{prop} best single",
            )
            ax0.plot(
                x_vals,
                sub["fusion_r2"].to_numpy(dtype=float),
                linestyle="-",
                marker="s",
                linewidth=2.2,
                markersize=6.5,
                color=color,
                alpha=0.95,
                label=f"{prop} fusion",
            )

        ax0.set_xticks(np.arange(len(size_order)))
        ax0.set_xticklabels([s.upper() for s in size_order])
        ax0.set_xlabel("Model Size")
        ax0.set_ylabel("Test R^2")
        ax0.grid(True, axis="y", linestyle="--", alpha=0.4)
        handles0, labels0 = ax0.get_legend_handles_labels()
        if handles0:
            max_props = 6
            max_entries = max_props * 2
            ax0.legend(
                handles0[:max_entries],
                labels0[:max_entries],
                loc="best",
                ncol=2,
                frameon=False,
                fontsize=max(8, font_size - 4),
            )

        gain_by_size = comp.groupby("model_size", as_index=False)["fusion_gain"].mean(numeric_only=True)
        gain_by_size["x"] = gain_by_size["model_size"].map(size_to_x)
        gain_by_size = gain_by_size.dropna(subset=["x"]).sort_values("x")
        ax1.bar(
            gain_by_size["x"].to_numpy(dtype=float),
            gain_by_size["fusion_gain"].to_numpy(dtype=float),
            color=NATURE_PALETTE[2],
            edgecolor=COLOR_TEXT,
            linewidth=0.7,
        )
        for idx, prop in enumerate(prop_order):
            sub = comp[comp["property"] == prop].copy()
            if sub.empty:
                continue
            sub["x"] = sub["model_size"].map(size_to_x)
            sub = sub.dropna(subset=["x", "fusion_gain"])
            if sub.empty:
                continue
            ax1.plot(
                sub["x"].to_numpy(dtype=float),
                sub["fusion_gain"].to_numpy(dtype=float),
                linestyle="",
                marker="o",
                markersize=4.8,
                color=NATURE_PALETTE[idx % len(NATURE_PALETTE)],
                alpha=0.8,
            )
        ax1.axhline(0.0, color=COLOR_TEXT, linewidth=1.0, alpha=0.75)
        ax1.set_xticks(np.arange(len(size_order)))
        ax1.set_xticklabels([s.upper() for s in size_order])
        ax1.set_xlabel("Model Size")
        ax1.set_ylabel("Fusion Gain in R^2")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    _panel_mark(ax0, "(a)", font_size)
    _panel_mark(ax1, "(b)", font_size)

    return _save_plot_figure(fig, output_path, dpi), caption, "data"


def _fig4_embedding_research(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_embedding_mvf: pd.DataFrame,
    fallback_panels: Sequence[Path],
    fallback_title: str,
    fallback_lines: Sequence[str],
    fallback_note: str,
) -> tuple[bool, str, str]:
    caption = MANUSCRIPT_CAPTIONS[2]
    if plt is None:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    if df_embedding_mvf is None or df_embedding_mvf.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    data = df_embedding_mvf.copy()
    view_col = _first_existing_col(data, ["view", "proposal_view", "representation"])
    size_col = _first_existing_col(data, ["model_size"])
    recall_col = _first_existing_col(data, ["mean_recall_at_10"])
    smooth_col = _first_existing_col(data, ["property_smoothness_mean"])
    if view_col is None or size_col is None or (recall_col is None and smooth_col is None):
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    data["proposal_view"] = data[view_col].astype(str).map(normalize_view_name)
    data = data[data["proposal_view"] != "all"].copy()
    data["model_size"] = data[size_col].apply(_normalize_model_size)
    data["mean_recall_at_10"] = pd.to_numeric(data[recall_col], errors="coerce") if recall_col else np.nan
    data["property_smoothness_mean"] = pd.to_numeric(data[smooth_col], errors="coerce") if smooth_col else np.nan
    data = (
        data.groupby(["proposal_view", "model_size"], as_index=False)[["mean_recall_at_10", "property_smoothness_mean"]]
        .mean(numeric_only=True)
    )
    data = data.dropna(subset=["mean_recall_at_10", "property_smoothness_mean"], how="all")
    if data.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    size_order = [s for s in MODEL_SIZE_ORDER if s in set(data["model_size"].astype(str))]
    for s in sorted(set(data["model_size"].astype(str))):
        if s not in size_order:
            size_order.append(s)
    if not size_order:
        size_order = ["unknown"]
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.2), squeeze=False)
    ax0, ax1 = axes[0]
    view_order = ordered_views(data["proposal_view"].tolist())
    if len(size_order) <= 1:
        recall_df = (
            data.groupby("proposal_view", as_index=False)["mean_recall_at_10"]
            .mean(numeric_only=True)
            .set_index("proposal_view")
            .reindex(view_order)
            .reset_index()
        )
        smooth_df = (
            data.groupby("proposal_view", as_index=False)["property_smoothness_mean"]
            .mean(numeric_only=True)
            .set_index("proposal_view")
            .reindex(view_order)
            .reset_index()
        )
        x = np.arange(len(view_order))
        recall_vals = pd.to_numeric(recall_df["mean_recall_at_10"], errors="coerce").to_numpy(dtype=float)
        smooth_vals = pd.to_numeric(smooth_df["property_smoothness_mean"], errors="coerce").to_numpy(dtype=float)
        bar_colors = [view_color(view) for view in view_order]
        if np.isfinite(recall_vals).any():
            ax0.bar(x, recall_vals, color=bar_colors, edgecolor=COLOR_TEXT, linewidth=0.7)
            ax0.set_xticks(x)
            ax0.set_xticklabels([view_label(view) for view in view_order], rotation=30, ha="right")
            ax0.set_xlabel("View")
            ax0.set_ylabel("Mean Recall@10")
            ax0.set_ylim(0.0, 1.0)
            ax0.grid(True, axis="y", linestyle="--", alpha=0.4)
        else:
            ax0.text(0.5, 0.5, "Recall@10 unavailable", ha="center", va="center", transform=ax0.transAxes)
            ax0.set_xticks([])
            ax0.set_yticks([])

        if np.isfinite(smooth_vals).any():
            ax1.bar(x, smooth_vals, color=bar_colors, edgecolor=COLOR_TEXT, linewidth=0.7)
            ax1.set_xticks(x)
            ax1.set_xticklabels([view_label(view) for view in view_order], rotation=30, ha="right")
            ax1.set_xlabel("View")
            ax1.set_ylabel("Property Smoothness")
            ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
        else:
            ax1.text(0.5, 0.5, "Property smoothness unavailable", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_xticks([])
            ax1.set_yticks([])
    else:
        size_to_x = {s: i for i, s in enumerate(size_order)}
        for proposal_view in view_order:
            sub = data[data["proposal_view"] == proposal_view].copy()
            if sub.empty:
                continue
            sub["x"] = sub["model_size"].map(size_to_x)
            color = view_color(proposal_view)
            label = view_label(proposal_view)

            sub_a = sub.dropna(subset=["x", "mean_recall_at_10"]).sort_values("x")
            if not sub_a.empty:
                ax0.plot(
                    sub_a["x"].to_numpy(dtype=float),
                    sub_a["mean_recall_at_10"].to_numpy(dtype=float),
                    linestyle="-",
                    marker="o",
                    linewidth=2.2,
                    markersize=7.2,
                    color=color,
                    label=label,
                )

            sub_b = sub.dropna(subset=["x", "property_smoothness_mean"]).sort_values("x")
            if not sub_b.empty:
                ax1.plot(
                    sub_b["x"].to_numpy(dtype=float),
                    sub_b["property_smoothness_mean"].to_numpy(dtype=float),
                    linestyle="-",
                    marker="o",
                    linewidth=2.2,
                    markersize=7.2,
                    color=color,
                    label=label,
                )

        ax0.set_xticks(np.arange(len(size_order)))
        ax0.set_xticklabels([s.upper() for s in size_order])
        ax0.set_xlabel("Model Size")
        ax0.set_ylabel("Mean Recall@10")
        ax0.set_ylim(0.0, 1.0)
        ax0.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax0.legend(loc="best", frameon=False)

        ax1.set_xticks(np.arange(len(size_order)))
        ax1.set_xticklabels([s.upper() for s in size_order])
        ax1.set_xlabel("Model Size")
        ax1.set_ylabel("Property Smoothness")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax1.legend(loc="best", frameon=False)

    _panel_mark(ax0, "(a)", font_size)
    _panel_mark(ax1, "(b)", font_size)

    return _save_plot_figure(fig, output_path, dpi), caption, "data"


def _fig5_inverse_design(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_inverse_base: pd.DataFrame,
    df_inverse_mvf: pd.DataFrame,
    df_inverse_topk: pd.DataFrame,
    fallback_panels: Sequence[Path],
    fallback_title: str,
    fallback_lines: Sequence[str],
    fallback_note: str,
) -> tuple[bool, str, str]:
    caption = MANUSCRIPT_CAPTIONS[4]
    if plt is None:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    def _prepare_view_frame(src_df: pd.DataFrame, value_col_candidates: Sequence[str], label: str) -> Optional[pd.DataFrame]:
        if src_df is None or src_df.empty:
            return None
        df = src_df.copy()
        prop_col = _first_existing_col(df, ["property", "Property"])
        size_col = _first_existing_col(df, ["model_size"])
        view_col = _first_existing_col(df, ["proposal_view", "representation", "Representation"])
        value_col = _first_existing_col(df, list(value_col_candidates))
        if prop_col is None or size_col is None or view_col is None or value_col is None:
            print(f"[F8] Warning: Figure 5 missing columns for {label}.")
            return None
        df = df.loc[df[prop_col].notna() & df[size_col].notna() & df[view_col].notna()].copy()
        if df.empty:
            return None
        df["property"] = df[prop_col].astype(str).str.strip()
        df = df[df["property"].astype(str).str.lower() != "nan"].copy()
        if df.empty:
            return None
        df["model_size"] = df[size_col].apply(_normalize_model_size)
        df["proposal_view"] = df[view_col].astype(str).map(normalize_view_name)
        df = df[(df["proposal_view"] != "") & (df["proposal_view"] != "all")].copy()
        if df.empty:
            return None
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["value"])
        if df.empty:
            return None
        return (
            df.groupby(["property", "proposal_view", "model_size"], as_index=False)["value"]
            .mean(numeric_only=True)
            .reset_index(drop=True)
        )

    f5 = _prepare_view_frame(df_inverse_mvf, ["fair_success_rate"], "F5")
    f5_topk = _prepare_view_frame(df_inverse_topk, ["top_k_fair_hit_rate"], "F5 top-k")

    if f5 is None or f5.empty or f5_topk is None or f5_topk.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    common_props = sorted(set(f5["property"].dropna().astype(str)) & set(f5_topk["property"].dropna().astype(str)))
    if not common_props:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    common_views = ordered_views(
        sorted(set(f5["proposal_view"].dropna().astype(str)) & set(f5_topk["proposal_view"].dropna().astype(str)))
    )
    if not common_views:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    if len(common_props) > 1:
        prop_order = _ordered_properties(common_props)
        fair_best = (
            f5.groupby(["property", "proposal_view"], as_index=False)["value"]
            .max(numeric_only=True)
            .pivot_table(index="property", columns="proposal_view", values="value", aggfunc="mean")
            .reindex(index=prop_order, columns=common_views)
        )
        topk_best = (
            f5_topk.groupby(["property", "proposal_view"], as_index=False)["value"]
            .max(numeric_only=True)
            .pivot_table(index="property", columns="proposal_view", values="value", aggfunc="mean")
            .reindex(index=prop_order, columns=common_views)
        )
        mats = [
            fair_best.to_numpy(dtype=float),
            topk_best.to_numpy(dtype=float),
        ]
        finite_values = np.concatenate([mat[np.isfinite(mat)] for mat in mats if np.isfinite(mat).any()]) if any(
            np.isfinite(mat).any() for mat in mats
        ) else np.asarray([], dtype=float)
        if finite_values.size == 0:
            ok, render_mode = _render_panels_or_placeholder(
                output_path=output_path,
                fallback_panels=fallback_panels,
                font_size=font_size,
                dpi=dpi,
                empty_title=fallback_title,
                empty_lines=fallback_lines,
            )
            return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

        shared_vmax = min(1.0, max(0.05, float(np.nanmax(finite_values))))
        cmap = _nature_sequential_cmap()
        cmap.set_bad(color="#D1D5DB")
        fig, axes = plt.subplots(1, 2, figsize=(16.2, 7.0), squeeze=False)
        ax0, ax1 = axes[0]
        for ax, mat, panel_tag, cbar_label in [
            (ax0, mats[0], "(a)", "Fair hit rate"),
            (ax1, mats[1], "(b)", "Top-k fair hit rate"),
        ]:
            im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0.0, vmax=shared_vmax, aspect="auto")
            ax.set_xticks(np.arange(len(common_views)))
            ax.set_xticklabels([view_label(view) for view in common_views], rotation=30, ha="right")
            ax.set_yticks(np.arange(len(prop_order)))
            ax.set_yticklabels(prop_order)
            ax.set_xlabel("Proposal View")
            ax.set_ylabel("Property")
            for row_idx in range(mat.shape[0]):
                for col_idx in range(mat.shape[1]):
                    val = mat[row_idx, col_idx]
                    if not np.isfinite(val):
                        continue
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=max(9, font_size - 6),
                        color="white" if val >= 0.55 * shared_vmax else COLOR_TEXT,
                    )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label)
            _panel_mark(ax, panel_tag, font_size)
        return _save_plot_figure(fig, output_path, dpi), caption, "data"

    target_prop = _ordered_properties(common_props)[0]
    f5 = f5[f5["property"] == target_prop].copy()
    f5_topk = f5_topk[f5_topk["property"] == target_prop].copy()
    if f5.empty or f5_topk.empty:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    size_order = [s for s in MODEL_SIZE_ORDER if s in set(pd.concat([f5["model_size"], f5_topk["model_size"]]).astype(str))]
    for s in sorted(set(pd.concat([f5["model_size"], f5_topk["model_size"]]).astype(str))):
        if s not in size_order:
            size_order.append(s)
    if not size_order:
        size_order = ["unknown"]

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.3), squeeze=False)
    ax0, ax1 = axes[0]
    all_views = ordered_views(pd.concat([f5["proposal_view"], f5_topk["proposal_view"]], ignore_index=True).tolist())

    if len(size_order) <= 1:
        fair_summary = (
            f5.groupby("proposal_view", as_index=False)["value"]
            .mean(numeric_only=True)
            .set_index("proposal_view")
            .reindex(all_views)
            .reset_index()
        )
        topk_summary = (
            f5_topk.groupby("proposal_view", as_index=False)["value"]
            .mean(numeric_only=True)
            .set_index("proposal_view")
            .reindex(all_views)
            .reset_index()
        )
        x = np.arange(len(all_views))
        colors = [view_color(view) for view in all_views]
        ax0.bar(
            x,
            pd.to_numeric(fair_summary["value"], errors="coerce").to_numpy(dtype=float),
            color=colors,
            edgecolor=COLOR_TEXT,
            linewidth=0.7,
        )
        ax0.set_xticks(x)
        ax0.set_xticklabels([view_label(view) for view in all_views], rotation=30, ha="right")
        ax0.set_xlabel("Proposal View")
        ax0.set_ylabel("F5 Fair Hit Rate")
        ax0.set_ylim(0.0, 1.0)
        ax0.grid(True, axis="y", linestyle="--", alpha=0.4)

        ax1.bar(
            x,
            pd.to_numeric(topk_summary["value"], errors="coerce").to_numpy(dtype=float),
            color=colors,
            edgecolor=COLOR_TEXT,
            linewidth=0.7,
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels([view_label(view) for view in all_views], rotation=30, ha="right")
        ax1.set_xlabel("Proposal View")
        ax1.set_ylabel("F5 Top-k Fair Hit Rate")
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    else:
        size_to_x = {s: i for i, s in enumerate(size_order)}
        for proposal_view in all_views:
            sub = f5[f5["proposal_view"] == proposal_view].copy()
            if sub.empty:
                continue
            sub["x"] = sub["model_size"].map(size_to_x)
            sub = sub.dropna(subset=["x"]).sort_values("x")
            if sub.empty:
                continue
            color = view_color(proposal_view)
            ax0.plot(
                sub["x"].to_numpy(dtype=float),
                sub["value"].to_numpy(dtype=float),
                linestyle="-",
                marker="o",
                linewidth=2.0,
                markersize=6.3,
                color=color,
                alpha=0.95,
                label=view_label(proposal_view),
            )
        ax0.set_xticks(np.arange(len(size_order)))
        ax0.set_xticklabels([s.upper() for s in size_order])
        ax0.set_xlabel("Model Size")
        ax0.set_ylabel("F5 Fair Hit Rate")
        ax0.set_ylim(0.0, 1.0)
        ax0.grid(True, axis="y", linestyle="--", alpha=0.4)
        handles0, labels0 = ax0.get_legend_handles_labels()
        if handles0:
            ax0.legend(handles0, labels0, loc="best", ncol=2, frameon=False, fontsize=max(8, font_size - 4))

        for proposal_view in all_views:
            sub = f5_topk[f5_topk["proposal_view"] == proposal_view].copy()
            if sub.empty:
                continue
            sub["x"] = sub["model_size"].map(size_to_x)
            sub = sub.dropna(subset=["x"])
            if sub.empty:
                continue
            color = view_color(proposal_view)
            ax1.plot(
                sub["x"].to_numpy(dtype=float),
                sub["value"].to_numpy(dtype=float),
                linestyle="-",
                marker="s",
                linewidth=2.2,
                markersize=6.3,
                color=color,
                alpha=0.95,
                label=view_label(proposal_view),
            )
        ax1.set_xticks(np.arange(len(size_order)))
        ax1.set_xticklabels([s.upper() for s in size_order])
        ax1.set_xlabel("Model Size")
        ax1.set_ylabel("F5 Top-k Fair Hit Rate")
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
        handles1, labels1 = ax1.get_legend_handles_labels()
        if handles1:
            ax1.legend(handles1, labels1, loc="best", ncol=2, frameon=False, fontsize=max(8, font_size - 4))

    _panel_mark(ax0, "(a)", font_size)
    _panel_mark(ax1, "(b)", font_size)

    return _save_plot_figure(fig, output_path, dpi), caption, "data"


def _fig6_chem_physics(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_descriptor: pd.DataFrame,
    df_motif: pd.DataFrame,
    df_physics: pd.DataFrame,
    df_nearest: pd.DataFrame,
    fallback_panels: Sequence[Path],
    fallback_title: str,
    fallback_lines: Sequence[str],
    fallback_note: str,
) -> tuple[bool, str, str]:
    caption = MANUSCRIPT_CAPTIONS[5]
    if plt is None:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    desc = df_descriptor.copy() if df_descriptor is not None else pd.DataFrame()
    motif = df_motif.copy() if df_motif is not None else pd.DataFrame()
    physics = df_physics.copy() if df_physics is not None else pd.DataFrame()
    nearest = df_nearest.copy() if df_nearest is not None else pd.DataFrame()

    prop_col = _first_existing_col(desc, ["property", "Property"])
    desc_col = _first_existing_col(desc, ["descriptor", "feature", "name"])
    desc_val_col = _first_existing_col(desc, ["delta_topk_vs_ref", "mean_shift", "delta", "shift"])
    if desc_val_col is None and {"topk_mean", "ref_mean"}.issubset(desc.columns):
        desc["delta_topk_vs_ref"] = pd.to_numeric(desc["topk_mean"], errors="coerce") - pd.to_numeric(desc["ref_mean"], errors="coerce")
        desc_val_col = "delta_topk_vs_ref"

    motif_prop_col = _first_existing_col(motif, ["property", "Property"])
    motif_col = _first_existing_col(motif, ["motif", "feature"])
    motif_val_col = _first_existing_col(
        motif,
        ["enrichment_ratio_topk_vs_ref", "enrichment_ratio", "log2_enrichment_topk_vs_ref", "delta_freq_topk_vs_ref"],
    )
    physics_prop_col = _first_existing_col(physics, ["property", "Property"])
    physics_sign_col = _first_existing_col(physics, ["sign_match"])
    nn_prop_col = _first_existing_col(nearest, ["property", "Property"])
    nn_val_col = _first_existing_col(nearest, ["nearest_tanimoto", "tanimoto", "similarity"])

    has_desc = prop_col is not None and desc_col is not None and desc_val_col is not None and not desc.empty
    has_motif = motif_prop_col is not None and motif_col is not None and motif_val_col is not None and not motif.empty
    has_physics = physics_prop_col is not None and physics_sign_col is not None and not physics.empty
    has_nn = nn_prop_col is not None and nn_val_col is not None and not nearest.empty
    if not has_desc and not has_motif and not has_physics and not has_nn:
        ok, render_mode = _render_panels_or_placeholder(
            output_path=output_path,
            fallback_panels=fallback_panels,
            font_size=font_size,
            dpi=dpi,
            empty_title=fallback_title,
            empty_lines=fallback_lines,
        )
        return ok, _caption_with_render_note(caption, render_mode, fallback_note), render_mode

    fig, axes = plt.subplots(2, 2, figsize=(16.4, 10.8), squeeze=False, constrained_layout=True)
    ax0, ax1 = axes[0]
    ax2, ax3 = axes[1]

    if has_desc:
        d = desc[[prop_col, desc_col, desc_val_col]].copy()
        d.columns = ["property", "descriptor", "value"]
        d["property"] = d["property"].astype(str).str.strip()
        d["descriptor"] = d["descriptor"].astype(str).str.strip()
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d = d.dropna(subset=["property", "descriptor", "value"])
        if d.empty:
            has_desc = False
        else:
            top_desc = (
                d.groupby("descriptor")["value"]
                .apply(lambda s: float(np.nanmean(np.abs(s.to_numpy(dtype=float)))))
                .sort_values(ascending=False)
                .head(8)
                .index.tolist()
            )
            d = d[d["descriptor"].isin(top_desc)]
            prop_order = _ordered_properties(d["property"].tolist())
            mat_df = d.pivot_table(index="property", columns="descriptor", values="value", aggfunc="mean")
            mat_df = mat_df.reindex(index=prop_order, columns=top_desc)
            mat = mat_df.to_numpy(dtype=float)
            finite = np.isfinite(mat)
            if finite.any():
                mean = np.nanmean(mat)
                std = np.nanstd(mat)
                if std > 1e-12:
                    mat = (mat - mean) / std
            im = ax0.imshow(np.ma.masked_invalid(mat), cmap=_nature_diverging_cmap(), aspect="auto")
            ax0.set_xticks(np.arange(len(top_desc)))
            ax0.set_xticklabels(top_desc, rotation=35, ha="right")
            ax0.set_yticks(np.arange(len(prop_order)))
            ax0.set_yticklabels(prop_order)
            ax0.set_xlabel("Descriptor")
            ax0.set_ylabel("Property")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    if np.isnan(val):
                        continue
                    ax0.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=max(9, font_size - 6))
            cbar0 = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
            cbar0.set_label("Normalized Shift")
    if not has_desc:
        ax0.text(0.5, 0.5, "Descriptor-shift data unavailable", ha="center", va="center", transform=ax0.transAxes)
        ax0.set_xticks([])
        ax0.set_yticks([])
    _panel_mark(ax0, "(a)", font_size)

    if has_physics:
        p = physics[[physics_prop_col, physics_sign_col]].copy()
        p.columns = ["property", "sign_match"]
        p["property"] = p["property"].astype(str).str.strip()
        p["sign_match"] = p["sign_match"].fillna(False).astype(bool)
        p = p.dropna(subset=["property"])
        if not p.empty:
            summary = (
                p.groupby("property", as_index=False)
                .agg(
                    consistency_rate=("sign_match", "mean"),
                    matched_rules=("sign_match", "sum"),
                    total_rules=("sign_match", "size"),
                )
                .set_index("property")
                .reindex(_ordered_properties(p["property"].tolist()))
                .dropna(how="all")
                .reset_index()
            )
            y = np.arange(len(summary))
            rates = pd.to_numeric(summary["consistency_rate"], errors="coerce").to_numpy(dtype=float)
            ax1.barh(
                y,
                rates,
                color=NATURE_PALETTE[1],
                edgecolor=COLOR_TEXT,
                linewidth=0.7,
            )
            ax1.set_yticks(y)
            ax1.set_yticklabels(summary["property"].tolist())
            ax1.set_xlabel("Physics-rule consistency rate")
            ax1.set_ylabel("Property")
            ax1.set_xlim(0.0, 1.0)
            ax1.grid(True, axis="x", linestyle="--", alpha=0.4)
            for yi, rate, matched, total in zip(
                y,
                rates,
                pd.to_numeric(summary["matched_rules"], errors="coerce").fillna(0).astype(int).tolist(),
                pd.to_numeric(summary["total_rules"], errors="coerce").fillna(0).astype(int).tolist(),
            ):
                if not np.isfinite(rate):
                    continue
                ax1.text(
                    min(rate + 0.02, 0.98),
                    yi,
                    f"{matched}/{max(total, 1)}",
                    va="center",
                    ha="left",
                    fontsize=max(9, font_size - 6),
                )
        else:
            has_physics = False
    if not has_physics:
        ax1.text(0.5, 0.5, "Physics-consistency data unavailable", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xticks([])
        ax1.set_yticks([])
    _panel_mark(ax1, "(b)", font_size)

    if has_motif:
        m = motif[[motif_prop_col, motif_col, motif_val_col]].copy()
        m.columns = ["property", "motif", "value"]
        m["motif"] = m["motif"].astype(str).str.strip()
        m["value"] = pd.to_numeric(m["value"], errors="coerce")
        m = m.dropna(subset=["motif", "value"])
        if not m.empty:
            if motif_val_col == "log2_enrichment_topk_vs_ref":
                m["score"] = m["value"].to_numpy(dtype=float)
                y_label = "Log2 Enrichment"
            elif motif_val_col == "delta_freq_topk_vs_ref":
                m["score"] = m["value"].to_numpy(dtype=float)
                y_label = "Delta Frequency (Top-k - Ref)"
            else:
                m["score"] = np.log2(np.clip(m["value"].to_numpy(dtype=float), a_min=1e-8, a_max=None))
                y_label = "Log2 Enrichment"
            motif_mean = (
                m.groupby("motif", as_index=False)["score"].mean(numeric_only=True)
            )
            top = (
                motif_mean.assign(abs_score=lambda x: np.abs(pd.to_numeric(x["score"], errors="coerce")))
                .sort_values("abs_score", ascending=False)
                .head(10)
                .sort_values("score", ascending=False)
            )
            bar_colors = np.where(
                pd.to_numeric(top["score"], errors="coerce").to_numpy(dtype=float) >= 0.0,
                NATURE_PALETTE[0],
                NATURE_PALETTE[3],
            )
            ax2.bar(
                np.arange(len(top)),
                top["score"].to_numpy(dtype=float),
                color=bar_colors,
                edgecolor=COLOR_TEXT,
                linewidth=0.7,
            )
            ax2.set_xticks(np.arange(len(top)))
            ax2.set_xticklabels(top["motif"].tolist(), rotation=35, ha="right")
            ax2.set_ylabel(y_label)
            ax2.axhline(0.0, color=COLOR_TEXT, linewidth=1.0, linestyle="--")
            ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
            _wrap_ticklabels(ax2, axis="x", width=14, rotation=35)
        else:
            has_motif = False
    if not has_motif:
        ax2.text(0.5, 0.5, "Motif-enrichment data unavailable", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    _panel_mark(ax2, "(c)", font_size)

    if has_nn:
        n = nearest[[nn_prop_col, nn_val_col]].copy()
        n.columns = ["property", "nearest_tanimoto"]
        n["property"] = n["property"].astype(str).str.strip()
        n["nearest_tanimoto"] = pd.to_numeric(n["nearest_tanimoto"], errors="coerce")
        n = n.dropna(subset=["property", "nearest_tanimoto"])
        if not n.empty:
            nn_props = _ordered_properties(n["property"].tolist())
            if len(nn_props) > 1:
                box_data = []
                box_labels = []
                for prop in nn_props:
                    vals = n.loc[n["property"] == prop, "nearest_tanimoto"].dropna().to_numpy(dtype=float)
                    if vals.size == 0:
                        continue
                    box_data.append(vals)
                    box_labels.append(prop)
                if box_data:
                    bp = ax3.boxplot(box_data, patch_artist=True, showfliers=False)
                    for idx, box in enumerate(bp["boxes"]):
                        box.set_facecolor(NATURE_PALETTE[idx % len(NATURE_PALETTE)])
                        box.set_alpha(0.85)
                    ax3.set_xticks(np.arange(1, len(box_labels) + 1))
                    ax3.set_xticklabels(box_labels, rotation=30, ha="right")
                    ax3.set_xlabel("Property")
                    ax3.set_ylabel("Nearest-neighbor Tanimoto")
                    ax3.set_ylim(0.0, 1.0)
                    ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
                else:
                    has_nn = False
            else:
                vals = n["nearest_tanimoto"].dropna().to_numpy(dtype=float)
                if vals.size:
                    bins = min(12, max(4, int(np.sqrt(vals.size))))
                    ax3.hist(vals, bins=bins, color=NATURE_PALETTE[2], edgecolor=COLOR_TEXT, linewidth=0.7)
                    ax3.axvline(float(np.nanmedian(vals)), color=NATURE_PALETTE[3], linestyle="--", linewidth=1.4)
                    ax3.set_xlabel("Nearest-neighbor Tanimoto")
                    ax3.set_ylabel("Count")
                    ax3.set_xlim(0.0, 1.0)
                    ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
                else:
                    has_nn = False
        else:
            has_nn = False
    if not has_nn:
        ax3.text(0.5, 0.5, "Nearest-neighbor data unavailable", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_xticks([])
        ax3.set_yticks([])
    _panel_mark(ax3, "(d)", font_size)

    return _save_plot_figure(fig, output_path, dpi), caption, "data"


def main(args):
    config = load_config(args.config)
    paper_cfg = config.get("paper_results", {}) or {}

    enabled = _to_bool(paper_cfg.get("enabled", True), True)
    if args.disable:
        enabled = False
    if not enabled:
        print("Paper package export disabled by config/flag.")
        return

    include_large_csv = _to_bool(paper_cfg.get("include_large_csv", True), True)
    include_figures = _to_bool(paper_cfg.get("include_figures", True), True)
    if args.skip_large_csv:
        include_large_csv = False
    if args.no_figures:
        include_figures = False

    # Requested fixed behavior: always generate 6 manuscript figures.
    manuscript_figure_count = MANUSCRIPT_FIGURE_COUNT

    max_panels_per_figure = _to_int(
        args.max_panels_per_figure
        if args.max_panels_per_figure is not None
        else paper_cfg.get("max_panels_per_figure", 4),
        4,
    )
    max_panels_per_figure = max(1, min(12, max_panels_per_figure))

    # Paper-package figures are hard-locked to 16pt for consistency with the
    # MVF publication figure policy, regardless of config or CLI overrides.
    figure_fontsize = 16

    figure_dpi = _to_int(
        args.figure_dpi if args.figure_dpi is not None else paper_cfg.get("figure_dpi", 600),
        600,
    )
    figure_dpi = max(72, min(1200, figure_dpi))

    method_dirs = _parse_method_dirs(args.method_dirs if args.method_dirs else paper_cfg.get("method_dirs"))

    if args.results_dir:
        results_dir = _resolve_path(args.results_dir)
    else:
        results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    explicit_results_dirs = _parse_results_dirs(paper_cfg.get("results_dirs"))
    auto_discover_results_dirs = _to_bool(paper_cfg.get("auto_discover_results_dirs", False), False)
    mvf_results_dirs = _discover_mvf_results_dirs(
        results_dir,
        explicit_results_dirs=explicit_results_dirs,
        auto_discover=auto_discover_results_dirs,
    )

    cfg_output = str(paper_cfg.get("output_dir", "")).strip()
    if args.output_dir:
        output_dir = _resolve_path(args.output_dir)
    elif cfg_output:
        cfg_path = Path(cfg_output)
        output_dir = cfg_path if cfg_path.is_absolute() else (results_dir / cfg_path)
    else:
        output_dir = results_dir / "paper_package"

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    manifest_dir = output_dir / "manifest"

    # Legacy structure kept for backward compatibility.
    tables_main_dir = output_dir / "tables" / "main"
    tables_supp_dir = output_dir / "tables" / "supplementary"
    figures_dir = output_dir / "figures"

    # New paper-organization structure requested by user.
    manuscript_results_dir = output_dir / "manuscript" / "results"
    manuscript_figures_dir = output_dir / "manuscript" / "figures"
    manuscript_captions_dir = output_dir / "manuscript" / "captions"

    si_results_dir = output_dir / "supporting_information" / "results"
    si_figures_dir = output_dir / "supporting_information" / "figures"
    si_captions_dir = output_dir / "supporting_information" / "captions"

    run_meta_dir = manifest_dir / "run_meta"

    for directory in [
        manifest_dir,
        tables_main_dir,
        tables_supp_dir,
        figures_dir,
        manuscript_results_dir,
        manuscript_figures_dir,
        manuscript_captions_dir,
        si_results_dir,
        si_figures_dir,
        si_captions_dir,
        run_meta_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    missing: list[dict] = []
    step_rows: list[dict] = []
    paper_table_map: dict[str, str] = {}

    source_config_candidates = [
        results_dir / "config_used.yaml",
        _resolve_path(args.config),
    ]
    cfg_src = _copy_first(
        source_config_candidates,
        manifest_dir / "config_used.yaml",
        category="manifest",
        copied=copied,
        missing=missing,
        results_dir=results_dir,
        output_dir=output_dir,
        missing_label="config_used.yaml",
    )
    if cfg_src is not None:
        _copy_file(
            cfg_src,
            manuscript_results_dir / "config_used.yaml",
            category="manifest",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    # F1 summary table from embedding meta across all detected MVF model-size runs.
    f1_df = _f1_embedding_summary_multi(mvf_results_dirs)
    f1_table_path = tables_main_dir / "table_f1_embedding_summary.csv"
    if not f1_df.empty:
        f1_df.to_csv(f1_table_path, index=False)
        copied.append(
            {
                "category": "table_main",
                "source": "generated_from_embedding_meta",
                "destination": _safe_rel(f1_table_path, output_dir),
            }
        )
        f1_df.to_csv(manuscript_results_dir / "table_f1_embedding_summary.csv", index=False)
        copied.append(
            {
                "category": "table_main",
                "source": "generated_from_embedding_meta",
                "destination": _safe_rel(manuscript_results_dir / "table_f1_embedding_summary.csv", output_dir),
            }
        )
    else:
        missing.append({"category": "table_main", "name": "table_f1_embedding_summary.csv"})
    paper_table_map["F1"] = _safe_rel(f1_table_path, output_dir)

    step_metric_map = [
        ("F2", "retrieval", "step2_retrieval", "metrics_alignment.csv", "table_f2_retrieval.csv"),
        ("F3", "property_heads", "step3_property", "metrics_property.csv", "table_f3_property_heads.csv"),
        ("F4", "embedding_research", "step4_embedding_research", "metrics_embedding_research.csv", "table_f4_embedding_research.csv"),
        ("F5", "foundation_inverse", "step5_foundation_inverse", "metrics_inverse.csv", "table_f5_inverse_design.csv"),
        (
            "F6",
            "dit_interpretability",
            "step6_dit_interpretability",
            "metrics_dit_interpretability.csv",
            "table_f6_dit_interpretability.csv",
        ),
        (
            "F7",
            "chem_physics_analysis",
            "step7_chem_physics_analysis",
            "metrics_chem_physics.csv",
            "table_f7_chem_physics.csv",
        ),
    ]

    step_metric_sources: dict[str, Optional[Path]] = {}
    for step_id, step_name, step_subdir, filename, out_name in step_metric_map:
        main_dst = tables_main_dir / out_name
        paper_table_map[step_id] = _safe_rel(main_dst, output_dir)
        include_property_scopes = step_id in {"F5", "F6", "F7"}
        suffix_regex = r"^metrics_dit_interpretability_(.+)\.csv$" if step_id == "F6" else None
        df_step = _load_mvf_csv_multi(
            mvf_results_dirs,
            step_subdir,
            filename,
            include_property_scopes=include_property_scopes,
            suffixed_filename_regex=suffix_regex,
        )
        if df_step.empty:
            missing.append({"category": "table_main", "name": out_name})
            step_metric_sources[step_id] = None
            continue

        main_dst.parent.mkdir(parents=True, exist_ok=True)
        df_step.to_csv(main_dst, index=False)
        copied.append(
            {
                "category": "table_main",
                "source": f"generated_from_{step_subdir}/{filename}_across_sizes",
                "destination": _safe_rel(main_dst, output_dir),
            }
        )
        step_metric_sources[step_id] = main_dst

        manuscript_dst = manuscript_results_dir / out_name
        manuscript_dst.parent.mkdir(parents=True, exist_ok=True)
        df_step.to_csv(manuscript_dst, index=False)
        copied.append(
            {
                "category": "table_main",
                "source": f"generated_from_{step_subdir}/{filename}_across_sizes",
                "destination": _safe_rel(manuscript_dst, output_dir),
            }
        )

    step4_dirs: list[Path] = []
    step5_dirs: list[Path] = []
    step6_dirs: list[Path] = []
    step7_files_dirs: list[Path] = []
    for mvf_dir in mvf_results_dirs:
        step4_dirs.extend(
            _collect_step_artifact_dirs(
                mvf_results_dir=mvf_dir,
                step_subdir="step4_embedding_research",
                include_property_scopes=False,
                include_root_when_property_scopes=True,
            )
        )
        step5_dirs.extend(
            _collect_step_artifact_dirs(
                mvf_results_dir=mvf_dir,
                step_subdir="step5_foundation_inverse",
                include_property_scopes=True,
                include_root_when_property_scopes=False,
            )
        )
        step6_dirs.extend(
            _collect_step_artifact_dirs(
                mvf_results_dir=mvf_dir,
                step_subdir="step6_dit_interpretability",
                include_property_scopes=True,
                include_root_when_property_scopes=False,
            )
        )
        step7_files_dirs.extend(
            _collect_step_artifact_dirs(
                mvf_results_dir=mvf_dir,
                step_subdir="step7_chem_physics_analysis",
                include_property_scopes=True,
                include_root_when_property_scopes=True,
            )
        )
    step4_dirs = _unique_paths(step4_dirs)
    step5_dirs = _unique_paths(step5_dirs)
    step6_dirs = _unique_paths(step6_dirs)
    step7_files_dirs = _unique_paths(step7_files_dirs)

    properties = _discover_properties(config, step5_dirs, step6_dirs, step_metric_sources.get("F7"))

    f5_artifacts: list[Path] = []
    f6_patterns = ["dit_token_summary*.csv", "metrics_dit_interpretability*.csv"]
    if include_large_csv:
        f6_patterns = [
            "dit_case_attributions*.csv",
            "dit_method_agreement*.csv",
            "dit_token_summary*.csv",
            "interpretability_selected_cases*.csv",
            "metrics_dit_interpretability*.csv",
            "dit_token_attributions*.csv",
        ]

    # Prefer property-scoped suffixed files to avoid double-copying duplicates
    # (e.g., candidate_scores.csv and candidate_scores_<PROPERTY>.csv in the same scope).
    f5_seen: set[str] = set()

    def _append_f5_unique(paths: list[Path]) -> None:
        for p in paths:
            key = str(p.resolve())
            if key in f5_seen:
                continue
            f5_seen.add(key)
            f5_artifacts.append(p)

    if include_large_csv:
        candidate_prop = _collect_glob_unique(step5_dirs, ["candidate_scores_*.csv"])
        if candidate_prop:
            _append_f5_unique(candidate_prop)
        else:
            _append_f5_unique(_collect_glob_unique(step5_dirs, ["candidate_scores.csv"]))

    accepted_prop = _collect_glob_unique(step5_dirs, ["accepted_candidates_*.csv"])
    if accepted_prop:
        _append_f5_unique(accepted_prop)
    else:
        _append_f5_unique(_collect_glob_unique(step5_dirs, ["accepted_candidates.csv"]))

    view_compare_patterns = [
        "metrics_view_compare*.csv",
        "view_compare_topk*.csv",
        "view_compare_descriptor_summary*.csv",
        "view_compare_class_distribution*.csv",
        "view_compare_space*.csv",
    ]
    if include_large_csv:
        view_compare_patterns.insert(1, "view_compare_scores*.csv")
    _append_f5_unique(_collect_glob_unique(step5_dirs, view_compare_patterns))

    def _size_tag_for(path: Path) -> str:
        for mvf_dir in mvf_results_dirs:
            if _is_relative_to(path, mvf_dir):
                return _normalize_model_size(_infer_model_size_from_results_dir(mvf_dir))
        return "unknown"

    for src in _collect_glob_unique(
        step4_dirs,
        [
            "view_geometry_summary.csv",
            "view_retrieval_summary.csv",
            "view_tokenizer_efficiency.csv",
            "view_semantic_structure.csv",
            "view_complementarity_summary.csv",
        ],
    ):
        rel_dst = _supplementary_dst_name(src, _size_tag_for(src))
        _copy_file(
            src,
            tables_supp_dir / rel_dst,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )
        _copy_file(
            src,
            si_results_dir / rel_dst,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    for src in f5_artifacts:
        rel_dst = _supplementary_dst_name(src, _size_tag_for(src))
        _copy_file(
            src,
            tables_supp_dir / rel_dst,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )
        _copy_file(
            src,
            si_results_dir / rel_dst,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    for src in _collect_glob_unique(step6_dirs, f6_patterns):
        rel_dst = _supplementary_dst_name(src, _size_tag_for(src))
        _copy_file(
            src,
            tables_supp_dir / rel_dst,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )
        _copy_file(
            src,
            si_results_dir / rel_dst,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    for src in _collect_glob_unique(
        step7_files_dirs,
        [
            "descriptor_shifts.csv",
            "motif_enrichment.csv",
            "physics_consistency.csv",
            "nearest_neighbor_explanations.csv",
            "design_filter_audit.csv",
            "property_input_files.csv",
        ],
    ):
        rel_dst = _supplementary_dst_name(src, _size_tag_for(src))
        _copy_file(
            src,
            tables_supp_dir / rel_dst,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )
        _copy_file(
            src,
            si_results_dir / rel_dst,
            category="table_supplementary",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    _copy_base_method_csvs(
        method_dirs=method_dirs,
        tables_supp_dir=tables_supp_dir,
        copied=copied,
        missing=missing,
        results_dir=results_dir,
        output_dir=output_dir,
    )
    _copy_tree_files(
        source_dir=tables_supp_dir / "base_methods",
        destination_dir=si_results_dir / "base_methods",
        category="table_supplementary",
        copied=copied,
        results_dir=results_dir,
        output_dir=output_dir,
    )

    step12_scale_summary_df = _collect_step12_scale_summary(method_dirs)
    if step12_scale_summary_df.empty:
        missing.append({"category": "table_supplementary", "name": "base_methods/aggregates/step12_scale_summary.csv"})
    else:
        step12_main = tables_supp_dir / "base_methods" / "aggregates" / "step12_scale_summary.csv"
        step12_si = si_results_dir / "base_methods" / "aggregates" / "step12_scale_summary.csv"
        step12_main.parent.mkdir(parents=True, exist_ok=True)
        step12_si.parent.mkdir(parents=True, exist_ok=True)
        step12_scale_summary_df.to_csv(step12_main, index=False)
        step12_scale_summary_df.to_csv(step12_si, index=False)
        copied.append(
            {
                "category": "table_supplementary",
                "source": "generated_from_step1_step2_method_results",
                "destination": _safe_rel(step12_main, output_dir),
            }
        )
        copied.append(
            {
                "category": "table_supplementary",
                "source": "generated_from_step1_step2_method_results",
                "destination": _safe_rel(step12_si, output_dir),
            }
        )

    step_rows = _build_step_status_rows(
        results_dirs=mvf_results_dirs,
        output_dir=output_dir,
        results_dir=results_dir,
        paper_table_map=paper_table_map,
    )
    step_status_map = {row["step_id"]: row for row in step_rows}

    generated_manuscript_figures: list[str] = []
    generated_si_figures: list[str] = []
    manuscript_caption_lines: list[str] = []
    si_caption_lines: list[str] = []
    manuscript_figure_records: list[dict] = []
    si_figure_records: list[dict] = []

    if include_figures:
        step_figure_roots: list[Path] = []
        for mvf_dir in mvf_results_dirs:
            step_figure_roots.extend(
                _existing_dirs(
                    [
                        mvf_dir / "step1_alignment_embeddings",
                        mvf_dir / "step2_retrieval",
                        mvf_dir / "step3_property",
                        mvf_dir / "step4_embedding_research",
                        mvf_dir / "step5_foundation_inverse",
                        mvf_dir / "step6_dit_interpretability",
                        mvf_dir / "step6_view_compare_analysis",
                        mvf_dir / "step6_ood_aware_inverse",
                        mvf_dir / "step7_chem_physics_analysis",
                    ]
                )
            )
        step_figure_roots = _unique_paths(step_figure_roots)

        def _mvf_source_tag(path: Path) -> str:
            for mvf_dir in mvf_results_dirs:
                if _is_relative_to(path, mvf_dir):
                    size = _normalize_model_size(_infer_model_size_from_results_dir(mvf_dir))
                    return f"{size}_{mvf_dir.name}"
            return "mvf_unknown"

        seen_mvf_fig_names: set[str] = set()
        for root in step_figure_roots:
            for src in sorted(root.rglob("figure_f*.png")):
                if _is_relative_to(src, output_dir):
                    continue
                base_name = src.name
                name = base_name
                if name in seen_mvf_fig_names:
                    name = f"{_mvf_source_tag(src)}_{base_name}"
                dupe_idx = 2
                while name in seen_mvf_fig_names:
                    stem = Path(base_name).stem
                    suffix = Path(base_name).suffix
                    name = f"{_mvf_source_tag(src)}_{stem}_{dupe_idx}{suffix}"
                    dupe_idx += 1
                seen_mvf_fig_names.add(name)
                _copy_file(
                    src,
                    figures_dir / name,
                    category="figure",
                    copied=copied,
                    results_dir=results_dir,
                    output_dir=output_dir,
                )

        if plt is None:
            missing.append(
                {
                    "category": "figure",
                    "name": "matplotlib_not_available_for_composite_figures",
                }
            )
        else:
            _set_publication_style(figure_fontsize)
            generated_step12_scale_panels = _generate_step12_scale_panels(
                summary_df=step12_scale_summary_df,
                panel_dir=output_dir / "figures" / "generated_step12_scale",
                font_size=figure_fontsize,
                dpi=figure_dpi,
                max_panels_per_figure=max_panels_per_figure,
            )
            for panel in generated_step12_scale_panels:
                copied.append(
                    {
                        "category": "figure",
                        "source": "generated_from_step1_step2_method_results",
                        "destination": _safe_rel(panel, output_dir),
                    }
                )

            themed_panels = _collect_source_panels(
                results_dirs=mvf_results_dirs,
                output_dir=output_dir,
                method_dirs=method_dirs,
            )
            if generated_step12_scale_panels:
                base_panels = themed_panels.setdefault("base_step12", [])
                seen_base = {str(p.resolve()) for p in base_panels if p.exists()}
                for panel in generated_step12_scale_panels:
                    key = str(panel.resolve())
                    if key in seen_base:
                        continue
                    base_panels.insert(0, panel)
                    seen_base.add(key)

            repo_results_root = REPO_ROOT / "results"
            figure1_primary_panels = _figure1_primary_panels(repo_results_root)
            df_agg_generation = _load_aggregate_csv(repo_results_root, "metrics_generation.csv")
            df_agg_alignment = _load_aggregate_csv(repo_results_root, "metrics_alignment.csv")
            df_agg_property = _load_aggregate_csv(repo_results_root, "metrics_property.csv")
            df_agg_inverse = _load_aggregate_csv(repo_results_root, "metrics_inverse.csv")

            df_mvf_alignment = _load_mvf_csv_multi(mvf_results_dirs, "step2_retrieval", "metrics_alignment.csv")
            df_mvf_property = _load_mvf_csv_multi(
                mvf_results_dirs,
                "step3_property",
                "metrics_property.csv",
                include_property_scopes=True,
            )
            df_mvf_embedding = _load_mvf_csv_multi(mvf_results_dirs, "step4_embedding_research", "metrics_embedding_research.csv")
            df_mvf_inverse = _load_mvf_csv_multi(
                mvf_results_dirs,
                "step5_foundation_inverse",
                "metrics_inverse.csv",
                include_property_scopes=True,
            )
            df_mvf_inverse_topk = _load_mvf_csv_multi(
                mvf_results_dirs,
                "step5_foundation_inverse",
                "metrics_view_compare.csv",
                include_property_scopes=True,
                suffixed_filename_regex=r"^metrics_view_compare_(.+)\.csv$",
            )
            df_mvf_desc = _load_mvf_csv_multi(
                mvf_results_dirs,
                "step7_chem_physics_analysis",
                "descriptor_shifts.csv",
                include_property_scopes=True,
            )
            df_mvf_motif = _load_mvf_csv_multi(
                mvf_results_dirs,
                "step7_chem_physics_analysis",
                "motif_enrichment.csv",
                include_property_scopes=True,
            )
            df_mvf_physics = _load_mvf_csv_multi(
                mvf_results_dirs,
                "step7_chem_physics_analysis",
                "physics_consistency.csv",
                include_property_scopes=True,
            )
            df_mvf_nearest = _load_mvf_csv_multi(
                mvf_results_dirs,
                "step7_chem_physics_analysis",
                "nearest_neighbor_explanations.csv",
                include_property_scopes=True,
            )

            def _append_theme_panel(theme: str, panel: Path, *, prepend: bool = False) -> None:
                bucket = themed_panels.setdefault(theme, [])
                key = str(panel.resolve())
                seen_bucket = {str(p.resolve()) for p in bucket if p.exists()}
                if key in seen_bucket:
                    return
                if prepend:
                    bucket.insert(0, panel)
                else:
                    bucket.append(panel)

            derived_panel_dir = figures_dir / "derived_step_panels"
            derived_panel_dir.mkdir(parents=True, exist_ok=True)
            f1_panel_path = derived_panel_dir / "figure_f1_embedding_summary_f8.png"
            if _build_f1_summary_panel(
                output_path=f1_panel_path,
                font_size=figure_fontsize,
                dpi=figure_dpi,
                f1_df=f1_df,
            ):
                _append_theme_panel("mvf_f1", f1_panel_path, prepend=True)

            manuscript_specs = [
                {
                    "themes": ["base_step0", "base_step12"],
                    "panel_fallback_order": ["base_step0", "base_step12"],
                    "step_ids": [],
                    "fallback_title": "Figure 1 unavailable",
                    "fallback_note": "No baseline generation metrics or baseline source panels were available during F8 export.",
                    "fn": _fig1_baseline_generation,
                    "kwargs": {
                        "df_generation": df_agg_generation,
                        "df_step12": step12_scale_summary_df,
                        "primary_panels": figure1_primary_panels,
                    },
                },
                {
                    "themes": ["mvf_f2"],
                    "panel_fallback_order": ["mvf_f2"],
                    "step_ids": ["F2"],
                    "fallback_title": "Figure 2 unavailable",
                    "fallback_note": "No non-empty MVF F2 retrieval outputs were available during F8 export.",
                    "si_seed_theme": "mvf_f2",
                    "fn": _fig2_mvf_alignment,
                    "kwargs": {
                        "df_alignment": df_mvf_alignment if not df_mvf_alignment.empty else df_agg_alignment,
                    },
                },
                {
                    "themes": ["mvf_f4"],
                    "panel_fallback_order": ["mvf_f4"],
                    "step_ids": ["F4"],
                    "fallback_title": "Figure 3 unavailable",
                    "fallback_note": "No non-empty MVF F4 embedding-research outputs were available during F8 export.",
                    "si_seed_theme": "mvf_f4",
                    "fn": _fig4_embedding_research,
                    "kwargs": {
                        "df_embedding_mvf": df_mvf_embedding,
                    },
                },
                {
                    "themes": ["mvf_f3"],
                    "panel_fallback_order": ["mvf_f3"],
                    "step_ids": ["F3"],
                    "fallback_title": "Figure 4 unavailable",
                    "fallback_note": "No non-empty MVF F3 property-head outputs were available during F8 export.",
                    "si_seed_theme": "mvf_f3",
                    "fn": _fig3_property_prediction,
                    "kwargs": {
                        "df_property_base": df_agg_property,
                        "df_property_mvf": df_mvf_property,
                    },
                },
                {
                    "themes": ["mvf_f5"],
                    "panel_fallback_order": ["mvf_f5"],
                    "step_ids": ["F5"],
                    "fallback_title": "Figure 5 unavailable",
                    "fallback_note": "No non-empty MVF F5 inverse-design outputs were available during F8 export.",
                    "si_seed_theme": "mvf_f5",
                    "fn": _fig5_inverse_design,
                    "kwargs": {
                        "df_inverse_base": df_agg_inverse,
                        "df_inverse_mvf": df_mvf_inverse,
                        "df_inverse_topk": df_mvf_inverse_topk,
                    },
                },
                {
                    "themes": ["mvf_f7"],
                    "panel_fallback_order": ["mvf_f7"],
                    "step_ids": ["F7"],
                    "fallback_title": "Figure 6 unavailable",
                    "fallback_note": "No non-empty MVF F7 chemistry/physics outputs were available during F8 export.",
                    "si_seed_theme": "mvf_f7",
                    "fn": _fig6_chem_physics,
                    "kwargs": {
                        "df_descriptor": df_mvf_desc,
                        "df_motif": df_mvf_motif,
                        "df_physics": df_mvf_physics,
                        "df_nearest": df_mvf_nearest,
                    },
                },
            ]
            manuscript_specs = manuscript_specs[:manuscript_figure_count]

            manuscript_used_keys: set[str] = set()
            for figure_num, spec in enumerate(manuscript_specs, start=1):
                fallback_panels = _pick_panels(
                    themed=themed_panels,
                    theme_group=spec["themes"],
                    used=manuscript_used_keys,
                    max_panels=max_panels_per_figure,
                    allow_reuse=False,
                    fallback_order=spec["panel_fallback_order"],
                )
                for panel in fallback_panels:
                    manuscript_used_keys.add(str(panel.resolve()))

                fig_path = manuscript_figures_dir / f"Figure_{figure_num}.png"
                fallback_lines = _placeholder_lines_for_step_ids(spec["step_ids"], step_status_map)
                ok, caption_text, render_mode = spec["fn"](
                    output_path=fig_path,
                    font_size=figure_fontsize,
                    dpi=figure_dpi,
                    fallback_panels=fallback_panels,
                    fallback_title=spec["fallback_title"],
                    fallback_lines=fallback_lines,
                    fallback_note=spec["fallback_note"],
                    **spec["kwargs"],
                )
                manuscript_figure_records.append(
                    {
                        "figure_id": f"Figure_{figure_num}",
                        "themes": list(spec["themes"]),
                        "step_ids": list(spec["step_ids"]),
                        "render_mode": render_mode,
                        "fallback_panel_count": len(fallback_panels),
                        "output_path": _safe_rel(fig_path, output_dir),
                    }
                )
                if ok:
                    generated_manuscript_figures.append(_safe_rel(fig_path, output_dir))
                    copied.append(
                        {
                            "category": "figure_manuscript",
                            "source": "generated_data_driven_or_fallback",
                            "destination": _safe_rel(fig_path, output_dir),
                        }
                    )
                    seed_theme = spec.get("si_seed_theme")
                    if render_mode == "data" and isinstance(seed_theme, str) and seed_theme:
                        _append_theme_panel(seed_theme, fig_path)
                else:
                    missing.append({"category": "figure_manuscript", "name": str(fig_path.name)})

                manuscript_caption_lines.append(caption_text)

            all_panels = _flatten_themed_panels(themed_panels, FIGURE_FALLBACK_ORDER)

            # Always keep traceable source panels in SI.
            source_panels_dir = si_figures_dir / "source_panels"
            source_panels_dir.mkdir(parents=True, exist_ok=True)
            for panel in all_panels:
                dst_name = _stable_hashed_copy_name(panel, root=REPO_ROOT, default_stem="panel")
                dst = source_panels_dir / dst_name
                if dst.exists():
                    counter = 2
                    stem = Path(dst_name).stem
                    suffix = Path(dst_name).suffix
                    while dst.exists():
                        dst = source_panels_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                _copy_file(
                    panel,
                    dst,
                    category="figure_source",
                    copied=copied,
                    results_dir=results_dir,
                    output_dir=output_dir,
                )

            si_specs = [
                {
                    "themes": ["base_step0"],
                    "step_ids": [],
                    "fallback_title": "Figure S1 unavailable",
                    "fallback_note": "No baseline Step0 panels were available during F8 export.",
                    "fallback_lines": [
                        "No baseline Step0 source panels were found.",
                        "Expected one of: step0_data_prep/figures/*.png from the five baseline methods.",
                    ],
                },
                {
                    "themes": ["base_step12"],
                    "step_ids": [],
                    "fallback_title": "Figure S2 unavailable",
                    "fallback_note": "No baseline Step1/Step2 panels were available during F8 export.",
                    "fallback_lines": [
                        "No baseline Step1/Step2 source panels were found.",
                        "Expected one of: aggregate_step12 figures or step1_backbone/step2_sampling panels.",
                    ],
                },
                {
                    "themes": ["mvf_f1"],
                    "step_ids": ["F1"],
                    "fallback_title": "Figure S3 unavailable",
                    "fallback_note": "No non-empty MVF F1 embedding-extraction outputs were available during F8 export.",
                },
                {
                    "themes": ["mvf_f2"],
                    "step_ids": ["F2"],
                    "fallback_title": "Figure S4 unavailable",
                    "fallback_note": "No non-empty MVF F2 retrieval outputs were available during F8 export.",
                },
                {
                    "themes": ["mvf_f3"],
                    "step_ids": ["F3"],
                    "fallback_title": "Figure S5 unavailable",
                    "fallback_note": "No non-empty MVF F3 property-head outputs were available during F8 export.",
                },
                {
                    "themes": ["mvf_f4"],
                    "step_ids": ["F4"],
                    "fallback_title": "Figure S6 unavailable",
                    "fallback_note": "No non-empty MVF F4 embedding-research outputs were available during F8 export.",
                },
                {
                    "themes": ["mvf_f5"],
                    "step_ids": ["F5"],
                    "fallback_title": "Figure S7 unavailable",
                    "fallback_note": "No non-empty MVF F5 inverse-design outputs were available during F8 export.",
                },
                {
                    "themes": ["mvf_f6"],
                    "step_ids": ["F6"],
                    "fallback_title": "Figure S8 unavailable",
                    "fallback_note": "No non-empty MVF F6 interpretability outputs were available during F8 export.",
                },
                {
                    "themes": ["mvf_f7"],
                    "step_ids": ["F7"],
                    "fallback_title": "Figure S9 unavailable",
                    "fallback_note": "No non-empty MVF F7 chemistry/physics outputs were available during F8 export.",
                },
            ]
            si_specs = si_specs[:SUPPORTING_INFORMATION_FIGURE_COUNT]

            si_used_keys: set[str] = set()
            for si_idx, spec in enumerate(si_specs, start=1):
                chunk = _pick_panels(
                    themed=themed_panels,
                    theme_group=spec["themes"],
                    used=si_used_keys,
                    max_panels=max_panels_per_figure,
                    allow_reuse=False,
                    fallback_order=spec["themes"],
                )
                for panel in chunk:
                    si_used_keys.add(str(panel.resolve()))

                fig_path = si_figures_dir / f"Figure_S{si_idx}.png"
                fallback_lines = list(spec.get("fallback_lines") or _placeholder_lines_for_step_ids(spec["step_ids"], step_status_map))
                ok, render_mode = _render_panels_or_placeholder(
                    output_path=fig_path,
                    fallback_panels=chunk,
                    font_size=figure_fontsize,
                    dpi=figure_dpi,
                    empty_title=spec["fallback_title"],
                    empty_lines=fallback_lines,
                )
                si_figure_records.append(
                    {
                        "figure_id": f"Figure_S{si_idx}",
                        "themes": list(spec["themes"]),
                        "step_ids": list(spec["step_ids"]),
                        "render_mode": render_mode,
                        "fallback_panel_count": len(chunk),
                        "output_path": _safe_rel(fig_path, output_dir),
                    }
                )
                if ok:
                    generated_si_figures.append(_safe_rel(fig_path, output_dir))
                    copied.append(
                        {
                            "category": "figure_supporting_information",
                            "source": "generated_theme_composite_or_placeholder",
                            "destination": _safe_rel(fig_path, output_dir),
                        }
                    )
                else:
                    missing.append({"category": "figure_supporting_information", "name": str(fig_path.name)})
                if si_idx <= len(SI_CAPTIONS):
                    si_caption_lines.append(_caption_with_render_note(SI_CAPTIONS[si_idx - 1], render_mode, spec["fallback_note"]))
                else:
                    si_caption_lines.append(
                        _caption_with_render_note(
                            f"Figure S{si_idx}. Supporting-information diagnostic panel.",
                            render_mode,
                            spec["fallback_note"],
                        )
                    )

            manuscript_caption_path = manuscript_captions_dir / "figure_captions.txt"
            si_caption_path = si_captions_dir / "figure_captions.txt"
            _write_caption_file(caption_lines=manuscript_caption_lines, caption_path=manuscript_caption_path)
            _write_caption_file(caption_lines=si_caption_lines, caption_path=si_caption_path)
            copied.append(
                {
                    "category": "caption_manuscript",
                    "source": "generated_from_panel_sources",
                    "destination": _safe_rel(manuscript_caption_path, output_dir),
                }
            )
            copied.append(
                {
                    "category": "caption_supporting_information",
                    "source": "generated_from_panel_sources",
                    "destination": _safe_rel(si_caption_path, output_dir),
                }
            )

    for src in _collect_glob_unique(
        step5_dirs + step6_dirs + step7_files_dirs,
        ["run_meta*.json"],
    ):
        dst_name = str(_supplementary_dst_name(src, _size_tag_for(src)))
        _copy_file(
            src,
            run_meta_dir / dst_name,
            category="manifest",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    copied_summary = _collapse_copied_artifacts(copied)

    manifest_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results_dir": str(results_dir),
        "mvf_results_dirs": [str(p) for p in mvf_results_dirs],
        "paper_package_dir": str(output_dir),
        "config_path": str(_resolve_path(args.config)),
        "properties_detected": properties,
        "include_large_csv": bool(include_large_csv),
        "include_figures": bool(include_figures),
        "manuscript_figure_count": manuscript_figure_count,
        "max_panels_per_figure": max_panels_per_figure,
        "figure_fontsize": figure_fontsize,
        "figure_dpi": figure_dpi,
        "method_dirs": method_dirs,
        "step12_scale_rows": int(len(step12_scale_summary_df)),
        "generated_manuscript_figures": generated_manuscript_figures,
        "generated_supporting_information_figures": generated_si_figures,
        "manuscript_figure_records": manuscript_figure_records,
        "supporting_information_figure_records": si_figure_records,
        "steps": step_rows,
        "copied_artifacts": copied_summary,
        "copy_operations": copied,
        "copied_artifact_count": len(copied_summary),
        "copy_operation_count": len(copied),
        "missing_artifacts": missing,
    }

    with open(manifest_dir / "pipeline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    pd.DataFrame(step_rows).to_csv(manifest_dir / "step_status.csv", index=False)

    print(f"Saved paper package to {output_dir}")
    print(f"Manuscript figures: {len(generated_manuscript_figures)}")
    print(f"Supporting-information figures: {len(generated_si_figures)}")
    print(f"Manifest: {manifest_dir / 'pipeline_manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--method_dirs", type=str, default=None)
    parser.add_argument("--max_panels_per_figure", type=int, default=None)
    parser.add_argument("--figure_fontsize", type=int, default=None)
    parser.add_argument("--figure_dpi", type=int, default=None)
    parser.add_argument("--skip_large_csv", action="store_true")
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--disable", action="store_true")
    parser.add_argument("--clean", action="store_true")
    main(parser.parse_args())
