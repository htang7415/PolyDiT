#!/usr/bin/env python
"""F8: Build a paper-ready package from F1-F7 outputs and 5-method baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
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
    plt = None

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from src.utils.config import load_config


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
    "cohesive_energy_density",
    "electron_affinity",
    "electron_injection_barrier",
    "ionization_energy",
]
STEP2_METRIC_COLUMNS = ["step2_validity", "step2_uniqueness", "step2_novelty", "step2_diversity"]
METHOD_PLOT_COLORS = {
    "Bi_Diffusion_SMILES": "#2563EB",
    "Bi_Diffusion_SMILES_BPE": "#0EA5E9",
    "Bi_Diffusion_SELFIES": "#16A34A",
    "Bi_Diffusion_Group_SELFIES": "#F97316",
    "Bi_Diffusion_graph": "#7C3AED",
}

MANUSCRIPT_CAPTIONS = [
    "Figure 1. Generation quality of five bidirectional diffusion models across molecular representation types. (a) Valid polymer fraction for each representation at the best-performing model size. (b) Unique fraction among valid generated polymers, reflecting generation diversity. (c) Validity-uniqueness trade-off colored by novelty.",
    "Figure 2. Cross-view molecular alignment in the multi-view foundation model. (a) Recall@1 heatmap: fraction of queries for which the correct paired molecule ranks first under cosine similarity. (b) Recall@10 heatmap: fraction of correct pairs recovered within the top-10 retrieved candidates.",
    "Figure 3. Property prediction with multi-view foundation embeddings across model scales. (a) Test-set R^2 versus model size across configured target properties, comparing the best baseline representation and the best MVF/fusion representation at each size. (b) Fusion gain in R^2 across model sizes.",
    "Figure 4. Out-of-distribution (OOD) shift analysis across model scales. (a) Mean distance between training-set (D1) and target-domain (D2) polymer embeddings as a function of model size for baseline and MVF. (b) Fraction of generated polymers within the D2 neighborhood across model sizes.",
    "Figure 5. Multi-view foundation model-guided inverse polymer design across model scales. (a) Inverse-design success rate versus model size across configured target properties, comparing baseline and MVF reranking. (b) Absolute success-rate improvement from MVF reranking across model sizes.",
    "Figure 6. Chemistry and physics analysis of inversely designed polymers. (a) Normalized descriptor shifts of accepted candidates relative to the D1 reference set for key physicochemical features and each target property. (b) Motif enrichment ratios for discriminative polymer substructures in accepted candidates compared to the reference distribution.",
]

SI_CAPTIONS = [
    "Figure S1. Distribution of sequence length and synthetic accessibility (SA) score in training and validation sets for each polymer representation.",
    "Figure S2. Training convergence and generation quality across representations and model sizes.",
    "Figure S3. Multi-view embedding extraction summary: embedding dimensionalities, model sizes, and sample counts per view.",
    "Figure S4. Cross-view retrieval evaluation: full Recall@K heatmaps across all view pairs.",
    "Figure S5. Property head training diagnostics: per-split metrics, head leaderboard, and coverage.",
    "Figure S6. OOD diagnostics: distance distributions between D1/D2 and generated embedding sets.",
    "Figure S7. Foundation-guided inverse design diagnostics: candidate score distributions and accepted-candidate profiles by view.",
    "Figure S8. OOD-aware objective diagnostics: conservative reranking score distributions and top-k candidate selection.",
    "Figure S9. Chemistry/physics analysis: per-property descriptor distributions, physics consistency checks, and nearest-neighbor explanations.",
]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
    text = str(value).strip()
    if not text:
        return ""
    p = Path(text)
    if p.suffix.lower() == ".csv":
        text = p.stem
    return text.strip()


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
        "step4_ood",
        "step5_foundation_inverse",
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
    elif "step4_ood" in lower_rel:
        step_hint = "MVF F4"
    elif "step5_foundation_inverse" in lower_rel:
        step_hint = "MVF F5"
    elif "step6_ood_aware_inverse" in lower_rel:
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
    if plt is None:
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
        }
    )


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

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 5.0 * nrows))
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

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04, hspace=0.06)
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


def _property_from_suffixed_filename(filename: str) -> str:
    m = re.match(
        r"^(?:candidate_scores|accepted_candidates|accepted_polymer_report|accepted_polymer_summary|"
        r"ood_objective_scores|ood_objective_topk|metrics_inverse_ood_objective|"
        r"descriptor_shifts|motif_enrichment|physics_consistency|nearest_neighbor_explanations|"
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
    newest_any_path: Optional[Path] = None

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
            newest_any_path = path
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
    fig1, ax1 = plt.subplots(figsize=(9.0, 6.2))
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
    fig1.tight_layout()
    p1 = panel_dir / "figure_step12_scale_step1_bpb_vs_size.png"
    fig1.savefig(p1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    generated.append(p1)

    # Panel 2: Step2 quality vs model size for each method.
    fig2, ax2 = plt.subplots(figsize=(9.0, 6.2))
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
    fig2.tight_layout()
    p2 = panel_dir / "figure_step12_scale_step2_quality_vs_size.png"
    fig2.savefig(p2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    generated.append(p2)

    # Panel 3: Step1-Step2 tradeoff frontier across methods and scales.
    fig3, ax3 = plt.subplots(figsize=(8.8, 6.2))
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
                edgecolors="#111827",
                linewidths=0.7,
                alpha=0.92,
            )
            ax3.text(x, y, size.upper(), fontsize=max(10, font_size - 4), ha="left", va="bottom")

    ax3.set_xlabel("Step1 BPB (lower is better)")
    ax3.set_ylabel("Step2 Composite Quality (higher is better)")
    ax3.grid(True, linestyle="--", alpha=0.45)
    fig3.tight_layout()
    p3 = panel_dir / "figure_step12_scale_frontier.png"
    fig3.savefig(p3, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    generated.append(p3)

    # Optional panel 4: metric ribbons for validity/novelty/diversity by size.
    if max_panels_per_figure >= 4:
        metric_plot_cols = ["step2_validity", "step2_novelty", "step2_diversity"]
        fig4, axes = plt.subplots(1, len(metric_plot_cols), figsize=(5.6 * len(metric_plot_cols), 5.6), squeeze=False)
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
        fig4.tight_layout()
        p4 = panel_dir / "figure_step12_scale_step2_metric_trends.png"
        fig4.savefig(p4, dpi=dpi, bbox_inches="tight")
        plt.close(fig4)
        generated.append(p4)

    return generated


def _render_fallback_panels(
    *,
    output_path: Path,
    fallback_panels: Sequence[Path],
    font_size: int,
    dpi: int,
) -> bool:
    return _compose_multi_panel_figure(
        panel_paths=fallback_panels,
        output_path=output_path,
        font_size=font_size,
        dpi=dpi,
    )


def _fig1_baseline_generation(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_generation: pd.DataFrame,
    df_step12: pd.DataFrame,
    primary_panels: Sequence[Path],
    fallback_panels: Sequence[Path],
) -> tuple[bool, str]:
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
        return ok, caption

    if plt is None:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

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
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    agg = (
        use_df.groupby(["representation_label", "model_size"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )
    agg = agg.dropna(subset=["validity", "uniqueness"], how="all")
    if agg.empty:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    best_idx = agg.groupby("representation_label")["validity"].idxmax()
    best = agg.loc[best_idx].copy().sort_values("validity", ascending=False).reset_index(drop=True)
    scatter_df = agg.copy()

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.2), squeeze=False)
    ax0, ax1, ax2 = axes[0]

    x = np.arange(len(best))
    ax0.bar(x, best["validity"].to_numpy(dtype=float), color="#4E79A7", edgecolor="#1F2937", linewidth=0.8)
    ax0.set_ylim(0.0, 1.02)
    ax0.set_ylabel("Validity")
    ax0.set_xticks(x)
    ax0.set_xticklabels(best["representation_label"].tolist(), rotation=25, ha="right")
    ax0.grid(True, axis="y", linestyle="--", alpha=0.4)
    _panel_mark(ax0, "(a)", font_size)

    ax1.bar(x, best["uniqueness"].to_numpy(dtype=float), color="#F28E2B", edgecolor="#1F2937", linewidth=0.8)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_ylabel("Uniqueness")
    ax1.set_xticks(x)
    ax1.set_xticklabels(best["representation_label"].tolist(), rotation=25, ha="right")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    _panel_mark(ax1, "(b)", font_size)

    novelty_vals = pd.to_numeric(scatter_df.get("novelty"), errors="coerce")
    scatter = ax2.scatter(
        pd.to_numeric(scatter_df.get("validity"), errors="coerce"),
        pd.to_numeric(scatter_df.get("uniqueness"), errors="coerce"),
        c=novelty_vals,
        cmap="viridis",
        s=140,
        edgecolors="#1F2937",
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

    fig.tight_layout()
    return _save_plot_figure(fig, output_path, dpi), caption


def _fig2_mvf_alignment(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_alignment: pd.DataFrame,
    fallback_panels: Sequence[Path],
) -> tuple[bool, str]:
    caption = MANUSCRIPT_CAPTIONS[1]
    if plt is None or df_alignment.empty:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    df = df_alignment.copy()
    r1_col = _first_existing_col(df, ["recall_at_1", "r1"])
    r10_col = _first_existing_col(df, ["recall_at_10", "r10"])
    if r1_col is None or r10_col is None:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    df = _coerce_numeric(df, [r1_col, r10_col])
    if {"source_view", "target_view"}.issubset(df.columns):
        df["source_label"] = df["source_view"].astype(str).map(_repr_label)
        df["target_label"] = df["target_view"].astype(str).map(_repr_label)
    elif "view_pair" in df.columns:
        pairs = df["view_pair"].apply(_split_view_pair)
        df["source_label"] = [p[0] for p in pairs]
        df["target_label"] = [p[1] for p in pairs]
    else:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    df = df.dropna(subset=["source_label", "target_label"])
    if df.empty:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    views = sorted(set(df["source_label"]).union(set(df["target_label"])))
    p1 = df.pivot_table(index="source_label", columns="target_label", values=r1_col, aggfunc="mean").reindex(index=views, columns=views)
    p10 = df.pivot_table(index="source_label", columns="target_label", values=r10_col, aggfunc="mean").reindex(index=views, columns=views)
    if p1.isna().all().all() and p10.isna().all().all():
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), squeeze=False)
    ax0, ax1 = axes[0]
    mats = [p1.to_numpy(dtype=float), p10.to_numpy(dtype=float)]
    for ax, mat, panel_tag in [(ax0, mats[0], "(a)"), (ax1, mats[1], "(b)")]:
        im = ax.imshow(np.ma.masked_invalid(mat), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
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
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=max(10, font_size - 5), color="#111827")
        _panel_mark(ax, panel_tag, font_size)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Recall")

    fig.tight_layout()
    return _save_plot_figure(fig, output_path, dpi), caption


def _fig3_property_prediction(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_property_base: pd.DataFrame,
    df_property_mvf: pd.DataFrame,
    fallback_panels: Sequence[Path],
) -> tuple[bool, str]:
    caption = MANUSCRIPT_CAPTIONS[2]
    if plt is None:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

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
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["property", "representation_label", "model_size", "r2"])
    if data.empty:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    size_order = [s for s in MODEL_SIZE_ORDER if s in set(data["model_size"].astype(str))]
    for s in sorted(set(data["model_size"].astype(str))):
        if s not in size_order:
            size_order.append(s)
    if not size_order:
        size_order = ["unknown"]

    prop_order = _ordered_properties(data["property"].tolist())

    agg = data.groupby(["property", "model_size", "representation_label", "source_name"], as_index=False)["r2"].mean(numeric_only=True)
    if agg.empty:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    fusion_tokens = ("mvf", "multiview", "multi view", "fusion", "mean")
    comp_rows: list[dict[str, object]] = []
    for prop in prop_order:
        for size in size_order:
            block = agg[(agg["property"] == prop) & (agg["model_size"] == size)]
            if block.empty:
                continue
            baseline_vals = block.loc[block["source_name"] == "baseline", "r2"]
            mvf_block = block.loc[block["source_name"] == "mvf"].copy()
            mvf_fusion_vals = mvf_block.loc[
                mvf_block["representation_label"].astype(str).str.lower().apply(
                    lambda t: any(tok in t for tok in fusion_tokens)
                ),
                "r2",
            ]
            baseline_best = float(baseline_vals.max()) if not baseline_vals.dropna().empty else np.nan
            if mvf_fusion_vals.dropna().empty:
                mvf_best = float(mvf_block["r2"].max()) if not mvf_block["r2"].dropna().empty else np.nan
            else:
                mvf_best = float(mvf_fusion_vals.max())
            gain = mvf_best - baseline_best if np.isfinite(mvf_best) and np.isfinite(baseline_best) else np.nan
            comp_rows.append(
                {
                    "property": prop,
                    "model_size": size,
                    "baseline_best_r2": baseline_best,
                    "mvf_best_r2": mvf_best,
                    "fusion_gain": gain,
                }
            )

    comp = pd.DataFrame(comp_rows)
    if comp.empty:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    size_to_x = {s: i for i, s in enumerate(size_order)}
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.2), squeeze=False)
    ax0, ax1 = axes[0]

    cmap = plt.get_cmap("tab10")
    for idx, prop in enumerate(prop_order):
        sub = comp[comp["property"] == prop].copy()
        if sub.empty:
            continue
        sub["x"] = sub["model_size"].map(size_to_x)
        sub = sub.dropna(subset=["x"]).sort_values("x")
        if sub.empty:
            continue
        color = cmap(idx % 10)
        x_vals = sub["x"].to_numpy(dtype=float)
        ax0.plot(
            x_vals,
            sub["baseline_best_r2"].to_numpy(dtype=float),
            linestyle="--",
            marker="o",
            linewidth=2.0,
            markersize=6.5,
            color=color,
            alpha=0.65,
            label=f"{prop} baseline",
        )
        ax0.plot(
            x_vals,
            sub["mvf_best_r2"].to_numpy(dtype=float),
            linestyle="-",
            marker="s",
            linewidth=2.2,
            markersize=6.5,
            color=color,
            alpha=0.95,
            label=f"{prop} MVF",
        )

    ax0.set_xticks(np.arange(len(size_order)))
    ax0.set_xticklabels([s.upper() for s in size_order])
    ax0.set_xlabel("Model Size")
    ax0.set_ylabel("Test R^2")
    ax0.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax0.legend(loc="best", frameon=False)
    _panel_mark(ax0, "(a)", font_size)

    gain_by_size = comp.groupby("model_size", as_index=False)["fusion_gain"].mean(numeric_only=True)
    gain_by_size["x"] = gain_by_size["model_size"].map(size_to_x)
    gain_by_size = gain_by_size.dropna(subset=["x"]).sort_values("x")
    ax1.bar(
        gain_by_size["x"].to_numpy(dtype=float),
        gain_by_size["fusion_gain"].to_numpy(dtype=float),
        color="#59A14F",
        edgecolor="#1F2937",
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
            color=cmap(idx % 10),
            alpha=0.8,
        )
    ax1.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.75)
    ax1.set_xticks(np.arange(len(size_order)))
    ax1.set_xticklabels([s.upper() for s in size_order])
    ax1.set_xlabel("Model Size")
    ax1.set_ylabel("Fusion Gain in R^2")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    _panel_mark(ax1, "(b)", font_size)

    fig.tight_layout()
    return _save_plot_figure(fig, output_path, dpi), caption


def _fig4_ood_analysis(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_ood_base: pd.DataFrame,
    df_ood_mvf: pd.DataFrame,
    fallback_panels: Sequence[Path],
) -> tuple[bool, str]:
    caption = MANUSCRIPT_CAPTIONS[3]
    if plt is None:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    frames: list[pd.DataFrame] = []
    for source, src_df in [("baseline", df_ood_base), ("mvf", df_ood_mvf)]:
        if src_df is None or src_df.empty:
            continue
        df = src_df.copy()
        dist_col = _first_existing_col(df, ["d1_to_d2_mean_dist", "d1_d2_mean_dist"])
        frac_col = _first_existing_col(df, ["frac_generated_near_d2", "fraction_generated_near_d2"])
        if dist_col is None and frac_col is None:
            continue
        size_col = _first_existing_col(df, ["model_size"])
        if size_col is not None:
            df["model_size"] = df[size_col].apply(_normalize_model_size)
        else:
            df["model_size"] = "unknown"
        if dist_col is not None:
            df["dist"] = pd.to_numeric(df[dist_col], errors="coerce")
        else:
            df["dist"] = np.nan
        if frac_col is not None:
            df["frac"] = pd.to_numeric(df[frac_col], errors="coerce")
        else:
            df["frac"] = np.nan
        df["source_name"] = source
        frames.append(df[["source_name", "model_size", "dist", "frac"]])

    if not frames:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    data = pd.concat(frames, ignore_index=True)
    data = data.groupby(["source_name", "model_size"], as_index=False)[["dist", "frac"]].mean(numeric_only=True)
    data = data.dropna(subset=["dist", "frac"], how="all")
    if data.empty:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    size_order = [s for s in MODEL_SIZE_ORDER if s in set(data["model_size"].astype(str))]
    for s in sorted(set(data["model_size"].astype(str))):
        if s not in size_order:
            size_order.append(s)
    if not size_order:
        size_order = ["unknown"]
    size_to_x = {s: i for i, s in enumerate(size_order)}

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.2), squeeze=False)
    ax0, ax1 = axes[0]
    for source, color, marker, style in [
        ("baseline", "#4E79A7", "o", "--"),
        ("mvf", "#F28E2B", "s", "-"),
    ]:
        sub = data[data["source_name"] == source].copy()
        if sub.empty:
            continue
        sub["x"] = sub["model_size"].map(size_to_x)
        sub = sub.dropna(subset=["x", "dist"]).sort_values("x")
        if sub.empty:
            continue
        label = "Baseline" if source == "baseline" else "MVF"
        ax0.plot(
            sub["x"].to_numpy(dtype=float),
            sub["dist"].to_numpy(dtype=float),
            linestyle=style,
            marker=marker,
            linewidth=2.2,
            markersize=7.2,
            color=color,
            label=label,
        )
    ax0.set_xticks(np.arange(len(size_order)))
    ax0.set_xticklabels([s.upper() for s in size_order])
    ax0.set_xlabel("Model Size")
    ax0.set_ylabel("D1-D2 Mean Distance")
    ax0.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax0.legend(loc="best", frameon=False)
    _panel_mark(ax0, "(a)", font_size)

    for source, color, marker, style in [
        ("baseline", "#4E79A7", "o", "--"),
        ("mvf", "#F28E2B", "s", "-"),
    ]:
        sub = data[data["source_name"] == source].copy()
        if sub.empty:
            continue
        sub["x"] = sub["model_size"].map(size_to_x)
        sub = sub.dropna(subset=["x", "frac"]).sort_values("x")
        if sub.empty:
            continue
        label = "Baseline" if source == "baseline" else "MVF"
        ax1.plot(
            sub["x"].to_numpy(dtype=float),
            sub["frac"].to_numpy(dtype=float),
            linestyle=style,
            marker=marker,
            linewidth=2.2,
            markersize=7.2,
            color=color,
            label=label,
        )
    ax1.set_xticks(np.arange(len(size_order)))
    ax1.set_xticklabels([s.upper() for s in size_order])
    ax1.set_xlabel("Model Size")
    ax1.set_ylabel("Fraction Near D2")
    max_frac = pd.to_numeric(data["frac"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ymax = float(max_frac.max() * 1.1) if not max_frac.empty else 1.0
    ax1.set_ylim(0.0, max(1.0, ymax))
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="best", frameon=False)
    _panel_mark(ax1, "(b)", font_size)

    fig.tight_layout()
    return _save_plot_figure(fig, output_path, dpi), caption


def _fig5_inverse_design(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_inverse_base: pd.DataFrame,
    df_inverse_mvf: pd.DataFrame,
    fallback_panels: Sequence[Path],
) -> tuple[bool, str]:
    caption = MANUSCRIPT_CAPTIONS[4]
    if plt is None:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    frames: list[pd.DataFrame] = []
    for source, src_df in [("baseline", df_inverse_base), ("mvf", df_inverse_mvf)]:
        if src_df is None or src_df.empty:
            continue
        df = src_df.copy()
        prop_col = _first_existing_col(df, ["property", "Property"])
        succ_col = _first_existing_col(df, ["success_rate"])
        size_col = _first_existing_col(df, ["model_size"])
        if prop_col is None or succ_col is None:
            continue
        df["property"] = df[prop_col].astype(str).str.strip()
        df["success_rate"] = pd.to_numeric(df[succ_col], errors="coerce")
        if size_col is not None:
            df["model_size"] = df[size_col].apply(_normalize_model_size)
        else:
            df["model_size"] = "unknown"
        rerank_col = _first_existing_col(df, ["rerank_success_rate"])
        if rerank_col is not None:
            df["rerank_success_rate"] = pd.to_numeric(df[rerank_col], errors="coerce")
        else:
            df["rerank_success_rate"] = np.nan
        flag_col = _first_existing_col(df, ["rerank_applied"])
        if flag_col is not None:
            df["rerank_applied"] = df[flag_col].astype(str).str.lower().isin(["1", "true", "yes", "y"])
        else:
            df["rerank_applied"] = False
        df["source_name"] = source
        frames.append(df[["property", "model_size", "success_rate", "rerank_success_rate", "rerank_applied", "source_name"]])

    if not frames:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["property", "model_size"])
    if data.empty:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    size_order = [s for s in MODEL_SIZE_ORDER if s in set(data["model_size"].astype(str))]
    for s in sorted(set(data["model_size"].astype(str))):
        if s not in size_order:
            size_order.append(s)
    if not size_order:
        size_order = ["unknown"]
    size_to_x = {s: i for i, s in enumerate(size_order)}

    prop_order = _ordered_properties(data["property"].tolist())

    rows: list[dict[str, object]] = []
    for prop in prop_order:
        for size in size_order:
            baseline_vals = data[
                (data["property"] == prop)
                & (data["model_size"] == size)
                & (data["source_name"] == "baseline")
            ]["success_rate"]
            mvf_block = data[
                (data["property"] == prop)
                & (data["model_size"] == size)
                & (data["source_name"] == "mvf")
            ]
            if baseline_vals.dropna().empty and mvf_block.empty:
                continue
            mvf_plain = mvf_block["success_rate"]
            mvf_rerank = mvf_block.loc[mvf_block["rerank_success_rate"].notna(), "rerank_success_rate"]
            base_mean = float(baseline_vals.mean()) if not baseline_vals.dropna().empty else np.nan
            mvf_mean = float(mvf_plain.mean()) if not mvf_plain.dropna().empty else np.nan
            mvf_rerank_mean = float(mvf_rerank.mean()) if not mvf_rerank.dropna().empty else mvf_mean
            gain = mvf_rerank_mean - base_mean if np.isfinite(mvf_rerank_mean) and np.isfinite(base_mean) else np.nan
            rows.append(
                {
                    "property": prop,
                    "model_size": size,
                    "baseline_success": base_mean,
                    "mvf_success": mvf_mean,
                    "mvf_rerank_success": mvf_rerank_mean,
                    "rerank_gain": gain,
                }
            )

    comp = pd.DataFrame(rows)
    if comp.empty:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.3), squeeze=False)
    ax0, ax1 = axes[0]

    cmap = plt.get_cmap("tab10")
    for idx, prop in enumerate(prop_order):
        sub = comp[comp["property"] == prop].copy()
        if sub.empty:
            continue
        sub["x"] = sub["model_size"].map(size_to_x)
        sub = sub.dropna(subset=["x"]).sort_values("x")
        if sub.empty:
            continue
        color = cmap(idx % 10)
        x_vals = sub["x"].to_numpy(dtype=float)
        ax0.plot(
            x_vals,
            sub["baseline_success"].to_numpy(dtype=float),
            linestyle="--",
            marker="o",
            linewidth=2.0,
            markersize=6.3,
            color=color,
            alpha=0.65,
            label=f"{prop} baseline",
        )
        ax0.plot(
            x_vals,
            sub["mvf_rerank_success"].to_numpy(dtype=float),
            linestyle="-",
            marker="s",
            linewidth=2.2,
            markersize=6.3,
            color=color,
            alpha=0.95,
            label=f"{prop} MVF rerank",
        )
    ax0.set_xticks(np.arange(len(size_order)))
    ax0.set_xticklabels([s.upper() for s in size_order])
    ax0.set_xlabel("Model Size")
    ax0.set_ylabel("Success Rate")
    ax0.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax0.legend(loc="best", frameon=False)
    _panel_mark(ax0, "(a)", font_size)

    gain_by_size = comp.groupby("model_size", as_index=False)["rerank_gain"].mean(numeric_only=True)
    gain_by_size["x"] = gain_by_size["model_size"].map(size_to_x)
    gain_by_size = gain_by_size.dropna(subset=["x"]).sort_values("x")
    ax1.bar(
        gain_by_size["x"].to_numpy(dtype=float),
        gain_by_size["rerank_gain"].to_numpy(dtype=float),
        color="#59A14F",
        edgecolor="#1F2937",
        linewidth=0.7,
    )
    for idx, prop in enumerate(prop_order):
        sub = comp[comp["property"] == prop].copy()
        if sub.empty:
            continue
        sub["x"] = sub["model_size"].map(size_to_x)
        sub = sub.dropna(subset=["x", "rerank_gain"])
        if sub.empty:
            continue
        ax1.plot(
            sub["x"].to_numpy(dtype=float),
            sub["rerank_gain"].to_numpy(dtype=float),
            linestyle="",
            marker="o",
            markersize=4.8,
            color=cmap(idx % 10),
            alpha=0.8,
        )
    ax1.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.75)
    ax1.set_xticks(np.arange(len(size_order)))
    ax1.set_xticklabels([s.upper() for s in size_order])
    ax1.set_xlabel("Model Size")
    ax1.set_ylabel("MVF Rerank Improvement")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    _panel_mark(ax1, "(b)", font_size)

    fig.tight_layout()
    return _save_plot_figure(fig, output_path, dpi), caption


def _fig6_chem_physics(
    *,
    output_path: Path,
    font_size: int,
    dpi: int,
    df_descriptor: pd.DataFrame,
    df_motif: pd.DataFrame,
    fallback_panels: Sequence[Path],
) -> tuple[bool, str]:
    caption = MANUSCRIPT_CAPTIONS[5]
    if plt is None:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    desc = df_descriptor.copy() if df_descriptor is not None else pd.DataFrame()
    motif = df_motif.copy() if df_motif is not None else pd.DataFrame()

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

    has_desc = prop_col is not None and desc_col is not None and desc_val_col is not None and not desc.empty
    has_motif = motif_prop_col is not None and motif_col is not None and motif_val_col is not None and not motif.empty
    if not has_desc and not has_motif:
        return _render_fallback_panels(output_path=output_path, fallback_panels=fallback_panels, font_size=font_size, dpi=dpi), caption

    layout_properties: list[str] = []
    if has_desc and prop_col in desc.columns:
        layout_properties = _ordered_properties(desc[prop_col].dropna().astype(str).str.strip().tolist())
    max_prop_label_len = max([len(p) for p in layout_properties], default=12)
    prop_count = max(1, len(layout_properties))
    fig_width = max(15.8, 14.0 + 0.10 * max_prop_label_len)
    fig_height = max(6.8, 5.2 + 0.45 * prop_count)
    left_margin = min(0.36, 0.14 + 0.006 * max_prop_label_len)

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), squeeze=False)
    ax0, ax1 = axes[0]

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
            im = ax0.imshow(np.ma.masked_invalid(mat), cmap="coolwarm", aspect="auto")
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

    if has_motif:
        m = motif[[motif_prop_col, motif_col, motif_val_col]].copy()
        m.columns = ["property", "motif", "value"]
        m["motif"] = m["motif"].astype(str).str.strip()
        m["value"] = pd.to_numeric(m["value"], errors="coerce")
        m = m.dropna(subset=["motif", "value"])
        if not m.empty:
            if motif_val_col != "log2_enrichment_topk_vs_ref":
                m["score"] = np.log2(np.clip(m["value"].to_numpy(dtype=float), a_min=1e-8, a_max=None))
            else:
                m["score"] = m["value"].to_numpy(dtype=float)
            top = (
                m.groupby("motif", as_index=False)["score"].mean(numeric_only=True)
                .sort_values("score", ascending=False)
                .head(10)
            )
            ax1.bar(
                np.arange(len(top)),
                top["score"].to_numpy(dtype=float),
                color="#E15759",
                edgecolor="#1F2937",
                linewidth=0.7,
            )
            ax1.set_xticks(np.arange(len(top)))
            ax1.set_xticklabels(top["motif"].tolist(), rotation=35, ha="right")
            ax1.set_ylabel("Log2 Enrichment")
            ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
        else:
            has_motif = False
    if not has_motif:
        ax1.text(0.5, 0.5, "Motif-enrichment data unavailable", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xticks([])
        ax1.set_yticks([])
    _panel_mark(ax1, "(b)", font_size)

    fig.subplots_adjust(left=left_margin, right=0.985, bottom=0.22, top=0.94, wspace=0.34)
    return _save_plot_figure(fig, output_path, dpi), caption


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

    figure_fontsize = _to_int(
        args.figure_fontsize if args.figure_fontsize is not None else paper_cfg.get("figure_fontsize", 16),
        16,
    )
    figure_fontsize = max(8, min(48, figure_fontsize))

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
        f1_status = "completed"
    else:
        missing.append({"category": "table_main", "name": "table_f1_embedding_summary.csv"})
        f1_status = "missing"
    step_rows.append(
        {
            "step_id": "F1",
            "step_name": "alignment_embeddings",
            "status": f1_status,
            "source_metric": "embedding_meta_*.json",
            "paper_table": _safe_rel(f1_table_path, output_dir),
        }
    )

    step_metric_map = [
        ("F2", "retrieval", "step2_retrieval", "metrics_alignment.csv", "table_f2_retrieval.csv"),
        ("F3", "property_heads", "step3_property", "metrics_property.csv", "table_f3_property_heads.csv"),
        ("F4", "ood_analysis", "step4_ood", "metrics_ood.csv", "table_f4_ood_analysis.csv"),
        ("F5", "foundation_inverse", "step5_foundation_inverse", "metrics_inverse.csv", "table_f5_inverse_design.csv"),
        (
            "F6",
            "ood_aware_inverse",
            "step6_ood_aware_inverse",
            "metrics_inverse_ood_objective.csv",
            "table_f6_ood_aware_objective.csv",
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
        include_property_scopes = step_id in {"F5", "F6", "F7"}
        suffix_regex = r"^metrics_inverse_ood_objective_(.+)\.csv$" if step_id == "F6" else None
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
            step_rows.append(
                {
                    "step_id": step_id,
                    "step_name": step_name,
                    "status": "missing",
                    "source_metric": "",
                    "paper_table": _safe_rel(main_dst, output_dir),
                }
            )
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

        step_rows.append(
            {
                "step_id": step_id,
                "step_name": step_name,
                "status": "completed",
                "source_metric": f"{step_subdir}/{filename} across {len(mvf_results_dirs)} runs",
                "paper_table": _safe_rel(main_dst, output_dir),
            }
        )

    step5_dirs: list[Path] = []
    step6_dirs: list[Path] = []
    step7_files_dirs: list[Path] = []
    for mvf_dir in mvf_results_dirs:
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
                step_subdir="step6_ood_aware_inverse",
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
    step5_dirs = _unique_paths(step5_dirs)
    step6_dirs = _unique_paths(step6_dirs)
    step7_files_dirs = _unique_paths(step7_files_dirs)

    properties = _discover_properties(config, step5_dirs, step6_dirs, step_metric_sources.get("F7"))

    f5_artifacts: list[Path] = []
    f6_patterns = ["ood_objective_topk*.csv"]
    if include_large_csv:
        f6_patterns = ["ood_objective_scores*.csv", "ood_objective_topk*.csv"]

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

    def _size_tag_for(path: Path) -> str:
        for mvf_dir in mvf_results_dirs:
            if _is_relative_to(path, mvf_dir):
                return _normalize_model_size(_infer_model_size_from_results_dir(mvf_dir))
        return "unknown"

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

    generated_manuscript_figures: list[str] = []
    generated_si_figures: list[str] = []
    manuscript_caption_lines: list[str] = []
    si_caption_lines: list[str] = []

    if include_figures:
        step_figure_roots: list[Path] = []
        for mvf_dir in mvf_results_dirs:
            step_figure_roots.extend(
                _existing_dirs(
                    [
                        mvf_dir / "step1_alignment_embeddings",
                        mvf_dir / "step2_retrieval",
                        mvf_dir / "step3_property",
                        mvf_dir / "step4_ood",
                        mvf_dir / "step5_foundation_inverse",
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
            df_agg_ood = _load_aggregate_csv(repo_results_root, "metrics_ood.csv")
            df_agg_inverse = _load_aggregate_csv(repo_results_root, "metrics_inverse.csv")

            df_mvf_alignment = _load_mvf_csv_multi(mvf_results_dirs, "step2_retrieval", "metrics_alignment.csv")
            df_mvf_property = _load_mvf_csv_multi(
                mvf_results_dirs,
                "step3_property",
                "metrics_property.csv",
                include_property_scopes=True,
            )
            df_mvf_ood = _load_mvf_csv_multi(mvf_results_dirs, "step4_ood", "metrics_ood.csv")
            df_mvf_inverse = _load_mvf_csv_multi(
                mvf_results_dirs,
                "step5_foundation_inverse",
                "metrics_inverse.csv",
                include_property_scopes=True,
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

            manuscript_specs = [
                {
                    "themes": ["base_step0", "base_step12"],
                    "fn": _fig1_baseline_generation,
                    "kwargs": {
                        "df_generation": df_agg_generation,
                        "df_step12": step12_scale_summary_df,
                        "primary_panels": figure1_primary_panels,
                    },
                },
                {
                    "themes": ["mvf_f1", "mvf_f2"],
                    "fn": _fig2_mvf_alignment,
                    "kwargs": {
                        "df_alignment": df_mvf_alignment if not df_mvf_alignment.empty else df_agg_alignment,
                    },
                },
                {
                    "themes": ["mvf_f3"],
                    "fn": _fig3_property_prediction,
                    "kwargs": {
                        "df_property_base": df_agg_property,
                        "df_property_mvf": df_mvf_property,
                    },
                },
                {
                    "themes": ["mvf_f4"],
                    "fn": _fig4_ood_analysis,
                    "kwargs": {
                        "df_ood_base": df_agg_ood,
                        "df_ood_mvf": df_mvf_ood,
                    },
                },
                {
                    "themes": ["mvf_f5", "mvf_f6"],
                    "fn": _fig5_inverse_design,
                    "kwargs": {
                        "df_inverse_base": df_agg_inverse,
                        "df_inverse_mvf": df_mvf_inverse,
                    },
                },
                {
                    "themes": ["mvf_f7", "misc"],
                    "fn": _fig6_chem_physics,
                    "kwargs": {
                        "df_descriptor": df_mvf_desc,
                        "df_motif": df_mvf_motif,
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
                    fallback_order=FIGURE_FALLBACK_ORDER,
                )
                for panel in fallback_panels:
                    manuscript_used_keys.add(str(panel.resolve()))

                fig_path = manuscript_figures_dir / f"Figure_{figure_num}.png"
                ok, caption_text = spec["fn"](
                    output_path=fig_path,
                    font_size=figure_fontsize,
                    dpi=figure_dpi,
                    fallback_panels=fallback_panels,
                    **spec["kwargs"],
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
                },
                {
                    "themes": ["base_step12"],
                },
                {
                    "themes": ["mvf_f1"],
                },
                {
                    "themes": ["mvf_f2"],
                },
                {
                    "themes": ["mvf_f3"],
                },
                {
                    "themes": ["mvf_f4"],
                },
                {
                    "themes": ["mvf_f5"],
                },
                {
                    "themes": ["mvf_f6"],
                },
                {
                    "themes": ["mvf_f7", "misc"],
                },
            ]
            si_specs = si_specs[:SUPPORTING_INFORMATION_FIGURE_COUNT]
            if not all_panels:
                si_specs = [
                    {
                        "themes": ["misc"],
                    }
                ]

            si_used_keys: set[str] = set()
            si_reuse_cursor = 0
            for si_idx, spec in enumerate(si_specs, start=1):
                pre_used_keys = set(si_used_keys)
                chunk = _pick_panels(
                    themed=themed_panels,
                    theme_group=spec["themes"],
                    used=si_used_keys,
                    max_panels=max_panels_per_figure,
                    allow_reuse=True,
                    fallback_order=FIGURE_FALLBACK_ORDER,
                )
                if all_panels and (
                    not chunk
                    or all(str(panel.resolve()) in pre_used_keys for panel in chunk)
                ):
                    chunk = _rolling_panel_window(all_panels, si_reuse_cursor, max_panels_per_figure)
                    si_reuse_cursor += max_panels_per_figure
                for panel in chunk:
                    si_used_keys.add(str(panel.resolve()))

                fig_path = si_figures_dir / f"Figure_S{si_idx}.png"
                ok = _compose_multi_panel_figure(
                    panel_paths=chunk,
                    output_path=fig_path,
                    font_size=figure_fontsize,
                    dpi=figure_dpi,
                )
                if ok:
                    generated_si_figures.append(_safe_rel(fig_path, output_dir))
                    copied.append(
                        {
                            "category": "figure_supporting_information",
                            "source": "generated_composite",
                            "destination": _safe_rel(fig_path, output_dir),
                        }
                    )
                if si_idx <= len(SI_CAPTIONS):
                    si_caption_lines.append(SI_CAPTIONS[si_idx - 1])
                else:
                    si_caption_lines.append(f"Figure S{si_idx}. Supporting-information diagnostic panel.")

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
