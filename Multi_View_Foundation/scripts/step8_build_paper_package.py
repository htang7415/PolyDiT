#!/usr/bin/env python
"""F8: Build a paper-ready package from F1-F7 outputs and 5-method baselines."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

import pandas as pd

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
    seen_names: set[str] = set()
    for directory in source_dirs:
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                if path.name in seen_names:
                    continue
                seen_names.add(path.name)
                collected.append(path)
    return collected


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
    results_dir: Path,
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
            fig_dir = res_dir / "step0_data_prep" / "figures"
            if not fig_dir.exists():
                continue
            for png in sorted(fig_dir.glob("*.png")):
                add("base_step0", png)

    aggregate_root = REPO_ROOT / "results"
    if aggregate_root.exists():
        for png in sorted(aggregate_root.glob("aggregate*/figures/*.png")):
            add("base_step12", png)

    for png in sorted(results_dir.rglob("*.png")):
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


def _build_caption_text(
    *,
    figure_label: str,
    panel_paths: Sequence[Path],
    summary: str = "",
) -> str:
    sentences = []
    for idx, panel in enumerate(panel_paths):
        sentences.append(f"{_panel_label(idx)} {_describe_panel(panel)}.")
    if not sentences:
        sentences.append("(a) No source panel was available.")
    out = f"{figure_label}. "
    summary_text = str(summary).strip()
    if summary_text:
        if not summary_text.endswith("."):
            summary_text += "."
        out += summary_text + " "
    out += " ".join(sentences)
    return out


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

    # F1 summary table from embedding meta.
    f1_df = _f1_embedding_summary(results_dir)
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
        (
            "F2",
            "retrieval",
            [
                results_dir / "step2_retrieval" / "metrics" / "metrics_alignment.csv",
                results_dir / "metrics_alignment.csv",
                results_dir / "step2_retrieval" / "metrics_alignment.csv",
            ],
            "table_f2_retrieval.csv",
        ),
        (
            "F3",
            "property_heads",
            [
                results_dir / "step3_property" / "metrics" / "metrics_property.csv",
                results_dir / "metrics_property.csv",
                results_dir / "step3_property" / "metrics_property.csv",
            ],
            "table_f3_property_heads.csv",
        ),
        (
            "F4",
            "ood_analysis",
            [
                results_dir / "step4_ood" / "metrics" / "metrics_ood.csv",
                results_dir / "metrics_ood.csv",
                results_dir / "step4_ood" / "metrics_ood.csv",
            ],
            "table_f4_ood_analysis.csv",
        ),
        (
            "F5",
            "foundation_inverse",
            [
                results_dir / "step5_foundation_inverse" / "metrics" / "metrics_inverse.csv",
                results_dir / "metrics_inverse.csv",
                results_dir / "step5_foundation_inverse" / "metrics_inverse.csv",
            ],
            "table_f5_inverse_design.csv",
        ),
        (
            "F6",
            "ood_aware_inverse",
            [
                results_dir / "step6_ood_aware_inverse" / "metrics" / "metrics_inverse_ood_objective.csv",
                results_dir / "metrics_inverse_ood_objective.csv",
                results_dir / "step6_ood_aware_inverse" / "metrics_inverse_ood_objective.csv",
            ],
            "table_f6_ood_aware_objective.csv",
        ),
        (
            "F7",
            "chem_physics_analysis",
            [
                results_dir / "step7_chem_physics_analysis" / "metrics" / "metrics_chem_physics.csv",
                results_dir / "metrics_chem_physics.csv",
                results_dir / "step7_chem_physics_analysis" / "metrics_chem_physics.csv",
            ],
            "table_f7_chem_physics.csv",
        ),
    ]

    step_metric_sources: dict[str, Optional[Path]] = {}
    for step_id, step_name, candidates, out_name in step_metric_map:
        main_dst = tables_main_dir / out_name
        src = _copy_first(
            candidates,
            main_dst,
            category="table_main",
            copied=copied,
            missing=missing,
            results_dir=results_dir,
            output_dir=output_dir,
            missing_label=out_name,
        )
        step_metric_sources[step_id] = src
        if src is not None:
            _copy_file(
                src,
                manuscript_results_dir / out_name,
                category="table_main",
                copied=copied,
                results_dir=results_dir,
                output_dir=output_dir,
            )
        step_rows.append(
            {
                "step_id": step_id,
                "step_name": step_name,
                "status": "completed" if src is not None else "missing",
                "source_metric": _safe_rel(src, results_dir) if src is not None else "",
                "paper_table": _safe_rel(main_dst, output_dir),
            }
        )

    step5_dirs = _existing_dirs(
        [
            results_dir / "step5_foundation_inverse" / "files",
            results_dir / "step5_foundation_inverse",
        ]
    )
    step6_dirs = _existing_dirs(
        [
            results_dir / "step6_ood_aware_inverse" / "files",
            results_dir / "step6_ood_aware_inverse",
        ]
    )
    step7_files_dirs = _existing_dirs(
        [
            results_dir / "step7_chem_physics_analysis" / "files",
            results_dir / "step7_chem_physics_analysis",
        ]
    )

    properties = _discover_properties(config, step5_dirs, step6_dirs, step_metric_sources.get("F7"))

    f5_patterns = ["accepted_candidates*.csv"]
    f6_patterns = ["ood_objective_topk*.csv"]
    if include_large_csv:
        f5_patterns = ["candidate_scores*.csv", "accepted_candidates*.csv"]
        f6_patterns = ["ood_objective_scores*.csv", "ood_objective_topk*.csv"]

    for src in _collect_glob_unique(step5_dirs, f5_patterns):
        rel_dst = Path(src.name)
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
        rel_dst = Path(src.name)
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
        rel_dst = Path(src.name)
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

    generated_manuscript_figures: list[str] = []
    generated_si_figures: list[str] = []
    manuscript_caption_lines: list[str] = []
    si_caption_lines: list[str] = []

    if include_figures:
        step_figure_roots = _existing_dirs(
            [
                results_dir / "step1_alignment_embeddings",
                results_dir / "step2_retrieval",
                results_dir / "step3_property",
                results_dir / "step4_ood",
                results_dir / "step5_foundation_inverse",
                results_dir / "step6_ood_aware_inverse",
                results_dir / "step7_chem_physics_analysis",
            ]
        )

        seen_mvf_fig_names: set[str] = set()
        for root in step_figure_roots:
            for src in sorted(root.rglob("figure_f*.png")):
                if _is_relative_to(src, output_dir):
                    continue
                name = src.name
                if name in seen_mvf_fig_names:
                    name = f"{src.parent.parent.name}_{name}"
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
            themed_panels = _collect_source_panels(
                results_dir=results_dir,
                output_dir=output_dir,
                method_dirs=method_dirs,
            )

            manuscript_specs = [
                {
                    "themes": ["base_step0"],
                    "summary": "Representation-level preprocessing diagnostics for the five baseline methods",
                },
                {
                    "themes": ["base_step12"],
                    "summary": "Baseline Step1-Step2 performance trade-offs across methods and model scales",
                },
                {
                    "themes": ["mvf_f1", "mvf_f2"],
                    "summary": "Multi-view foundation alignment and cross-view retrieval evidence",
                },
                {
                    "themes": ["mvf_f3", "mvf_f4"],
                    "summary": "Property prediction and distribution-shift diagnostics before inverse design",
                },
                {
                    "themes": ["mvf_f5", "mvf_f6"],
                    "summary": "Foundation-guided inverse design and OOD-aware reranking outcomes",
                },
                {
                    "themes": ["mvf_f7", "misc"],
                    "summary": "Chemistry/physics interpretation and cross-property evidence for selected candidates",
                },
            ]
            manuscript_specs = manuscript_specs[:manuscript_figure_count]

            manuscript_used_keys: set[str] = set()
            for figure_num, spec in enumerate(manuscript_specs, start=1):
                panels = _pick_panels(
                    themed=themed_panels,
                    theme_group=spec["themes"],
                    used=manuscript_used_keys,
                    max_panels=max_panels_per_figure,
                    allow_reuse=False,
                    fallback_order=FIGURE_FALLBACK_ORDER,
                )
                for panel in panels:
                    manuscript_used_keys.add(str(panel.resolve()))

                fig_path = manuscript_figures_dir / f"Figure_{figure_num}.png"
                ok = _compose_multi_panel_figure(
                    panel_paths=panels,
                    output_path=fig_path,
                    font_size=figure_fontsize,
                    dpi=figure_dpi,
                )
                if ok:
                    generated_manuscript_figures.append(_safe_rel(fig_path, output_dir))
                    copied.append(
                        {
                            "category": "figure_manuscript",
                            "source": "generated_composite",
                            "destination": _safe_rel(fig_path, output_dir),
                        }
                    )
                else:
                    missing.append(
                        {
                            "category": "figure_manuscript",
                            "name": str(fig_path.name),
                        }
                    )

                manuscript_caption_lines.append(
                    _build_caption_text(
                        figure_label=f"Figure {figure_num}",
                        panel_paths=panels,
                        summary=str(spec.get("summary", "")),
                    )
                )

            all_panels = _flatten_themed_panels(themed_panels, FIGURE_FALLBACK_ORDER)

            # Always keep traceable source panels in SI.
            source_panels_dir = si_figures_dir / "source_panels"
            source_panels_dir.mkdir(parents=True, exist_ok=True)
            for panel in all_panels:
                rel = _safe_rel(panel, REPO_ROOT).replace("/", "__")
                dst = source_panels_dir / rel
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
                    "summary": "Expanded Step0 diagnostics across all representation pipelines",
                },
                {
                    "themes": ["base_step12"],
                    "summary": "Expanded baseline aggregate panels for Step1-Step2 rankings and trade-offs",
                },
                {
                    "themes": ["mvf_f1"],
                    "summary": "Detailed F1 embedding extraction and alignment summaries",
                },
                {
                    "themes": ["mvf_f2"],
                    "summary": "Detailed F2 retrieval performance across view pairs and recall metrics",
                },
                {
                    "themes": ["mvf_f3"],
                    "summary": "Detailed F3 property-head benchmarking, coverage, and generalization",
                },
                {
                    "themes": ["mvf_f4"],
                    "summary": "Detailed F4 OOD manifold-distance and neighbor diagnostics",
                },
                {
                    "themes": ["mvf_f5"],
                    "summary": "Detailed F5 inverse-generation diagnostics and accepted-candidate profiles",
                },
                {
                    "themes": ["mvf_f6"],
                    "summary": "Detailed F6 conservative objective term diagnostics for top-k reranking",
                },
                {
                    "themes": ["mvf_f7", "misc"],
                    "summary": "Detailed F7 chemistry/physics, motif enrichment, and nearest-neighbor evidence",
                },
            ]
            si_specs = si_specs[:SUPPORTING_INFORMATION_FIGURE_COUNT]
            if not all_panels:
                si_specs = [
                    {
                        "themes": ["misc"],
                        "summary": "Supporting-information placeholder because no source figures were available",
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
                si_caption_lines.append(
                    _build_caption_text(
                        figure_label=f"Figure S{si_idx}",
                        panel_paths=chunk,
                        summary=str(spec.get("summary", "")),
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
        _copy_file(
            src,
            run_meta_dir / src.name,
            category="manifest",
            copied=copied,
            results_dir=results_dir,
            output_dir=output_dir,
        )

    manifest_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results_dir": str(results_dir),
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
        "generated_manuscript_figures": generated_manuscript_figures,
        "generated_supporting_information_figures": generated_si_figures,
        "steps": step_rows,
        "copied_artifacts": copied,
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
