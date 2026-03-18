#!/usr/bin/env python
"""F5: Foundation inverse design benchmark with per-view comparison outputs."""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.data.view_converters import smiles_to_selfies
from shared.unlabeled_data import require_preprocessed_unlabeled_splits
from src.analysis.view_compare import (
    analyze_view_compare,
    plot_view_compare,
    save_view_compare_outputs,
)
from src.utils.foundation_assets import (
    SUPPORTED_VIEWS,
    VIEW_SPECS,
    TorchPropertyPredictor as _TorchPropertyPredictor,
    default_property_model_path as _default_property_model_path,
    load_property_model as _load_property_model,
    load_view_assets as _load_view_assets,
    resolve_view_device as _resolve_view_device,
)
from src.utils.output_layout import ensure_step_dirs, save_csv, save_json
from src.utils.runtime import (
    load_module as _shared_load_module,
    resolve_path as _shared_resolve_path,
    to_bool as _to_bool,
    to_int_or_none as _to_int_or_none,
)
from src.utils.polymer_patterns import POLYMER_CLASS_PATTERNS
from src.utils.property_names import (
    normalize_property_name as shared_normalize_property_name,
    property_display_name,
)
from src.utils.visualization import (
    PUBLICATION_STYLE,
    ordered_views,
    save_figure_png,
    set_publication_style,
    standardize_figure_text_and_legend,
    view_color,
    view_label,
)

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import RDConfig
    # Keep F5 logs readable when many invalid candidates are rejected.
    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")
except Exception:  # pragma: no cover
    Chem = None
    RDConfig = None
    rdBase = None

try:
    import os
    if RDConfig is not None:
        sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_dir not in sys.path:
            sys.path.append(sa_dir)
    import sascorer  # type: ignore
except Exception:  # pragma: no cover
    sascorer = None

DEFAULT_F5_RESAMPLE_SETTINGS = {
    "property_model_mode": "single",  # single|all
    "proposal_views": "all",  # all|<comma-separated views>
    "committee_properties": None,  # defaults to property.files
    "sampling_mode": "resample",  # fixed_budget|resample
    "candidate_budget_per_view": 10000,
    "sampling_target": 100,
    "sampling_num_per_batch": 100,
    "sampling_batch_size": 100,
    "sampling_max_batches": 10000,
    "top_k": 100,
    "sampling_temperature": None,
    "sampling_num_atoms": None,
    "target_class": "",
    "require_validity": True,
    "require_two_stars": True,
    "require_novel": True,
    "require_unique": True,
    "max_sa": 3.6,
    "per_view_min_hits": 0,
    "per_view_quota_relax_after_batches": 0,
}

if plt is not None:
    set_publication_style()


def _resolve_path(path_str: str) -> Path:
    return _shared_resolve_path(path_str, BASE_DIR)


def _view_to_representation(view: str) -> str:
    mapping = {
        "smiles": "SMILES",
        "smiles_bpe": "SMILES_BPE",
        "selfies": "SELFIES",
        "group_selfies": "Group_SELFIES",
        "graph": "Graph",
    }
    return mapping.get(view, view)


def _save_figure_png(fig, output_base: Path) -> None:
    save_figure_png(fig, output_base, font_size=16, dpi=600, legend_loc="best")


def _plot_f5_diagnostics(
    *,
    scored_df: pd.DataFrame,
    property_name: str,
    target_value: float,
    target_mode: str,
    epsilon: float,
    source_meta: dict,
    figures_dir: Path,
) -> None:
    if plt is None or scored_df.empty:
        return

    property_label = property_display_name(property_name)
    df = scored_df.copy()
    mode = str(target_mode).strip().lower()
    accepted_mask = pd.to_numeric(df.get("accepted", pd.Series([0] * len(df), index=df.index)), errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.reshape(-1)

    # A) score distribution.
    if mode in {"ge", "le"}:
        if "target_excess" in df.columns:
            excess_all = pd.to_numeric(df["target_excess"], errors="coerce").dropna()
            excess_acc = pd.to_numeric(df.loc[accepted_mask, "target_excess"], errors="coerce").dropna()
        else:
            pred_vals = pd.to_numeric(df.get("prediction", pd.Series(dtype=float)), errors="coerce")
            pred_acc_vals = pd.to_numeric(df.loc[accepted_mask, "prediction"] if "prediction" in df.columns else pd.Series(dtype=float), errors="coerce")
            excess_all = pd.Series(_compute_target_excess(pred_vals.to_numpy(dtype=np.float32), float(target_value), mode)).dropna()
            excess_acc = pd.Series(_compute_target_excess(pred_acc_vals.to_numpy(dtype=np.float32), float(target_value), mode)).dropna()
        if len(excess_all):
            ax0.hist(excess_all.to_numpy(dtype=np.float32), bins=40, color="#4E79A7", alpha=0.65, label="Scored")
        if len(excess_acc):
            ax0.hist(excess_acc.to_numpy(dtype=np.float32), bins=30, color="#E15759", alpha=0.75, label="Accepted")
        ax0.axvline(0.0, color="#222222", linestyle="--", linewidth=1.1, label="Target boundary")
        ax0.set_xlabel(_target_excess_axis_label(property_name, mode))
    else:
        pred = pd.to_numeric(df.get("prediction", pd.Series(dtype=float)), errors="coerce").dropna()
        pred_accepted = pd.to_numeric(df.loc[accepted_mask, "prediction"] if "prediction" in df.columns else pd.Series(dtype=float), errors="coerce").dropna()
        if len(pred):
            ax0.hist(pred.to_numpy(dtype=np.float32), bins=40, color="#4E79A7", alpha=0.65, label="Scored")
        if len(pred_accepted):
            ax0.hist(pred_accepted.to_numpy(dtype=np.float32), bins=30, color="#E15759", alpha=0.75, label="Accepted")
        ax0.axvline(float(target_value), color="#222222", linestyle="--", linewidth=1.1, label="Target")
        ax0.axvspan(float(target_value) - float(epsilon), float(target_value) + float(epsilon), color="#59A14F", alpha=0.15)
        ax0.set_xlabel(f"Predicted {property_label}")
    ax0.set_ylabel("Count")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="best", fontsize=15)

    # B) prediction vs OOD distance.
    ood_col = "ood_prop" if "ood_prop" in df.columns else "d2_distance"
    ood_axis_label = "OOD prop (cosine dist to D2)" if ood_col == "ood_prop" else "D2 distance"
    if {"prediction", ood_col}.issubset(set(df.columns)):
        scatter_df = df.dropna(subset=["prediction", ood_col]).copy()
        if not scatter_df.empty:
            if mode in {"ge", "le"}:
                if "target_excess" in scatter_df.columns:
                    y_vals = pd.to_numeric(scatter_df["target_excess"], errors="coerce")
                else:
                    y_vals = pd.Series(
                        _compute_target_excess(
                            pd.to_numeric(scatter_df["prediction"], errors="coerce").to_numpy(dtype=np.float32),
                            float(target_value),
                            mode,
                        ),
                        index=scatter_df.index,
                    )
                mask = y_vals.notna()
                scatter_df = scatter_df.loc[mask].copy()
                y_vals = y_vals.loc[mask]
                if not scatter_df.empty:
                    sc = ax1.scatter(
                        scatter_df[ood_col],
                        y_vals,
                        c=y_vals,
                        cmap="RdYlGn",
                        s=24,
                        alpha=0.82,
                    )
                    ax1.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
                    ax1.set_xlabel(ood_axis_label)
                    ax1.set_ylabel("Target excess (≥0 = hit)")
                    ax1.grid(alpha=0.25)
                    fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="Target excess")
                else:
                    ax1.text(0.5, 0.5, "No finite points", ha="center", va="center")
                    ax1.set_axis_off()
            else:
                hit_vals = scatter_df.get("property_hit", pd.Series([False] * len(scatter_df), index=scatter_df.index)).astype(bool)
                base = scatter_df[~hit_vals]
                hit = scatter_df[hit_vals]
                if not base.empty:
                    ax1.scatter(base[ood_col], base["prediction"], s=16, alpha=0.35, color="#9ECAE1", label="Non-hit")
                if not hit.empty:
                    ax1.scatter(hit[ood_col], hit["prediction"], s=24, alpha=0.9, color="#D94801", label="Hit")
                ax1.axhline(float(target_value), color="#222222", linestyle="--", linewidth=1.0)
                ax1.set_xlabel(ood_axis_label)
                ax1.set_ylabel(f"Predicted {property_name}")
                ax1.grid(alpha=0.25)
                ax1.legend(loc="best", fontsize=15)
        else:
            ax1.text(0.5, 0.5, "No finite points", ha="center", va="center")
            ax1.set_axis_off()
    else:
        ax1.text(0.5, 0.5, f"No {ood_col} column", ha="center", va="center")
        ax1.set_axis_off()

    # C) Sampling funnel: generated → structural valid → scored → accepted (property hit).
    # If multiple views, show per-view breakdown; otherwise show aggregate funnel.
    stats_dict = source_meta.get("stats", {})
    funnel_drawn = False
    if "proposal_view" in df.columns:
        all_proposal_views = sorted(df["proposal_view"].dropna().astype(str).unique().tolist())
        if len(all_proposal_views) > 1:
            # Multi-view: grouped bar showing generated / scored / accepted per view
            x_labels = all_proposal_views
            n_gen = np.array([float(stats_dict.get(f"n_generated_{v}", 0)) for v in all_proposal_views], dtype=np.float32)
            n_scored = np.array([float(stats_dict.get(f"n_scored_{v}", 0)) for v in all_proposal_views], dtype=np.float32)
            n_accepted = np.array([float(df.loc[accepted_mask & (df["proposal_view"] == v)].shape[0]) for v in all_proposal_views], dtype=np.float32)
            x = np.arange(len(x_labels), dtype=np.float32)
            w = 0.28
            ax2.bar(x - w, n_gen, width=w, label="Generated", color="#AEC7E8", alpha=0.85)
            ax2.bar(x, n_scored, width=w, label="Scored", color="#4E79A7", alpha=0.85)
            ax2.bar(x + w, n_accepted, width=w, label="Accepted", color="#59A14F", alpha=0.85)
            ax2.set_xticks(x)
            ax2.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=12)
            ax2.set_ylabel("Count")
            ax2.legend(loc="best", fontsize=11)
            ax2.grid(axis="y", alpha=0.25)
            funnel_drawn = True
    if not funnel_drawn:
        # Single/no view: show aggregate funnel as horizontal bar chart
        n_gen_total = float(stats_dict.get("n_generated", len(df)))
        n_valid = float(stats_dict.get("n_structural_valid", stats_dict.get("n_valid_any", 0)))
        n_scored_total = float(stats_dict.get("n_scored", len(df)))
        n_accepted_total = int(np.sum(accepted_mask))
        funnel_labels = ["Generated", "Struct. valid", "Scored", "Accepted"]
        funnel_vals = np.array([n_gen_total, n_valid, n_scored_total, float(n_accepted_total)], dtype=np.float32)
        colors = ["#AEC7E8", "#6BAED6", "#4E79A7", "#59A14F"]
        y_pos = np.arange(len(funnel_labels), dtype=np.float32)
        bars = ax2.barh(y_pos, funnel_vals, color=colors, alpha=0.9)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(funnel_labels, fontsize=12)
        ax2.set_xlabel("Count")
        ax2.set_title("Sampling funnel", fontsize=12)
        ax2.grid(axis="x", alpha=0.25)
        for bar, val in zip(bars, funnel_vals):
            if val > 0:
                ax2.text(float(val), bar.get_y() + bar.get_height() / 2.0,
                         f"  {int(val):,}", va="center", ha="left", fontsize=11)

    # D) cumulative hits by ranking.
    hit_col = df.get("property_hit", pd.Series([False] * len(df))).astype(bool).to_numpy(dtype=bool)
    if len(df):
        if mode in {"ge", "le"}:
            if "target_excess" in df.columns:
                excess = pd.to_numeric(df["target_excess"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float32)
            else:
                pred_vals = pd.to_numeric(df.get("prediction", pd.Series(dtype=float)), errors="coerce").fillna(np.nan).to_numpy(dtype=np.float32)
                excess = _compute_target_excess(pred_vals, float(target_value), mode)
                excess = np.nan_to_num(excess, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
            order = np.argsort(-excess)
        else:
            if "target_violation" in df.columns:
                order_vals = pd.to_numeric(df["target_violation"], errors="coerce").fillna(np.inf).to_numpy(dtype=np.float32)
            else:
                order_vals = pd.to_numeric(df.get("abs_error", pd.Series(dtype=float)), errors="coerce").fillna(np.inf).to_numpy(dtype=np.float32)
            order = np.argsort(order_vals)
        ordered_hits = hit_col[order].astype(np.int64, copy=False)
        cum_hits = np.cumsum(ordered_hits)
        x = np.arange(1, len(cum_hits) + 1, dtype=np.int64)
        ax3.plot(x, cum_hits, color="#4E79A7", linewidth=2.0)
        ax3.set_xlabel("Ranked candidate index")
        ax3.set_ylabel("Cumulative property hits")
        ax3.grid(alpha=0.25)
    else:
        ax3.text(0.5, 0.5, "No scored rows", ha="center", va="center")
        ax3.set_axis_off()

    # E) OOD-gen vs OOD-prop scatter: reveals generative vs property-relevance trade-off.
    if "ood_prop" in df.columns and "ood_gen" in df.columns:
        ood_prop_vals = pd.to_numeric(df["ood_prop"], errors="coerce")
        ood_gen_vals = pd.to_numeric(df["ood_gen"], errors="coerce")
        mask_e = ood_prop_vals.notna() & ood_gen_vals.notna()
        if mask_e.any():
            hit_col_e = df.get("property_hit", pd.Series([False] * len(df))).astype(bool)
            non_hit = df[mask_e & ~hit_col_e]
            hit = df[mask_e & hit_col_e]
            if not non_hit.empty:
                ax4.scatter(non_hit["ood_prop"], non_hit["ood_gen"],
                            s=12, alpha=0.3, color="#9ECAE1", label="Non-hit")
            if not hit.empty:
                ax4.scatter(hit["ood_prop"], hit["ood_gen"],
                            s=28, alpha=0.85, color="#D94801", label="Property hit", zorder=5)
            ax4.set_xlabel("OOD-prop (cosine dist to D2, property relevance)")
            ax4.set_ylabel("OOD-gen (cosine dist to D1, generative reliability)")
            ax4.set_title("(E) OOD-prop vs OOD-gen signal")
            ax4.grid(alpha=0.25)
            ax4.legend()
            # Annotate best-region quadrant
            xlim = ax4.get_xlim()
            ylim = ax4.get_ylim()
            ax4.text(xlim[0] + 0.02 * (xlim[1] - xlim[0]), ylim[1] - 0.02 * (ylim[1] - ylim[0]),
                     "Low OOD-prop\nLow OOD-gen\n(best region)", fontsize=9, va="top",
                     color="#2A9D8F", style="italic")
        else:
            ax4.text(0.5, 0.5, "No finite ood_prop/ood_gen data", ha="center", va="center")
            ax4.set_axis_off()
    else:
        ax4.text(0.5, 0.5, "ood_prop/ood_gen columns\nnot available", ha="center", va="center")
        ax4.set_axis_off()

    # F) Prediction uncertainty distribution (ensemble std).
    if "prediction_std" in df.columns:
        std_vals = pd.to_numeric(df["prediction_std"], errors="coerce").dropna()
        std_acc = pd.to_numeric(df.loc[accepted_mask, "prediction_std"], errors="coerce").dropna() if accepted_mask.any() else pd.Series(dtype=float)
        if std_vals.size:
            ax5.hist(std_vals.to_numpy(dtype=np.float32), bins=40, color="#B07AA1",
                     alpha=0.65, label=f"All scored (n={std_vals.size:,})")
        if std_acc.size:
            ax5.hist(std_acc.to_numpy(dtype=np.float32), bins=25, color="#F28E2B",
                     alpha=0.8, label=f"Accepted (n={std_acc.size:,})")
        ax5.set_xlabel(f"Prediction std (ensemble uncertainty for {property_name})")
        ax5.set_ylabel("Count")
        ax5.set_title("(F) Predictive uncertainty distribution")
        ax5.grid(alpha=0.25)
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "prediction_std not available\n(requires property_model_mode='all')",
                 ha="center", va="center", style="italic")
        ax5.set_axis_off()

    fig.suptitle(f"F5 Inverse Design Diagnostics: {property_label}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save_figure_png(fig, figures_dir / f"figure_f5_inverse_diagnostics_{property_name}")
    plt.close(fig)

    # Save compact accepted-by-view table for downstream inspection.
    accepted_rows = []
    if "proposal_view" in df.columns:
        grp = df.loc[accepted_mask].groupby("proposal_view").size().sort_values(ascending=False)
        accepted_rows.extend(
            {"proposal_view": str(k), "accepted_count": int(v)}
            for k, v in grp.items()
        )
    if not accepted_rows:
        accepted_by_view = source_meta.get("accepted_by_view", {})
        if isinstance(accepted_by_view, dict):
            accepted_rows.extend(
                {"proposal_view": str(k), "accepted_count": int(v)}
                for k, v in accepted_by_view.items()
            )
    if accepted_rows:
        save_csv(
            pd.DataFrame(accepted_rows),
            figures_dir / f"figure_f5_inverse_diagnostics_{property_name}_accepted_by_view.csv",
            index=False,
        )


def _rdkit_mol_from_polymer_smiles(smiles: str):
    if Chem is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    try:
        mol = Chem.MolFromSmiles(text.replace("*", "[H]"))
        if mol is not None:
            return mol
    except Exception:
        pass
    try:
        return Chem.MolFromSmiles(text)
    except Exception:
        return None


def _count_rings(smiles: str) -> Optional[int]:
    mol = _rdkit_mol_from_polymer_smiles(smiles)
    if mol is None:
        return None
    try:
        return int(mol.GetRingInfo().NumRings())
    except Exception:
        return None


def _count_heavy_atoms(smiles: str) -> Optional[int]:
    mol = _rdkit_mol_from_polymer_smiles(smiles)
    if mol is None:
        return None
    try:
        return int(mol.GetNumHeavyAtoms())
    except Exception:
        return None


def _short_text(text: Any, max_len: int = 44) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 3)] + "..."


def _build_f5_accepted_polymer_report(
    *,
    accepted_df: pd.DataFrame,
    property_name: str,
    target_value: float,
    target_mode: str,
    epsilon: float,
    sampling_target: int,
    sampling_target_total: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    mode = str(target_mode).strip().lower()
    target_total = int(sampling_target_total) if sampling_target_total is not None else int(sampling_target)
    report = accepted_df.copy()
    if report.empty:
        summary = {
            "property": property_name,
            "target_value": float(target_value),
            "target_mode": mode,
            "epsilon": float(epsilon),
            "sampling_target": int(sampling_target),
            "sampling_target_total": int(target_total),
            "n_accepted": 0,
            "acceptance_ratio_vs_target": 0.0,
        }
        return report, summary

    report["prediction"] = pd.to_numeric(report.get("prediction"), errors="coerce")
    report["target_excess"] = pd.to_numeric(report.get("target_excess"), errors="coerce")
    report["target_violation"] = pd.to_numeric(report.get("target_violation"), errors="coerce")
    report["ood_prop"] = pd.to_numeric(report.get("ood_prop"), errors="coerce")
    report["ood_gen"] = pd.to_numeric(report.get("ood_gen"), errors="coerce")
    report["d2_distance"] = pd.to_numeric(report.get("d2_distance"), errors="coerce")
    report["sa_score"] = pd.to_numeric(report.get("sa_score"), errors="coerce")
    report["prediction_std"] = pd.to_numeric(report.get("prediction_std"), errors="coerce")

    # Ranking policy:
    # - ge/le: maximize target_excess
    # - window: minimize target_violation
    if mode in {"ge", "le"}:
        primary = -report["target_excess"].fillna(-np.inf)
        rank_score = report["target_excess"].fillna(np.nan)
    else:
        fallback_violation = pd.to_numeric(report.get("abs_error"), errors="coerce")
        violation = report["target_violation"].where(report["target_violation"].notna(), fallback_violation)
        primary = violation.fillna(np.inf)
        rank_score = -violation.fillna(np.nan)

    report["_rank_primary"] = primary
    # Prefer ood_prop for secondary ranking (cosine dist to D2); fall back to d2_distance
    _ood_col_rank = "ood_prop" if "ood_prop" in report.columns else "d2_distance"
    report["_rank_secondary"] = report[_ood_col_rank].fillna(np.inf)
    report["_rank_tertiary"] = report["prediction_std"].fillna(np.inf)
    report["_rank_score"] = rank_score
    report = report.sort_values(
        ["_rank_primary", "_rank_secondary", "_rank_tertiary"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    report["rank"] = np.arange(1, len(report) + 1, dtype=np.int64)
    report["rank_score"] = pd.to_numeric(report.get("_rank_score"), errors="coerce")

    if "canonical_smiles" not in report.columns:
        report["canonical_smiles"] = report["smiles"].astype(str).map(_canonicalize_smiles)
    else:
        report["canonical_smiles"] = report["canonical_smiles"].astype(str)

    if "star_count" not in report.columns:
        report["star_count"] = report["smiles"].astype(str).map(lambda x: int(str(x).count("*")))
    else:
        report["star_count"] = pd.to_numeric(report["star_count"], errors="coerce")

    if "aromatic_ring_count" not in report.columns:
        report["aromatic_ring_count"] = report["smiles"].astype(str).map(_count_aromatic_rings)
    if "ring_count" not in report.columns:
        report["ring_count"] = report["smiles"].astype(str).map(_count_rings)
    if "heavy_atom_count" not in report.columns:
        report["heavy_atom_count"] = report["smiles"].astype(str).map(_count_heavy_atoms)

    if report["sa_score"].isna().any():
        missing_mask = report["sa_score"].isna()
        computed = report.loc[missing_mask, "smiles"].astype(str).map(_compute_sa_score)
        report.loc[missing_mask, "sa_score"] = pd.to_numeric(computed, errors="coerce")

    mean_pred = float(report["prediction"].mean()) if report["prediction"].notna().any() else np.nan
    mean_excess = float(report["target_excess"].mean()) if report["target_excess"].notna().any() else np.nan
    mean_violation = float(report["target_violation"].mean()) if report["target_violation"].notna().any() else np.nan
    mean_d2 = float(report["d2_distance"].mean()) if report["d2_distance"].notna().any() else np.nan
    mean_ood_prop = float(report["ood_prop"].mean()) if "ood_prop" in report.columns and report["ood_prop"].notna().any() else np.nan
    mean_ood_gen = float(report["ood_gen"].mean()) if "ood_gen" in report.columns and report["ood_gen"].notna().any() else np.nan
    mean_sa = float(report["sa_score"].mean()) if report["sa_score"].notna().any() else np.nan
    unique_ratio = float(report["canonical_smiles"].nunique() / max(len(report), 1))
    novelty_ratio = float(pd.to_numeric(report.get("is_novel"), errors="coerce").fillna(0).mean()) if "is_novel" in report.columns else np.nan

    summary = {
        "property": property_name,
        "target_value": float(target_value),
        "target_mode": mode,
        "epsilon": float(epsilon),
        "sampling_target": int(sampling_target),
        "sampling_target_total": int(target_total),
        "n_accepted": int(len(report)),
        "acceptance_ratio_vs_target": round(float(len(report) / max(target_total, 1)), 4),
        "mean_prediction": round(mean_pred, 6) if np.isfinite(mean_pred) else np.nan,
        "mean_target_excess": round(mean_excess, 6) if np.isfinite(mean_excess) else np.nan,
        "mean_target_violation": round(mean_violation, 6) if np.isfinite(mean_violation) else np.nan,
        "mean_d2_distance": round(mean_d2, 6) if np.isfinite(mean_d2) else np.nan,
        "mean_ood_prop": round(mean_ood_prop, 6) if np.isfinite(mean_ood_prop) else np.nan,
        "mean_ood_gen": round(mean_ood_gen, 6) if np.isfinite(mean_ood_gen) else np.nan,
        "mean_sa_score": round(mean_sa, 6) if np.isfinite(mean_sa) else np.nan,
        "unique_ratio_canonical": round(unique_ratio, 4),
        "novelty_ratio": round(novelty_ratio, 4) if np.isfinite(novelty_ratio) else np.nan,
    }

    ordered_cols = [
        "rank",
        "smiles",
        "canonical_smiles",
        "proposal_view",
        "matched_class",
        "prediction",
        "prediction_std",
        "target_excess",
        "target_violation",
        "ood_prop",
        "ood_gen",
        "d2_distance",
        "sa_score",
        "star_count",
        "aromatic_ring_count",
        "ring_count",
        "heavy_atom_count",
        "is_novel",
        "batch_idx",
        "rank_score",
    ]
    existing_cols = [c for c in ordered_cols if c in report.columns]
    other_cols = [c for c in report.columns if c not in existing_cols and not c.startswith("_")]
    report = report[existing_cols + other_cols]
    return report, summary


def _plot_f5_accepted_polymer_overview(
    *,
    report_df: pd.DataFrame,
    property_name: str,
    target_value: float,
    target_mode: str,
    epsilon: float,
    figures_dir: Path,
) -> None:
    if plt is None or report_df.empty:
        return

    property_label = property_display_name(property_name)
    mode = str(target_mode).strip().lower()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax0, ax1, ax2, ax3 = axes.reshape(-1)

    # A) Ranked score bars.
    top_n = min(30, len(report_df))
    top_df = report_df.head(top_n).copy()
    y = np.arange(top_n, dtype=np.float32)
    if mode in {"ge", "le"}:
        vals = pd.to_numeric(top_df.get("target_excess"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        x_label = _target_excess_axis_label(property_name, mode)
        colors = ["#2A9D8F" if v >= 0 else "#E76F51" for v in vals]
    else:
        violation = pd.to_numeric(top_df.get("target_violation"), errors="coerce").fillna(np.inf).to_numpy(dtype=np.float32)
        vals = -violation
        x_label = "Negative target violation (higher is better)"
        colors = ["#2A9D8F"] * len(vals)

    labels = [f"#{int(r)}" for r in top_df["rank"].tolist()]
    ax0.barh(y, vals, color=colors, alpha=0.9)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.invert_yaxis()
    ax0.set_xlabel(x_label)
    ax0.set_title(f"Top accepted {property_label} candidates", fontsize=14, fontweight="bold")
    ax0.grid(axis="x", alpha=0.25)

    # B) Prediction vs OOD distance scatter.
    ood_col = "ood_prop" if "ood_prop" in report_df.columns else "d2_distance"
    ood_xlabel = "OOD prop (cosine dist to D2, lower is better)" if ood_col == "ood_prop" else "D2 distance (lower is better)"
    scatter_df = report_df.dropna(subset=["prediction", ood_col]).copy()
    if not scatter_df.empty:
        color_vals = pd.to_numeric(scatter_df.get("target_excess"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        sc = ax1.scatter(
            scatter_df[ood_col].to_numpy(dtype=np.float32),
            scatter_df["prediction"].to_numpy(dtype=np.float32),
            c=color_vals,
            cmap="RdYlGn",
            s=34,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.3,
        )
        ax1.axhline(float(target_value), color="#111111", linestyle="--", linewidth=1.0, label="Target")
        if mode == "window":
            ax1.axhspan(float(target_value) - float(epsilon), float(target_value) + float(epsilon), color="#2A9D8F", alpha=0.12)
        ax1.set_xlabel(ood_xlabel)
        ax1.set_ylabel(f"Predicted {property_label}")
        ax1.set_title("Accepted candidates: property vs OOD", fontsize=14, fontweight="bold")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="best", fontsize=11)
        fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="Target excess")
    else:
        ax1.text(0.5, 0.5, f"No finite prediction/{ood_col} points", ha="center", va="center")
        ax1.set_axis_off()

    # C) Descriptor profile boxplots.
    descriptor_cols = [
        ("ood_prop", "OOD prop"),
        ("ood_gen", "OOD gen"),
        ("d2_distance", "D2"),
        ("sa_score", "SA"),
        ("aromatic_ring_count", "Aro rings"),
        ("ring_count", "Rings"),
        ("heavy_atom_count", "Heavy atoms"),
    ]
    arrays = []
    labels_c = []
    for col, label in descriptor_cols:
        if col in report_df.columns:
            arr = pd.to_numeric(report_df[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
            if arr.size:
                arrays.append(arr)
                labels_c.append(label)
    if arrays:
        bp = ax2.boxplot(arrays, labels=labels_c, patch_artist=True, showfliers=False)
        palette = ["#4E79A7", "#2A9D8F", "#F4A261", "#264653", "#E76F51"]
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax2.set_title("Accepted polymer descriptor profile", fontsize=14, fontweight="bold")
        ax2.grid(axis="y", alpha=0.25)
    else:
        ax2.text(0.5, 0.5, "No descriptor values", ha="center", va="center")
        ax2.set_axis_off()

    # D) Top-ranked table.
    show_n = min(12, len(report_df))
    table_df = report_df.head(show_n).copy()
    cols = []
    if "rank" in table_df.columns:
        cols.append(("rank", "Rank"))
    cols.append(("smiles", "Polymer p-SMILES"))
    if "prediction" in table_df.columns:
        cols.append(("prediction", "Pred"))
    if "target_excess" in table_df.columns:
        cols.append(("target_excess", "Excess"))
    if "ood_prop" in table_df.columns:
        cols.append(("ood_prop", "OODprop"))
    elif "d2_distance" in table_df.columns:
        cols.append(("d2_distance", "D2"))
    if "sa_score" in table_df.columns:
        cols.append(("sa_score", "SA"))

    col_labels = [x[1] for x in cols]
    cell_rows = []
    for _, row in table_df.iterrows():
        row_vals = []
        for key, _ in cols:
            value = row.get(key, "")
            if key == "smiles":
                row_vals.append(_short_text(value, max_len=38))
            elif isinstance(value, (float, np.floating)):
                row_vals.append(f"{float(value):.3f}" if np.isfinite(value) else "")
            else:
                row_vals.append(str(value))
        cell_rows.append(row_vals)

    ax3.axis("off")
    if cell_rows:
        table = ax3.table(cellText=cell_rows, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.15)
        ax3.set_title("Top accepted polymers (detailed info)", fontsize=14, fontweight="bold", pad=8)
    else:
        ax3.text(0.5, 0.5, "No accepted candidates", ha="center", va="center")

    fig.suptitle(f"F5 accepted polymer report: {property_label}", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save_figure_png(fig, figures_dir / f"figure_f5_accepted_polymer_overview_{property_name}")
    plt.close(fig)


def _plot_f5_accepted_polymer_gallery(
    *,
    report_df: pd.DataFrame,
    property_name: str,
    figures_dir: Path,
    top_n: int = 36,
) -> None:
    if report_df.empty or Chem is None:
        return
    try:
        from rdkit.Chem import Draw  # type: ignore
    except Exception:
        return

    gallery_df = report_df.head(max(int(top_n), 1)).copy()
    mols = []
    legends = []
    for _, row in gallery_df.iterrows():
        smi = str(row.get("smiles", "")).strip()
        mol = _rdkit_mol_from_polymer_smiles(smi)
        if mol is None:
            continue
        rank_val = row.get("rank", "")
        pred_val = pd.to_numeric(pd.Series([row.get("prediction")]), errors="coerce").iloc[0]
        ood_val = pd.to_numeric(pd.Series([row.get("ood_prop", row.get("d2_distance"))]), errors="coerce").iloc[0]
        rank_text = str(int(rank_val)) if pd.notna(rank_val) else "?"
        legend = f"#{rank_text} | pred={pred_val:.2f} | ood={ood_val:.3f}" if np.isfinite(pred_val) and np.isfinite(ood_val) else f"#{rank_text}"
        mols.append(mol)
        legends.append(legend)

    if not mols:
        return

    try:
        draw_options = Draw.rdMolDraw2D.MolDrawOptions()
        # Keep the RDKit-rendered gallery aligned with the publication-wide
        # typography requirement instead of relying on RDKit defaults.
        draw_options.fixedFontSize = 16
        draw_options.legendFontSize = 16
        draw_options.annotationFontScale = 1.0
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=6,
            subImgSize=(260, 200),
            legends=legends,
            drawOptions=draw_options,
            useSVG=False,
        )
        out_path = figures_dir / f"figure_f5_accepted_polymer_gallery_{property_name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(img, "save"):
            img.save(str(out_path))
    except Exception:
        return


def _f5_cfg(config: dict) -> dict:
    merged = dict(DEFAULT_F5_RESAMPLE_SETTINGS)
    merged.update(config.get("foundation_inverse", {}) or {})
    return merged


def _is_view_enabled(config: dict, view: str) -> bool:
    views_cfg = config.get("views", {})
    if view in views_cfg and not views_cfg[view].get("enabled", True):
        return False
    return True


def _select_encoder_view(config: dict, override: Optional[str]) -> str:
    requested = override
    if requested is None:
        requested = str(config.get("foundation_inverse", {}).get("encoder_view", "")).strip() or None

    if requested:
        if requested not in SUPPORTED_VIEWS:
            raise ValueError(f"Unsupported encoder_view={requested}. Supported: {', '.join(SUPPORTED_VIEWS)}")
        if not _is_view_enabled(config, requested):
            raise ValueError(f"encoder_view={requested} is disabled in config.views.")
        encoder_key = VIEW_SPECS[requested]["encoder_key"]
        if not config.get(encoder_key, {}).get("method_dir"):
            raise ValueError(f"Encoder config missing for view={requested} ({encoder_key}.method_dir).")
        return requested

    candidate_views = []
    configured_order = config.get("alignment_views", list(SUPPORTED_VIEWS))
    for view in configured_order:
        if view in SUPPORTED_VIEWS:
            candidate_views.append(view)
    for view in SUPPORTED_VIEWS:
        if view not in candidate_views:
            candidate_views.append(view)

    for view in candidate_views:
        if not _is_view_enabled(config, view):
            continue
        encoder_key = VIEW_SPECS[view]["encoder_key"]
        if config.get(encoder_key, {}).get("method_dir"):
            return view
    return "smiles"


def _normalize_property_name(value: Any) -> str:
    return shared_normalize_property_name(value)


def _parse_property_names_from_files(values: Any) -> List[str]:
    names: List[str] = []
    if values is None:
        return names
    if isinstance(values, str):
        raw_items = [x.strip() for x in values.split(",")]
    elif isinstance(values, (list, tuple, set)):
        raw_items = [str(x).strip() for x in values]
    else:
        raw_items = [str(values).strip()]
    for item in raw_items:
        name = _normalize_property_name(item)
        if name and name not in names:
            names.append(name)
    return names


def _lookup_property_config(mapping: Any, property_name: str):
    if not isinstance(mapping, dict):
        return None
    prop = _normalize_property_name(property_name).lower()
    for key, value in mapping.items():
        if _normalize_property_name(key).lower() == prop:
            return value
    return None


def _is_active_property_in_config(f5_cfg: dict, property_name: str) -> bool:
    cfg_prop = _normalize_property_name(f5_cfg.get("property", ""))
    arg_prop = _normalize_property_name(property_name)
    return bool(cfg_prop and arg_prop and cfg_prop.lower() == arg_prop.lower())


def _resolve_target_value(arg_target: Optional[float], property_name: str, f5_cfg: dict) -> float:
    if arg_target is not None:
        return float(arg_target)

    value = _lookup_property_config(f5_cfg.get("targets", {}) or {}, property_name)
    if value is not None:
        return float(value)

    if _is_active_property_in_config(f5_cfg, property_name):
        scalar = f5_cfg.get("target")
        if scalar is not None:
            return float(scalar)

    raise ValueError(
        f"Missing target for property='{property_name}'. Provide --target, or set "
        f"foundation_inverse.targets.{property_name} in config."
    )


def _resolve_target_mode(arg_target_mode: Optional[str], property_name: str, f5_cfg: dict) -> str:
    mode = str(arg_target_mode).strip().lower() if arg_target_mode is not None else ""
    if not mode:
        value = _lookup_property_config(f5_cfg.get("target_modes", {}) or {}, property_name)
        if value is not None:
            mode = str(value).strip().lower()
    if not mode and _is_active_property_in_config(f5_cfg, property_name):
        value = f5_cfg.get("target_mode")
        if value is not None:
            mode = str(value).strip().lower()
    if not mode:
        mode = "window"
    if mode not in {"window", "ge", "le"}:
        raise ValueError(
            f"Invalid target_mode='{mode}' for property='{property_name}'. "
            "Supported values: window|ge|le."
        )
    return mode


def _resolve_epsilon(arg_epsilon: Optional[float], property_name: str, f5_cfg: dict) -> float:
    if arg_epsilon is not None:
        return float(arg_epsilon)

    value = _lookup_property_config(f5_cfg.get("epsilons", {}) or {}, property_name)
    if value is not None:
        return float(value)

    if _is_active_property_in_config(f5_cfg, property_name):
        scalar = f5_cfg.get("epsilon")
        if scalar is not None:
            return float(scalar)

    return 30.0


def _resolve_committee_properties(config: dict, target_property: str, override: Optional[str]) -> List[str]:
    f5_cfg = _f5_cfg(config)
    names: List[str] = []

    if override is not None:
        names = _parse_property_names_from_files(override)
    if not names:
        names = _parse_property_names_from_files(f5_cfg.get("committee_properties"))
    if not names:
        names = _parse_property_names_from_files((config.get("property", {}) or {}).get("files"))

    target_name = _normalize_property_name(target_property)
    if target_name and target_name not in names:
        names.insert(0, target_name)
    if not names and target_name:
        names = [target_name]
    return names


def _parse_view_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.lower() == "all":
            return ["all"]
        raw = [x.strip() for x in text.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(x).strip() for x in value]
    else:
        raw = [str(value).strip()]

    views: List[str] = []
    for item in raw:
        if not item:
            continue
        low = item.lower()
        if low == "all":
            return ["all"]
        if low not in SUPPORTED_VIEWS:
            raise ValueError(f"Unsupported view '{item}'. Supported: {', '.join(SUPPORTED_VIEWS)}")
        if low not in views:
            views.append(low)
    return views


def _select_proposal_views(config: dict, override: Optional[str], fallback_view: str) -> List[str]:
    f5_cfg = _f5_cfg(config)
    raw = override if override is not None else f5_cfg.get("proposal_views", "all")
    parsed = _parse_view_list(raw)

    available: List[str] = []
    ordered = []
    for view in config.get("alignment_views", list(SUPPORTED_VIEWS)):
        if view in SUPPORTED_VIEWS and view not in ordered:
            ordered.append(view)
    for view in SUPPORTED_VIEWS:
        if view not in ordered:
            ordered.append(view)
    for view in ordered:
        if not _is_view_enabled(config, view):
            continue
        encoder_key = VIEW_SPECS[view]["encoder_key"]
        if not config.get(encoder_key, {}).get("method_dir"):
            continue
        available.append(view)

    if parsed == ["all"] or not parsed:
        selected = list(available)
    else:
        selected = []
        for view in parsed:
            if view not in available:
                raise ValueError(
                    f"proposal view '{view}' is unavailable (disabled or encoder config missing)."
                )
            if view not in selected:
                selected.append(view)

    if fallback_view not in selected:
        selected.append(fallback_view)
    if not selected:
        selected = [fallback_view]
    return selected


def _load_module(module_name: str, path: Path):
    return _shared_load_module(module_name, path, REPO_ROOT)


def _embed_sequence(inputs: List[str], assets: dict, device: str) -> np.ndarray:
    if not inputs:
        return np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32)
    embeddings = []
    for start in range(0, len(inputs), assets["batch_size"]):
        batch = inputs[start:start + assets["batch_size"]]
        encoded = assets["tokenizer"].batch_encode(batch)
        input_ids = torch.tensor(encoded["input_ids"], device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=device)
        timesteps = torch.full((input_ids.size(0),), int(assets["timestep"]), device=device, dtype=torch.long)
        with torch.no_grad():
            pooled = assets["backbone"].get_pooled_output(
                input_ids=input_ids,
                timesteps=timesteps,
                attention_mask=attention_mask,
                pooling=assets["pooling"],
            )
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _embed_graph(smiles_list: List[str], assets: dict, device: str) -> Tuple[np.ndarray, List[int]]:
    if not smiles_list:
        return np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32), []
    valid_indices = []
    graph_batches = []
    for idx, smi in enumerate(smiles_list):
        try:
            data = assets["tokenizer"].encode(smi)
            graph_batches.append(data)
            valid_indices.append(idx)
        except Exception as e:
            print(f"[F5] Warning: graph encoding failed for molecule {idx}: {str(e)[:80]}")
            continue
    embeddings = []
    for start in range(0, len(graph_batches), assets["batch_size"]):
        batch = graph_batches[start:start + assets["batch_size"]]
        if not batch:
            continue
        X = np.stack([b["X"] for b in batch])
        E = np.stack([b["E"] for b in batch])
        M = np.stack([b["M"] for b in batch])
        X_t = torch.tensor(X, device=device)
        E_t = torch.tensor(E, device=device)
        M_t = torch.tensor(M, device=device)
        timesteps = torch.full((X_t.size(0),), int(assets["timestep"]), device=device, dtype=torch.long)
        with torch.no_grad():
            pooled = assets["backbone"].get_node_embeddings(X_t, E_t, timesteps, M_t, pooling=assets["pooling"])
        embeddings.append(pooled.cpu().numpy())
    if embeddings:
        return np.concatenate(embeddings, axis=0), valid_indices
    return np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32), valid_indices


def _sanitize_sequence_inputs(inputs: List[str], tokenizer) -> Tuple[List[str], List[int]]:
    valid_inputs = []
    valid_indices = []
    for idx, text in enumerate(inputs):
        try:
            tokenizer.encode(text, add_special_tokens=True, padding=True, return_attention_mask=True)
            valid_inputs.append(text)
            valid_indices.append(idx)
        except Exception:
            continue
    return valid_inputs, valid_indices


def _embed_candidates(view: str, smiles_list: List[str], assets: dict, device: str) -> Tuple[np.ndarray, List[int]]:
    if view == "graph":
        return _embed_graph(smiles_list, assets, device)

    if view == "selfies":
        seq_inputs = []
        seq_indices = []
        for idx, smi in enumerate(smiles_list):
            converted = smiles_to_selfies(smi)
            if not converted:
                continue
            seq_inputs.append(converted)
            seq_indices.append(idx)
        seq_inputs, filtered_local_indices = _sanitize_sequence_inputs(seq_inputs, assets["tokenizer"])
        filtered_indices = [seq_indices[i] for i in filtered_local_indices]
        embeddings = _embed_sequence(seq_inputs, assets, device) if seq_inputs else np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32)
        return embeddings, filtered_indices

    seq_inputs, valid_indices = _sanitize_sequence_inputs(smiles_list, assets["tokenizer"])
    embeddings = _embed_sequence(seq_inputs, assets, device) if seq_inputs else np.zeros((0, assets["backbone"].hidden_size), dtype=np.float32)
    return embeddings, valid_indices


def _load_d2_embeddings(results_dir: Path, view: str) -> np.ndarray:
    direct_path = results_dir / f"embeddings_{view}_d2.npy"
    if direct_path.exists():
        return np.load(direct_path)
    step1_path = results_dir / "step1_alignment_embeddings" / "files" / f"embeddings_{view}_d2.npy"
    if step1_path.exists():
        return np.load(step1_path)
    if view == "smiles":
        legacy_path = results_dir / "embeddings_d2.npy"
        if legacy_path.exists():
            return np.load(legacy_path)
    raise FileNotFoundError(
        f"D2 embeddings not found for view={view}. Expected {direct_path} "
        f"(or step1 path {step1_path}, or legacy embeddings_d2.npy for smiles). "
        "Run F1 with this view enabled first."
    )


def _load_d1_embeddings(results_dir: Path, view: str) -> np.ndarray:
    """Load D1 (backbone training set) embeddings for the given view.

    Searches in order:
      1. results_dir/embeddings_{view}_d1.npy
      2. results_dir/step1_alignment_embeddings/files/embeddings_{view}_d1.npy
      3. Legacy results_dir/embeddings_d1.npy (smiles view only)

    Raises FileNotFoundError if none exist.
    """
    direct_path = results_dir / f"embeddings_{view}_d1.npy"
    if direct_path.exists():
        return np.load(direct_path)
    step1_path = results_dir / "step1_alignment_embeddings" / "files" / f"embeddings_{view}_d1.npy"
    if step1_path.exists():
        return np.load(step1_path)
    if view == "smiles":
        legacy_path = results_dir / "embeddings_d1.npy"
        if legacy_path.exists():
            return np.load(legacy_path)
    raise FileNotFoundError(
        f"D1 embeddings not found for view={view}. Expected {direct_path} "
        f"(or step1 path {step1_path}, or legacy embeddings_d1.npy for smiles). "
        "Run F1 with this view enabled first."
    )


def _discover_property_model_paths(
    results_dir: Path,
    property_name: str,
    *,
    include_multiview_mean: bool = False,
) -> Dict[str, Path]:
    model_dir = results_dir / "step3_property"
    files_dir = model_dir / "files"
    prop = _normalize_property_name(property_name)
    prop_files_dir = model_dir / prop / "files" if prop else None
    prop_step_dir = model_dir / prop if prop else None
    discovered: Dict[str, Path] = {}

    for view in SUPPORTED_VIEWS:
        filenames = [f"{prop}_{view}_mlp.pt"]
        if view == "smiles":
            # Backward compatibility for older F3 runs.
            filenames.append(f"{prop}_mlp.pt")
        candidates = []
        for filename in filenames:
            if prop_files_dir is not None:
                candidates.append(prop_files_dir / filename)
            if prop_step_dir is not None:
                candidates.append(prop_step_dir / filename)
            candidates.extend([files_dir / filename, model_dir / filename])
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is not None:
            discovered[view] = model_path

    if include_multiview_mean:
        mv_filename = f"{prop}_multiview_mean_mlp.pt"
        mv_candidates = []
        if prop_files_dir is not None:
            mv_candidates.append(prop_files_dir / mv_filename)
        if prop_step_dir is not None:
            mv_candidates.append(prop_step_dir / mv_filename)
        mv_candidates.extend([files_dir / mv_filename, model_dir / mv_filename])
        mv_path = next((p for p in mv_candidates if p.exists()), None)
        if mv_path is not None:
            discovered["multiview_mean"] = mv_path

    return discovered


def _build_prediction_columns_from_all_models(
    *,
    config: dict,
    results_dir: Path,
    smiles_list: List[str],
    model_paths: Dict[str, Path],
) -> Dict[str, Any]:
    n = len(smiles_list)
    if n == 0:
        return {
            "prediction_by_model": {},
            "prediction_ensemble": np.zeros((0,), dtype=np.float32),
            "prediction_std": np.zeros((0,), dtype=np.float32),
            "prediction_valid_count": np.zeros((0,), dtype=np.int64),
            "model_order": [],
            "models_used": {},
        }

    view_cache: Dict[str, Dict[str, Any]] = {}
    prediction_by_model: Dict[str, np.ndarray] = {}
    models_used: Dict[str, str] = {}

    def _view_pack(view: str) -> Dict[str, Any]:
        cached = view_cache.get(view)
        if cached is not None:
            return cached

        encoder_cfg = config.get(VIEW_SPECS[view]["encoder_key"], {})
        device = encoder_cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assets = _load_view_assets(config=config, view=view, device=device)
        embeddings, kept_indices = _embed_candidates(
            view=view,
            smiles_list=smiles_list,
            assets=assets,
            device=device,
        )
        packed = {
            "embeddings": embeddings,
            "kept_indices": [int(i) for i in kept_indices],
        }
        view_cache[view] = packed
        return packed

    # Per-view models
    for view in SUPPORTED_VIEWS:
        model_path = model_paths.get(view)
        if model_path is None:
            continue
        packed = _view_pack(view)
        pred_col = np.full((n,), np.nan, dtype=np.float32)
        if packed["embeddings"].size and packed["kept_indices"]:
            model = _load_property_model(model_path)
            preds = np.asarray(model.predict(packed["embeddings"]), dtype=np.float32).reshape(-1)
            for local_i, global_i in enumerate(packed["kept_indices"]):
                pred_col[global_i] = preds[local_i]
        prediction_by_model[view] = pred_col
        models_used[view] = str(model_path)

    # Multiview-mean model
    mv_path = model_paths.get("multiview_mean")
    if mv_path is not None:
        available_views = [v for v in SUPPORTED_VIEWS if v in prediction_by_model]
        common_indices = None
        for view in available_views:
            idx_set = set(view_cache[view]["kept_indices"])
            common_indices = idx_set if common_indices is None else common_indices.intersection(idx_set)
        common = sorted(common_indices) if common_indices else []

        pred_col = np.full((n,), np.nan, dtype=np.float32)
        if common:
            dim_set = set()
            fused_blocks = []
            for view in available_views:
                packed = view_cache[view]
                idx_to_row = {idx: row_i for row_i, idx in enumerate(packed["kept_indices"])}
                block = packed["embeddings"][[idx_to_row[i] for i in common]]
                fused_blocks.append(block)
                dim_set.add(int(block.shape[1]))
            if len(dim_set) == 1:
                fused = np.mean(np.stack(fused_blocks, axis=0), axis=0)
                mv_model = _load_property_model(mv_path)
                preds = np.asarray(mv_model.predict(fused), dtype=np.float32).reshape(-1)
                for local_i, global_i in enumerate(common):
                    pred_col[global_i] = preds[local_i]
            else:
                print(
                    "[F5] Warning: skipping multiview_mean model due to embedding-dimension mismatch "
                    f"across views: {sorted(dim_set)}"
                )

        prediction_by_model["multiview_mean"] = pred_col
        models_used["multiview_mean"] = str(mv_path)

    if not prediction_by_model:
        return {
            "prediction_by_model": {},
            "prediction_ensemble": np.full((n,), np.nan, dtype=np.float32),
            "prediction_std": np.full((n,), np.nan, dtype=np.float32),
            "prediction_valid_count": np.zeros((n,), dtype=np.int64),
            "model_order": [],
            "models_used": {},
        }

    ordered = [k for k in list(SUPPORTED_VIEWS) + ["multiview_mean"] if k in prediction_by_model]
    pred_matrix = np.column_stack([prediction_by_model[k] for k in ordered]).astype(np.float32, copy=False)
    valid_counts = np.sum(np.isfinite(pred_matrix), axis=1)
    with np.errstate(invalid="ignore"):
        ensemble = np.nanmean(pred_matrix, axis=1).astype(np.float32, copy=False)
    with np.errstate(invalid="ignore"):
        ensemble_std = np.nanstd(pred_matrix, axis=1).astype(np.float32, copy=False)
    ensemble[valid_counts == 0] = np.nan
    ensemble_std[valid_counts == 0] = np.nan

    return {
        "prediction_by_model": prediction_by_model,
        "prediction_ensemble": ensemble,
        "prediction_std": ensemble_std,
        "prediction_valid_count": valid_counts.astype(np.int64, copy=False),
        "model_order": ordered,
        "models_used": models_used,
    }


def _check_validity(smiles: str) -> bool:
    if Chem is None:
        return True
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _count_stars(smiles: str) -> int:
    return smiles.count("*") if isinstance(smiles, str) else 0


def _load_training_smiles(path: Path) -> set:
    shared_df = _load_shared_unlabeled_train_df()
    if not shared_df.empty and "p_smiles" in shared_df.columns:
        return set(shared_df["p_smiles"].astype(str).tolist())

    if not path.exists():
        return set()
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    col = "p_smiles" if "p_smiles" in df.columns else "SMILES" if "SMILES" in df.columns else None
    if col is None:
        return set()
    return set(df[col].astype(str).tolist())


@lru_cache(maxsize=1)
def _load_shared_unlabeled_train_df_cached() -> pd.DataFrame:
    try:
        train_path, _ = require_preprocessed_unlabeled_splits(REPO_ROOT)
        return pd.read_csv(train_path)
    except Exception:
        return pd.DataFrame()


def _load_shared_unlabeled_train_df() -> pd.DataFrame:
    return _load_shared_unlabeled_train_df_cached().copy()


def _step2_lengths_from_psmiles(train_df: pd.DataFrame, tokenizer: Any, num_samples: int, random_seed: int) -> Optional[List[int]]:
    if train_df is None or train_df.empty or "p_smiles" not in train_df.columns or num_samples <= 0:
        return None
    replace = num_samples > len(train_df)
    sampled = train_df["p_smiles"].sample(
        n=num_samples,
        replace=replace,
        random_state=int(random_seed),
    )
    max_length = int(getattr(tokenizer, "max_length", 256))
    return [
        min(len(tokenizer.tokenize(str(s))) + 2, max_length)
        for s in sampled.tolist()
    ]


def _step2_lengths_from_selfies(
    train_df: pd.DataFrame,
    tokenizer: Any,
    num_samples: int,
    random_seed: int,
    sample_selfies_from_dataframe_fn: Optional[Any],
) -> Optional[List[int]]:
    if (
        train_df is None
        or train_df.empty
        or num_samples <= 0
        or sample_selfies_from_dataframe_fn is None
    ):
        return None
    sampled = sample_selfies_from_dataframe_fn(
        train_df,
        num_samples=int(num_samples),
        random_seed=int(random_seed),
    )
    max_length = int(getattr(tokenizer, "max_length", 256))
    return [
        min(len(tokenizer.tokenize(str(s))) + 2, max_length)
        for s in sampled
    ]


def _achievement_rates(preds: np.ndarray, target: float) -> dict:
    if preds is None or len(preds) == 0:
        return {
            "achievement_5p": 0.0,
            "achievement_10p": 0.0,
            "achievement_15p": 0.0,
            "achievement_20p": 0.0,
        }
    denom = max(abs(float(target)), 1e-9)
    return {
        "achievement_5p": float(np.mean(np.abs(preds - target) <= 0.05 * denom)),
        "achievement_10p": float(np.mean(np.abs(preds - target) <= 0.10 * denom)),
        "achievement_15p": float(np.mean(np.abs(preds - target) <= 0.15 * denom)),
        "achievement_20p": float(np.mean(np.abs(preds - target) <= 0.20 * denom)),
    }


def _compute_hits(preds: np.ndarray, target: float, epsilon: float, target_mode: str) -> np.ndarray:
    mode = str(target_mode).strip().lower()
    if mode == "window":
        return np.abs(preds - target) <= epsilon
    if mode == "ge":
        return preds >= target
    if mode == "le":
        return preds <= target
    raise ValueError(f"Unsupported target_mode={target_mode}. Use window|ge|le.")


def _compute_target_excess(preds: np.ndarray, target: float, target_mode: str) -> np.ndarray:
    """Signed distance to target boundary.

    Positive values indicate better achievement:
    - window: pred - target
    - ge: pred - target
    - le: target - pred
    """
    arr = np.asarray(preds, dtype=np.float32).reshape(-1)
    mode = str(target_mode).strip().lower()
    if mode == "le":
        return (float(target) - arr).astype(np.float32, copy=False)
    return (arr - float(target)).astype(np.float32, copy=False)


def _compute_target_violation(preds: np.ndarray, target: float, target_mode: str) -> np.ndarray:
    """Non-negative boundary violation in property units."""
    excess = _compute_target_excess(preds, target, target_mode)
    mode = str(target_mode).strip().lower()
    if mode == "window":
        return np.abs(excess).astype(np.float32, copy=False)
    return np.maximum(0.0, -excess).astype(np.float32, copy=False)


def _target_excess_axis_label(property_name: str, target_mode: str) -> str:
    property_label = property_display_name(property_name)
    mode = str(target_mode).strip().lower()
    if mode == "le":
        return f"Target excess (target - predicted {property_label})"
    return f"Target excess (predicted {property_label} - target)"


_SA_SCORE_FN = None


def _get_sa_score_fn():
    global _SA_SCORE_FN
    if _SA_SCORE_FN is not None:
        return _SA_SCORE_FN
    chem_path = REPO_ROOT / "Bi_Diffusion_SMILES" / "src" / "utils" / "chemistry.py"
    if not chem_path.exists():
        _SA_SCORE_FN = False
        return None
    try:
        chem_mod = _load_module("mvf_sa_chemistry", chem_path)
        _SA_SCORE_FN = getattr(chem_mod, "compute_sa_score", False)
    except Exception:
        _SA_SCORE_FN = False
    return _SA_SCORE_FN if callable(_SA_SCORE_FN) else None


def _compute_sa_score(smiles: str) -> Optional[float]:
    fn = _get_sa_score_fn()
    if fn is None:
        return None
    try:
        score = fn(smiles)
    except Exception:
        return None
    if score is None:
        return None
    try:
        return float(score)
    except Exception:
        return None


def _count_aromatic_rings(smiles: str) -> Optional[int]:
    """Count aromatic rings in a SMILES string using RDKit."""
    if Chem is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.replace("*", "[H]"))
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        from rdkit.Chem import rdMolDescriptors
        return int(rdMolDescriptors.CalcNumAromaticRings(mol))
    except Exception:
        return None


def _compute_pairwise_tanimoto_diversity(smiles_list: List[str], max_pairs: int = 10000) -> Optional[float]:
    """Compute mean pairwise Tanimoto diversity (1 - similarity) over accepted candidates."""
    if Chem is None or len(smiles_list) < 2:
        return None
    try:
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        return None
    fps = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        except Exception:
            continue
    n = len(fps)
    if n < 2:
        return None
    # For large sets, subsample pairs to avoid O(n^2) blowup
    if n * (n - 1) // 2 <= max_pairs:
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                total_dist += 1.0 - sim
                count += 1
        return total_dist / count if count > 0 else None
    else:
        rng = np.random.RandomState(42)
        indices = rng.randint(0, n, size=(max_pairs, 2))
        # Ensure different indices in each pair
        mask = indices[:, 0] != indices[:, 1]
        indices = indices[mask]
        total_dist = 0.0
        count = 0
        for i, j in indices:
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total_dist += 1.0 - sim
            count += 1
        return total_dist / count if count > 0 else None


def _canonicalize_smiles(smiles: str) -> str:
    text = str(smiles).strip()
    if not text or Chem is None:
        return text
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return text
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return text


def _parse_target_classes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        items = [x.strip() for x in str(value).split(",")]
    return [x for x in items if x]


def _compile_polymer_patterns(patterns: Dict[str, str]) -> Dict[str, Any]:
    if Chem is None:
        return {}
    compiled = {}
    for name, smarts in (patterns or {}).items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                compiled[str(name)] = patt
        except Exception:
            continue
    return compiled


def _match_polymer_class(smiles: str, target_classes: List[str], compiled_patterns: Dict[str, Any]) -> Tuple[bool, str]:
    if not target_classes:
        return True, ""
    if Chem is None:
        return False, ""
    try:
        mol = Chem.MolFromSmiles(smiles.replace("*", "[*]"))
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        return False, ""
    for class_name in target_classes:
        patt = compiled_patterns.get(class_name)
        if patt is None:
            continue
        try:
            if mol.HasSubstructMatch(patt):
                return True, class_name
        except Exception:
            continue
    return False, ""


def _create_generator(view: str, assets: dict, device: str, sampling_temperature: Optional[float], sampling_num_atoms: Optional[int]):
    method_dir = assets["method_dir"]
    method_cfg = assets.get("method_cfg", {}) or {}
    diffusion_cfg = method_cfg.get("diffusion", {}) or {}
    sampling_cfg = method_cfg.get("sampling", {}) or {}
    data_cfg = method_cfg.get("data", {}) or {}
    num_steps = int(diffusion_cfg.get("num_steps", 50))
    use_constraints = _to_bool(sampling_cfg.get("use_constraints", True), True)
    temperature = sampling_temperature if sampling_temperature is not None else float(sampling_cfg.get("temperature", 1.0))
    random_seed = int(data_cfg.get("random_seed", 42))

    if view == "graph":
        graph_diffusion_mod = _load_module(
            f"graph_diffusion_{method_dir.name}",
            method_dir / "src" / "model" / "graph_diffusion.py",
        )
        graph_sampler_mod = _load_module(
            f"graph_sampler_{method_dir.name}",
            method_dir / "src" / "sampling" / "graph_sampler.py",
        )
        graph_cfg = assets.get("graph_config", {}) or {}
        diffusion_model = graph_diffusion_mod.GraphMaskingDiffusion(
            backbone=assets["backbone"],
            num_steps=num_steps,
            beta_min=float(diffusion_cfg.get("beta_min", 1e-4)),
            beta_max=float(diffusion_cfg.get("beta_max", 2e-2)),
            force_clean_t0=_to_bool(diffusion_cfg.get("force_clean_t0", False), False),
            node_mask_id=assets["tokenizer"].mask_id,
            edge_mask_id=assets["tokenizer"].edge_vocab.get("MASK", 5),
            node_pad_id=assets["tokenizer"].pad_id,
            edge_none_id=assets["tokenizer"].edge_vocab.get("NONE", 0),
            lambda_node=float(diffusion_cfg.get("lambda_node", 1.0)),
            lambda_edge=float(diffusion_cfg.get("lambda_edge", 0.5)),
        )
        diffusion_model.to(device)
        diffusion_model.eval()
        sampler = graph_sampler_mod.GraphSampler(
            backbone=diffusion_model.backbone,
            graph_tokenizer=assets["tokenizer"],
            num_steps=num_steps,
            device=device,
            atom_count_distribution=graph_cfg.get("atom_count_distribution"),
            use_constraints=use_constraints,
        )
        return {
            "view": view,
            "sampler": sampler,
            "temperature": float(temperature),
            "num_atoms": sampling_num_atoms,
        }

    diffusion_mod = _load_module(
        f"diffusion_{method_dir.name}",
        method_dir / "src" / "model" / "diffusion.py",
    )
    sampler_mod = _load_module(
        f"sampler_{method_dir.name}",
        method_dir / "src" / "sampling" / "sampler.py",
    )
    diffusion_model = diffusion_mod.DiscreteMaskingDiffusion(
        backbone=assets["backbone"],
        num_steps=num_steps,
        beta_min=float(diffusion_cfg.get("beta_min", 1e-4)),
        force_clean_t0=_to_bool(diffusion_cfg.get("force_clean_t0", False), False),
        mask_token_id=assets["tokenizer"].mask_token_id,
        pad_token_id=assets["tokenizer"].pad_token_id,
        bos_token_id=assets["tokenizer"].bos_token_id,
        eos_token_id=assets["tokenizer"].eos_token_id,
    )
    diffusion_model.to(device)
    diffusion_model.eval()
    sampler = sampler_mod.ConstrainedSampler(
        diffusion_model=diffusion_model,
        tokenizer=assets["tokenizer"],
        num_steps=num_steps,
        temperature=float(temperature),
        use_constraints=use_constraints,
        device=device,
    )

    selfies_to_psmiles = None
    sample_selfies_from_dataframe_fn = None
    step2_sampling_mode = "fixed_length"
    canonicalize_outputs = False
    train_df = _load_shared_unlabeled_train_df()
    if view == "selfies":
        selfies_utils_mod = _load_module(
            f"selfies_utils_{method_dir.name}",
            method_dir / "src" / "utils" / "selfies_utils.py",
        )
        selfies_to_psmiles = getattr(selfies_utils_mod, "selfies_to_psmiles")
        sample_selfies_from_dataframe_fn = getattr(selfies_utils_mod, "sample_selfies_from_dataframe", None)
        if train_df is not None and not train_df.empty:
            step2_sampling_mode = "lengths_from_selfies"
    elif view in {"smiles", "smiles_bpe"}:
        if train_df is not None and not train_df.empty:
            step2_sampling_mode = "lengths_from_psmiles"
    elif view == "group_selfies":
        canonicalize_outputs = True

    return {
        "view": view,
        "sampler": sampler,
        "tokenizer": assets["tokenizer"],
        "seq_length": int(getattr(assets["tokenizer"], "max_length", 256)),
        "selfies_to_psmiles": selfies_to_psmiles,
        "sample_selfies_from_dataframe": sample_selfies_from_dataframe_fn,
        "step2_train_df": train_df,
        "step2_sampling_mode": step2_sampling_mode,
        "step2_sampling_round": 0,
        "step2_random_seed": random_seed,
        "canonicalize_outputs": canonicalize_outputs,
    }


def _sample_batch_from_generator(generator: dict, n: int, batch_size: int) -> List[str]:
    return [row["smiles"] for row in _sample_batch_records_from_generator(generator, n, batch_size) if row.get("smiles")]


def _sample_batch_records_from_generator(generator: dict, n: int, batch_size: int) -> List[dict]:
    if n <= 0:
        return []
    view = generator["view"]
    sampler = generator["sampler"]
    if view == "graph":
        raw = sampler.sample_batch(
            num_samples=int(n),
            batch_size=int(batch_size),
            show_progress=False,
            temperature=float(generator.get("temperature", 1.0)),
            num_atoms=generator.get("num_atoms"),
        )
        rows = []
        for sample in raw:
            text = str(sample).strip() if isinstance(sample, str) else ""
            rows.append(
                {
                    "raw_output": text,
                    "smiles": text,
                    "is_convertible": bool(text),
                }
            )
        return rows

    sample_kwargs = {
        "num_samples": int(n),
        "seq_length": int(generator["seq_length"]),
        "batch_size": int(batch_size),
        "show_progress": False,
    }
    sampling_mode = str(generator.get("step2_sampling_mode", "fixed_length")).strip().lower()
    sampling_round = int(generator.get("step2_sampling_round", 0))
    sampling_seed = int(generator.get("step2_random_seed", 42)) + sampling_round
    generator["step2_sampling_round"] = sampling_round + 1

    if sampling_mode == "lengths_from_psmiles":
        lengths = _step2_lengths_from_psmiles(
            generator.get("step2_train_df"),
            generator.get("tokenizer"),
            int(n),
            sampling_seed,
        )
        if lengths:
            sample_kwargs["lengths"] = lengths
    elif sampling_mode == "lengths_from_selfies":
        lengths = _step2_lengths_from_selfies(
            generator.get("step2_train_df"),
            generator.get("tokenizer"),
            int(n),
            sampling_seed,
            generator.get("sample_selfies_from_dataframe"),
        )
        if lengths:
            sample_kwargs["lengths"] = lengths

    _, outputs = sampler.sample_batch(**sample_kwargs)
    if view == "selfies":
        converter = generator.get("selfies_to_psmiles")
        converted = []
        for text in outputs:
            raw_text = str(text).strip() if isinstance(text, str) else ""
            try:
                psmiles = converter(raw_text) if converter is not None and raw_text else None
            except Exception:
                psmiles = None
            converted_text = str(psmiles).strip() if psmiles else ""
            converted.append(
                {
                    "raw_output": raw_text,
                    "smiles": converted_text,
                    "is_convertible": bool(converted_text),
                }
            )
        return converted

    rows = []
    for sample in outputs:
        raw_text = str(sample).strip() if isinstance(sample, str) else ""
        text = _canonicalize_smiles(raw_text) if generator.get("canonicalize_outputs", False) else raw_text
        rows.append(
            {
                "raw_output": raw_text,
                "smiles": text,
                "is_convertible": bool(text),
            }
        )
    return rows


def _resample_candidates_until_target(
    *,
    config: dict,
    args,
    proposal_views: List[str],
    scoring_view: str,
    scoring_assets: dict,
    scoring_device: str,
    property_model,
    training_set: set,
    target_value: float,
    epsilon: float,
    target_mode: str,
) -> dict:
    f5_cfg = _f5_cfg(config)
    sampling_target = int(args.sampling_target if args.sampling_target is not None else f5_cfg.get("sampling_target", 100))
    sampling_num_per_batch = int(args.sampling_num_per_batch if args.sampling_num_per_batch is not None else f5_cfg.get("sampling_num_per_batch", 100))
    sampling_batch_size = int(args.sampling_batch_size if args.sampling_batch_size is not None else f5_cfg.get("sampling_batch_size", 100))
    sampling_max_batches = int(args.sampling_max_batches if args.sampling_max_batches is not None else f5_cfg.get("sampling_max_batches", 200))

    sampling_temperature = args.sampling_temperature if args.sampling_temperature is not None else f5_cfg.get("sampling_temperature", None)
    sampling_num_atoms = args.sampling_num_atoms if args.sampling_num_atoms is not None else f5_cfg.get("sampling_num_atoms", None)
    if sampling_num_atoms in ("", "none", None):
        sampling_num_atoms = None
    elif sampling_num_atoms is not None:
        sampling_num_atoms = int(sampling_num_atoms)
    if sampling_temperature in ("", "none", None):
        sampling_temperature = None
    elif sampling_temperature is not None:
        sampling_temperature = float(sampling_temperature)

    target_class = args.target_class if args.target_class is not None else f5_cfg.get("target_class", "")
    require_validity = _to_bool(f5_cfg.get("require_validity", True), True)
    require_two_stars = _to_bool(f5_cfg.get("require_two_stars", True), True)
    require_novel = _to_bool(f5_cfg.get("require_novel", True), True)
    require_unique = _to_bool(f5_cfg.get("require_unique", True), True)
    max_sa = args.max_sa if args.max_sa is not None else f5_cfg.get("max_sa", None)
    per_view_min_hits = _to_int_or_none(
        args.per_view_min_hits if args.per_view_min_hits is not None else f5_cfg.get("per_view_min_hits", 0)
    )
    per_view_quota_relax_after_batches = _to_int_or_none(
        args.per_view_quota_relax_after_batches
        if args.per_view_quota_relax_after_batches is not None
        else f5_cfg.get("per_view_quota_relax_after_batches", 0)
    )
    if max_sa in ("", "none", None):
        max_sa = None
    else:
        max_sa = float(max_sa)
    if per_view_min_hits is None:
        per_view_min_hits = 0
    per_view_min_hits = int(per_view_min_hits)
    if per_view_min_hits < 0:
        raise ValueError("per_view_min_hits must be >= 0")
    if per_view_quota_relax_after_batches is None:
        per_view_quota_relax_after_batches = 0
    per_view_quota_relax_after_batches = int(per_view_quota_relax_after_batches)
    if per_view_quota_relax_after_batches < 0:
        raise ValueError("per_view_quota_relax_after_batches must be >= 0")
    if per_view_min_hits > 0 and per_view_quota_relax_after_batches == 0:
        # Keep quota phase short by default, then relax to avoid stalling when
        # some views cannot satisfy the requested hit floor.
        per_view_quota_relax_after_batches = max(1, len(proposal_views) * 2)

    if sampling_target <= 0:
        raise ValueError("sampling_target must be > 0")
    if sampling_num_per_batch <= 0 or sampling_batch_size <= 0:
        raise ValueError("sampling_num_per_batch and sampling_batch_size must both be > 0")
    if sampling_max_batches <= 0:
        raise ValueError("sampling_max_batches must be > 0")
    if not proposal_views:
        raise ValueError("proposal_views must be non-empty.")

    patterns = config.get("polymer_classes") or f5_cfg.get("polymer_class_patterns") or POLYMER_CLASS_PATTERNS
    target_classes = _parse_target_classes(target_class)
    compiled_patterns = _compile_polymer_patterns(patterns)
    for class_name in target_classes:
        if class_name not in compiled_patterns:
            raise ValueError(f"Unknown target class '{class_name}'. Available classes: {sorted(compiled_patterns.keys())}")
    if target_classes:
        print(f"[F5 resample] target_class filter enabled: {','.join(target_classes)}")
    if per_view_min_hits > 0:
        print(
            "[F5 resample] per-view balancing enabled: "
            f"min_hits_per_view={per_view_min_hits}, relax_after_batch={per_view_quota_relax_after_batches}"
        )

    scoring_generator = _create_generator(
        view=scoring_view,
        assets=scoring_assets,
        device=scoring_device,
        sampling_temperature=sampling_temperature,
        sampling_num_atoms=sampling_num_atoms,
    )
    stats = Counter()
    scored_records: List[dict] = []
    structural_valid_smiles: List[str] = []
    generated_smiles_all: List[str] = []
    seen_keys_by_view: Dict[str, set] = {view: set() for view in proposal_views}
    accepted_total = 0
    accepted_by_view: Counter = Counter()
    batches_by_view: Counter = Counter()
    active_generator_view: Optional[str] = None
    active_generator_assets: Optional[dict] = None
    active_generator: Optional[dict] = None
    pending_views = list(proposal_views)
    disabled_proposal_views: Dict[str, str] = {}

    def _get_proposal_generator(view: str) -> dict:
        nonlocal active_generator_view, active_generator_assets, active_generator
        if view == scoring_view:
            return scoring_generator
        if active_generator_view == view and active_generator is not None:
            return active_generator

        # Keep at most one non-scoring proposal generator resident to reduce peak memory.
        if active_generator_assets is not None:
            try:
                active_generator_assets.get("backbone").to("cpu")
            except Exception:
                pass
            active_generator_assets = None
            active_generator = None
            active_generator_view = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        proposal_device = _resolve_view_device(config, view)
        proposal_assets = _load_view_assets(config=config, view=view, device=proposal_device)
        proposal_generator = _create_generator(
            view=view,
            assets=proposal_assets,
            device=proposal_device,
            sampling_temperature=sampling_temperature,
            sampling_num_atoms=sampling_num_atoms,
        )
        active_generator_assets = proposal_assets
        active_generator = proposal_generator
        active_generator_view = view
        return proposal_generator

    batch_idx = 0
    while pending_views:
        progress_made = False
        for proposal_view in list(pending_views):
            if accepted_by_view[proposal_view] >= sampling_target:
                pending_views = [v for v in pending_views if v != proposal_view]
                continue
            if batches_by_view[proposal_view] >= sampling_max_batches:
                stats["stop_max_batches"] += 1
                stats[f"stop_max_batches_{proposal_view}"] += 1
                pending_views = [v for v in pending_views if v != proposal_view]
                print(
                    f"[F5 resample] view={proposal_view} reached max batches "
                    f"without hitting target ({accepted_by_view[proposal_view]}/{sampling_target})."
                )
                continue

            batch_idx += 1
            progress_made = True

            try:
                generator = _get_proposal_generator(proposal_view)
            except Exception as exc:
                stats["proposal_view_init_failed"] += 1
                stats[f"proposal_view_init_failed_{proposal_view}"] += 1
                disabled_proposal_views[proposal_view] = str(exc)
                pending_views = [v for v in pending_views if v != proposal_view]
                print(f"[F5 resample] disabling view={proposal_view} due to init error: {exc}")
                continue

            try:
                batch_smiles = _sample_batch_from_generator(generator, sampling_num_per_batch, sampling_batch_size)
            except Exception as exc:
                stats["proposal_view_sample_failed"] += 1
                stats[f"proposal_view_sample_failed_{proposal_view}"] += 1
                disabled_proposal_views[proposal_view] = str(exc)
                pending_views = [v for v in pending_views if v != proposal_view]
                print(f"[F5 resample] disabling view={proposal_view} due to sample error: {exc}")
                continue

            stats[f"n_generated_{proposal_view}"] += len(batch_smiles)
            batches_by_view[proposal_view] += 1
            stats[f"n_batches_{proposal_view}"] = int(batches_by_view[proposal_view])
            stats["n_generated"] += len(batch_smiles)
            generated_smiles_all.extend(batch_smiles)
            if not batch_smiles:
                stats["empty_batches"] += 1
                continue

            prefilter_smiles: List[str] = []
            prefilter_meta: List[dict] = []

            def _bump(key: str) -> None:
                stats[key] += 1
                stats[f"{key}_{proposal_view}"] += 1

            seen_keys_for_view = seen_keys_by_view.setdefault(proposal_view, set())
            for smi in batch_smiles:
                text = str(smi).strip()
                if not text:
                    _bump("reject_empty")
                    continue

                is_valid = _check_validity(text)
                is_two_star = _count_stars(text) == 2
                if is_valid:
                    stats["n_valid_any"] += 1
                    stats[f"n_valid_any_{proposal_view}"] += 1
                if is_valid and is_two_star:
                    stats["n_structural_valid"] += 1
                    stats[f"n_structural_valid_{proposal_view}"] += 1
                    structural_valid_smiles.append(text)

                if require_validity and not is_valid:
                    _bump("reject_invalid")
                    continue
                if require_two_stars and not is_two_star:
                    _bump("reject_two_star")
                    continue

                canonical_key = _canonicalize_smiles(text)
                if require_unique and canonical_key in seen_keys_for_view:
                    _bump("reject_duplicate")
                    continue

                is_novel = text not in training_set
                if require_novel and not is_novel:
                    _bump("reject_non_novel")
                    continue

                class_ok, matched_class = _match_polymer_class(text, target_classes, compiled_patterns)
                if target_classes and not class_ok:
                    _bump("reject_class")
                    continue

                sa_value = None
                if max_sa is not None:
                    sa_value = _compute_sa_score(text)
                    if sa_value is None or sa_value >= max_sa:
                        _bump("reject_sa")
                        continue

                prefilter_smiles.append(text)
                prefilter_meta.append(
                    {
                        "smiles": text,
                        "canonical_smiles": canonical_key,
                        "proposal_view": proposal_view,
                        "is_valid": bool(is_valid),
                        "is_two_star": bool(is_two_star),
                        "is_novel": bool(is_novel),
                        "class_match": bool(class_ok) if target_classes else True,
                        "matched_class": matched_class,
                        "sa_score": sa_value,
                        "sa_pass": bool(sa_value < max_sa) if (max_sa is not None and sa_value is not None) else True,
                        "batch_idx": batch_idx,
                    }
                )
                if require_unique:
                    seen_keys_for_view.add(canonical_key)

            if not prefilter_smiles:
                print(
                    f"[F5 resample] batch={batch_idx} view={proposal_view} generated={len(batch_smiles)} prefilter=0 "
                    f"accepted_view={accepted_by_view[proposal_view]}/{sampling_target} "
                    f"accepted_total={accepted_total}"
                )
                continue

            embeddings, kept_indices = _embed_candidates(
                view=scoring_view,
                smiles_list=prefilter_smiles,
                assets=scoring_assets,
                device=scoring_device,
            )
            stats["n_prefilter"] += len(prefilter_smiles)
            stats[f"n_prefilter_{proposal_view}"] += len(prefilter_smiles)
            if len(kept_indices) < len(prefilter_smiles):
                dropped = len(prefilter_smiles) - len(kept_indices)
                stats["reject_embed"] += dropped
                stats[f"reject_embed_{proposal_view}"] += dropped

            if embeddings.size == 0:
                print(
                    f"[F5 resample] batch={batch_idx} view={proposal_view} generated={len(batch_smiles)} "
                    f"prefilter={len(prefilter_smiles)} scored=0 accepted_view={accepted_by_view[proposal_view]}/{sampling_target} "
                    f"accepted_total={accepted_total}"
                )
                continue

            kept_meta = [prefilter_meta[idx] for idx in kept_indices]
            preds = np.asarray(property_model.predict(embeddings), dtype=np.float32).reshape(-1)
            hits = _compute_hits(preds, target_value, epsilon, target_mode)

            scored_this_batch = 0
            for row_idx, meta in enumerate(kept_meta):
                pred_value = float(preds[row_idx])
                hit = bool(hits[row_idx])
                excess_value = float(
                    _compute_target_excess(np.asarray([pred_value], dtype=np.float32), target_value, target_mode)[0]
                )
                violation_value = float(
                    _compute_target_violation(np.asarray([pred_value], dtype=np.float32), target_value, target_mode)[0]
                )
                record = {
                    **meta,
                    "prediction": pred_value,
                    "abs_error": abs(pred_value - target_value),
                    "target_excess": excess_value,
                    "target_violation": violation_value,
                    "property_hit": hit,
                    "accepted": False,
                }
                if hit and accepted_by_view[proposal_view] < sampling_target:
                    if per_view_min_hits > 0 and batch_idx <= per_view_quota_relax_after_batches:
                        underfilled = [v for v in proposal_views if accepted_by_view[v] < per_view_min_hits]
                        if underfilled and meta.get("proposal_view") not in underfilled:
                            stats["reject_view_quota"] += 1
                            stats[f"reject_view_quota_{proposal_view}"] += 1
                            hit = False

                if hit and accepted_by_view[proposal_view] < sampling_target:
                    record["accepted"] = True
                    accepted_total += 1
                    stats["n_hits"] += 1
                    stats[f"n_hits_{proposal_view}"] += 1
                    accepted_by_view[proposal_view] += 1
                scored_records.append(record)
                scored_this_batch += 1
                if accepted_by_view[proposal_view] >= sampling_target:
                    break

            stats["n_scored"] += scored_this_batch
            stats[f"n_scored_{proposal_view}"] += scored_this_batch
            print(
                f"[F5 resample] batch={batch_idx} view={proposal_view} generated={len(batch_smiles)} "
                f"prefilter={len(prefilter_smiles)} scored={scored_this_batch} "
                f"accepted_view={accepted_by_view[proposal_view]}/{sampling_target} "
                f"accepted_total={accepted_total}"
            )
            if accepted_by_view[proposal_view] >= sampling_target:
                pending_views = [v for v in pending_views if v != proposal_view]

        if not progress_made:
            stats["stop_no_proposal_views"] += 1
            print("[F5 resample] no proposal views remain; stopping early.")
            break

    if active_generator_assets is not None:
        try:
            active_generator_assets.get("backbone").to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scored_df = pd.DataFrame(scored_records)
    if scored_df.empty:
        scored_df = pd.DataFrame(
            columns=[
                "smiles",
                "prediction",
                "abs_error",
                "target_excess",
                "target_violation",
                "property_hit",
                "accepted",
            ]
        )
    accepted_df = scored_df[scored_df["accepted"] == True].copy()  # noqa: E712

    if len(accepted_df) == 0 and target_classes and stats.get("reject_class", 0) > 0:
        print(
            f"[F5 resample] warning: class filter rejected {int(stats.get('reject_class', 0))} "
            "candidates; consider relaxing foundation_inverse.target_class."
        )

    per_view_stats = {}
    tracked_keys = [
        "n_generated",
        "n_batches",
        "n_valid_any",
        "n_structural_valid",
        "n_prefilter",
        "n_scored",
        "n_hits",
        "reject_empty",
        "reject_invalid",
        "reject_two_star",
        "reject_duplicate",
        "reject_non_novel",
        "reject_class",
        "reject_sa",
        "reject_embed",
        "reject_view_quota",
    ]
    for view in proposal_views:
        row = {}
        for key in tracked_keys:
            row[key] = int(stats.get(f"{key}_{view}", 0))
        row["accepted"] = int(accepted_by_view.get(view, 0))
        per_view_stats[view] = row

    result = {
        "candidate_source": "resample",
        "generated_smiles": generated_smiles_all,
        "structurally_valid_smiles": structural_valid_smiles,
        "scored_df": scored_df,
        "accepted_df": accepted_df,
        "sampling_target": sampling_target,
        "sampling_target_total": int(sampling_target * len(proposal_views)),
        "sampling_num_per_batch": sampling_num_per_batch,
        "sampling_batch_size": sampling_batch_size,
        "sampling_max_batches": sampling_max_batches,
        "proposal_views": list(proposal_views),
        "proposal_views_remaining": list(pending_views),
        "proposal_views_disabled": dict(disabled_proposal_views),
        "scoring_view": scoring_view,
        "target_classes": target_classes,
        "require_validity": require_validity,
        "require_two_stars": require_two_stars,
        "require_novel": require_novel,
        "require_unique": require_unique,
        "max_sa": max_sa,
        "per_view_min_hits": per_view_min_hits,
        "per_view_quota_relax_after_batches": per_view_quota_relax_after_batches,
        "accepted_by_view": {k: int(v) for k, v in accepted_by_view.items()},
        "accepted_total": int(accepted_total),
        "completed_views": [view for view in proposal_views if int(accepted_by_view.get(view, 0)) >= sampling_target],
        "incomplete_views": [view for view in proposal_views if int(accepted_by_view.get(view, 0)) < sampling_target],
        "per_view_stats": per_view_stats,
        "stats": dict(stats),
        "completed": all(int(accepted_by_view.get(view, 0)) >= sampling_target for view in proposal_views),
    }
    return result


def _resolve_sampling_mode(args, f5_cfg: dict) -> str:
    mode = args.sampling_mode if getattr(args, "sampling_mode", None) is not None else f5_cfg.get("sampling_mode", "resample")
    mode = str(mode).strip().lower()
    if mode not in {"fixed_budget", "resample"}:
        raise ValueError("sampling_mode must be one of: fixed_budget|resample")
    return mode


def _candidate_scored_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    if "is_scored" in df.columns:
        return df["is_scored"].fillna(False).astype(bool)
    if "prediction" in df.columns:
        return pd.to_numeric(df["prediction"], errors="coerce").notna()
    return pd.Series([False] * len(df), index=df.index, dtype=bool)


def _sample_fixed_budget_candidates(
    *,
    config: dict,
    args,
    proposal_views: List[str],
    scoring_view: str,
    scoring_assets: dict,
    scoring_device: str,
    property_model,
    training_set: set,
    target_value: float,
    epsilon: float,
    target_mode: str,
) -> dict:
    f5_cfg = _f5_cfg(config)
    candidate_budget = int(
        args.candidate_budget_per_view
        if getattr(args, "candidate_budget_per_view", None) is not None
        else f5_cfg.get("candidate_budget_per_view", 10000)
    )
    sampling_num_per_batch = int(
        args.sampling_num_per_batch if args.sampling_num_per_batch is not None else f5_cfg.get("sampling_num_per_batch", 100)
    )
    sampling_batch_size = int(
        args.sampling_batch_size if args.sampling_batch_size is not None else f5_cfg.get("sampling_batch_size", 100)
    )
    sampling_max_batches = int(
        args.sampling_max_batches
        if args.sampling_max_batches is not None
        else f5_cfg.get("sampling_max_batches", max(4, math.ceil(candidate_budget / max(sampling_num_per_batch, 1)) + 2))
    )
    sampling_temperature = args.sampling_temperature if args.sampling_temperature is not None else f5_cfg.get("sampling_temperature", None)
    sampling_num_atoms = args.sampling_num_atoms if args.sampling_num_atoms is not None else f5_cfg.get("sampling_num_atoms", None)
    if sampling_num_atoms in ("", "none", None):
        sampling_num_atoms = None
    elif sampling_num_atoms is not None:
        sampling_num_atoms = int(sampling_num_atoms)
    if sampling_temperature in ("", "none", None):
        sampling_temperature = None
    elif sampling_temperature is not None:
        sampling_temperature = float(sampling_temperature)

    require_validity = _to_bool(f5_cfg.get("require_validity", True), True)
    require_two_stars = _to_bool(f5_cfg.get("require_two_stars", True), True)
    max_sa = args.max_sa if args.max_sa is not None else f5_cfg.get("max_sa", None)
    if max_sa in ("", "none", None):
        max_sa = None
    else:
        max_sa = float(max_sa)

    if candidate_budget <= 0:
        raise ValueError("candidate_budget_per_view must be > 0.")
    if sampling_num_per_batch <= 0 or sampling_batch_size <= 0:
        raise ValueError("sampling_num_per_batch and sampling_batch_size must both be > 0.")
    if sampling_max_batches <= 0:
        raise ValueError("sampling_max_batches must be > 0.")
    if not proposal_views:
        raise ValueError("proposal_views must be non-empty.")

    scoring_generator = _create_generator(
        view=scoring_view,
        assets=scoring_assets,
        device=scoring_device,
        sampling_temperature=sampling_temperature,
        sampling_num_atoms=sampling_num_atoms,
    )

    records: List[dict] = []
    structural_valid_smiles: List[str] = []
    generated_smiles_all: List[str] = []
    stats = Counter()
    proposal_view_failures: Dict[str, str] = {}

    active_generator_view: Optional[str] = None
    active_generator_assets: Optional[dict] = None
    active_generator: Optional[dict] = None

    def _get_proposal_generator(view: str) -> dict:
        nonlocal active_generator_view, active_generator_assets, active_generator
        if view == scoring_view:
            return scoring_generator
        if active_generator_view == view and active_generator is not None:
            return active_generator
        if active_generator_assets is not None:
            try:
                active_generator_assets.get("backbone").to("cpu")
            except Exception:
                pass
            active_generator_assets = None
            active_generator = None
            active_generator_view = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        proposal_device = _resolve_view_device(config, view)
        proposal_assets = _load_view_assets(config=config, view=view, device=proposal_device)
        proposal_generator = _create_generator(
            view=view,
            assets=proposal_assets,
            device=proposal_device,
            sampling_temperature=sampling_temperature,
            sampling_num_atoms=sampling_num_atoms,
        )
        active_generator_assets = proposal_assets
        active_generator = proposal_generator
        active_generator_view = view
        return proposal_generator

    for proposal_view in proposal_views:
        try:
            generator = _get_proposal_generator(proposal_view)
        except Exception as exc:
            proposal_view_failures[proposal_view] = str(exc)
            stats["proposal_view_init_failed"] += 1
            stats[f"proposal_view_init_failed_{proposal_view}"] += 1
            print(f"[F5 fixed_budget] skipping view={proposal_view} due to init error: {exc}")
            continue

        view_record_indices: List[int] = []
        view_structural_smiles: List[str] = []
        view_structural_record_indices: List[int] = []
        view_generated = 0
        batch_idx = 0

        while view_generated < candidate_budget and batch_idx < sampling_max_batches:
            request_n = min(sampling_num_per_batch, candidate_budget - view_generated)
            batch_idx += 1
            try:
                batch_records = _sample_batch_records_from_generator(generator, request_n, sampling_batch_size)
            except Exception as exc:
                proposal_view_failures[proposal_view] = str(exc)
                stats["proposal_view_sample_failed"] += 1
                stats[f"proposal_view_sample_failed_{proposal_view}"] += 1
                print(f"[F5 fixed_budget] disabling view={proposal_view} due to sample error: {exc}")
                break

            if not batch_records:
                stats["empty_batches"] += 1
                stats[f"empty_batches_{proposal_view}"] += 1
                continue

            stats["n_batches"] += 1
            stats[f"n_batches_{proposal_view}"] += 1
            view_generated += len(batch_records)
            stats["n_generated_raw"] += len(batch_records)
            stats[f"n_generated_raw_{proposal_view}"] += len(batch_records)

            for local_idx, sample in enumerate(batch_records):
                raw_output = str(sample.get("raw_output", "")).strip()
                smiles_text = str(sample.get("smiles", "")).strip()
                is_convertible = bool(sample.get("is_convertible", False) and smiles_text)
                if is_convertible:
                    stats["n_convertible"] += 1
                    stats[f"n_convertible_{proposal_view}"] += 1
                    generated_smiles_all.append(smiles_text)

                is_valid = bool(_check_validity(smiles_text)) if smiles_text else False
                is_two_star = bool(_count_stars(smiles_text) == 2) if smiles_text else False
                if is_valid:
                    stats["n_valid_any"] += 1
                    stats[f"n_valid_any_{proposal_view}"] += 1
                if is_valid and is_two_star:
                    stats["n_structural_valid"] += 1
                    stats[f"n_structural_valid_{proposal_view}"] += 1
                    structural_valid_smiles.append(smiles_text)
                    view_structural_smiles.append(smiles_text)

                canonical_smiles = _canonicalize_smiles(smiles_text) if smiles_text else ""
                is_novel = bool(smiles_text not in training_set) if smiles_text else False
                sa_score = _compute_sa_score(smiles_text) if (smiles_text and max_sa is not None) else None
                sa_pass = bool(sa_score is not None and sa_score < max_sa) if max_sa is not None else bool(smiles_text)

                record = {
                    "proposal_view": proposal_view,
                    "scoring_view": scoring_view,
                    "raw_output": raw_output,
                    "smiles": smiles_text,
                    "canonical_smiles": canonical_smiles,
                    "batch_idx": batch_idx,
                    "sample_index_within_view": int(view_generated - len(batch_records) + local_idx + 1),
                    "is_convertible": bool(is_convertible),
                    "is_valid": bool(is_valid),
                    "is_two_star": bool(is_two_star),
                    "is_structural_valid": bool(is_valid and is_two_star),
                    "is_novel": bool(is_novel),
                    "sa_score": sa_score,
                    "sa_pass": bool(sa_pass),
                    "is_scored": False,
                    "prediction": np.nan,
                    "abs_error": np.nan,
                    "target_excess": np.nan,
                    "target_violation": np.nan,
                    "property_hit": False,
                    "fair_hit": False,
                    "accepted": False,
                    "is_unique_within_view": False,
                }
                records.append(record)
                record_idx = len(records) - 1
                view_record_indices.append(record_idx)
                if record["is_structural_valid"]:
                    view_structural_record_indices.append(record_idx)

            print(
                f"[F5 fixed_budget] view={proposal_view} batch={batch_idx} "
                f"generated={view_generated}/{candidate_budget} structural_valid={len(view_structural_smiles)}"
            )

        if view_structural_smiles:
            embeddings, kept_indices = _embed_candidates(
                view=scoring_view,
                smiles_list=view_structural_smiles,
                assets=scoring_assets,
                device=scoring_device,
            )
            kept_index_set = set(int(i) for i in kept_indices)
            dropped = len(view_structural_smiles) - len(kept_indices)
            if dropped > 0:
                stats["reject_embed"] += dropped
                stats[f"reject_embed_{proposal_view}"] += dropped

            if embeddings.size > 0:
                preds = np.asarray(property_model.predict(embeddings), dtype=np.float32).reshape(-1)
                hits = _compute_hits(preds, target_value, epsilon, target_mode)
                stats["n_scored"] += len(kept_indices)
                stats[f"n_scored_{proposal_view}"] += len(kept_indices)
                stats["n_hits"] += int(np.sum(hits))
                stats[f"n_hits_{proposal_view}"] += int(np.sum(hits))
                for emb_idx, kept_local_idx in enumerate(kept_indices):
                    record_idx = view_structural_record_indices[int(kept_local_idx)]
                    pred_value = float(preds[emb_idx])
                    hit = bool(hits[emb_idx])
                    records[record_idx]["is_scored"] = True
                    records[record_idx]["prediction"] = pred_value
                    records[record_idx]["abs_error"] = abs(pred_value - target_value)
                    records[record_idx]["target_excess"] = float(
                        _compute_target_excess(np.asarray([pred_value], dtype=np.float32), target_value, target_mode)[0]
                    )
                    records[record_idx]["target_violation"] = float(
                        _compute_target_violation(np.asarray([pred_value], dtype=np.float32), target_value, target_mode)[0]
                    )
                    records[record_idx]["property_hit"] = hit
            for local_idx, record_idx in enumerate(view_structural_record_indices):
                if local_idx not in kept_index_set:
                    records[record_idx]["is_scored"] = False

    if active_generator_assets is not None:
        try:
            active_generator_assets.get("backbone").to("cpu")
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scored_df = pd.DataFrame(records)
    if scored_df.empty:
        scored_df = pd.DataFrame(
            columns=[
                "proposal_view",
                "smiles",
                "prediction",
                "target_excess",
                "target_violation",
                "property_hit",
                "fair_hit",
                "accepted",
            ]
        )

    per_view_stats = {}
    tracked_keys = [
        "n_generated_raw",
        "n_batches",
        "n_convertible",
        "n_valid_any",
        "n_structural_valid",
        "n_scored",
        "n_hits",
        "reject_embed",
    ]
    for view in proposal_views:
        row = {}
        for key in tracked_keys:
            row[key] = int(stats.get(f"{key}_{view}", 0))
        per_view_stats[view] = row

    return {
        "candidate_source": "fixed_budget",
        "generated_smiles": generated_smiles_all,
        "structurally_valid_smiles": structural_valid_smiles,
        "scored_df": scored_df,
        "accepted_df": pd.DataFrame(),
        "sampling_target": candidate_budget,
        "candidate_budget_per_view": candidate_budget,
        "sampling_num_per_batch": sampling_num_per_batch,
        "sampling_batch_size": sampling_batch_size,
        "sampling_max_batches": sampling_max_batches,
        "proposal_views": list(proposal_views),
        "proposal_views_remaining": list(proposal_views),
        "proposal_views_disabled": dict(proposal_view_failures),
        "scoring_view": scoring_view,
        "target_classes": [],
        "require_validity": require_validity,
        "require_two_stars": require_two_stars,
        "require_novel": False,
        "require_unique": False,
        "max_sa": max_sa,
        "per_view_min_hits": 0,
        "per_view_quota_relax_after_batches": 0,
        "accepted_by_view": {},
        "per_view_stats": per_view_stats,
        "stats": dict(stats),
        "completed": True,
    }


def _augment_f5_posthoc_flags(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df
    df = scored_df.copy()
    if "proposal_view" not in df.columns:
        df["proposal_view"] = "all"
    if "canonical_smiles" not in df.columns:
        df["canonical_smiles"] = df["smiles"].astype(str).map(_canonicalize_smiles)
    scored_mask = _candidate_scored_mask(df)
    df["is_unique_within_view"] = False
    for view, idx in df.loc[scored_mask].groupby("proposal_view").groups.items():
        subset = df.loc[list(idx), "canonical_smiles"].astype(str)
        duplicated = subset.duplicated(keep="first")
        df.loc[list(idx), "is_unique_within_view"] = ~duplicated.to_numpy(dtype=bool)
    if "is_valid" not in df.columns:
        df["is_valid"] = False
    if "is_two_star" not in df.columns:
        if "is_structural_valid" in df.columns:
            df["is_two_star"] = df["is_structural_valid"]
        else:
            df["is_two_star"] = False
    if "sa_pass" not in df.columns:
        df["sa_pass"] = False
    df["property_hit"] = df.get("property_hit", False).fillna(False).astype(bool)
    df["is_valid"] = df["is_valid"].fillna(False).astype(bool)
    df["is_two_star"] = df["is_two_star"].fillna(False).astype(bool)
    df["is_novel"] = df.get("is_novel", False).fillna(False).astype(bool)
    df["sa_pass"] = df["sa_pass"].fillna(False).astype(bool)
    df["fair_hit"] = (
        scored_mask
        & df["property_hit"]
        & df["is_valid"]
        & df["is_two_star"]
        & df["is_novel"]
        & df["is_unique_within_view"]
        & df["sa_pass"]
    )
    df["accepted"] = df["fair_hit"]
    return df


def _build_f5_process_counts(
    *,
    scored_df: pd.DataFrame,
    proposal_views: List[str],
    stats: dict,
) -> pd.DataFrame:
    rows: List[dict] = []
    df = scored_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["proposal_view", "stage", "count"])
    df["proposal_view"] = df.get("proposal_view", "all").astype(str)
    for view in ordered_views(proposal_views):
        view_df = df[df["proposal_view"] == view].copy()
        n_generated = int(stats.get(f"n_generated_raw_{view}", stats.get(f"n_generated_{view}", int(len(view_df)))))
        n_convertible = int(
            stats.get(
                f"n_convertible_{view}",
                stats.get(f"n_generated_{view}", int(view_df.get("is_convertible", pd.Series(dtype=bool)).sum())),
            )
        )
        n_valid = int(stats.get(f"n_valid_any_{view}", int(view_df.get("is_valid", pd.Series(dtype=bool)).sum())))
        n_two_star = int(stats.get(f"n_structural_valid_{view}", int(view_df.get("is_structural_valid", pd.Series(dtype=bool)).sum())))
        n_scored = int((_candidate_scored_mask(view_df)).sum())
        n_property_hits = int(view_df.get("property_hit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        n_fair_hits = int(view_df.get("fair_hit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        stage_counts = [
            ("generated_raw", n_generated),
            ("convertible", n_convertible),
            ("valid", n_valid),
            ("two_star", n_two_star),
            ("scored", n_scored),
            ("property_hit", n_property_hits),
            ("fair_hit", n_fair_hits),
        ]
        for stage, count in stage_counts:
            rate_vs_scored = float(count / max(n_scored, 1)) if n_scored > 0 else np.nan
            if stage in {"generated_raw", "convertible", "valid", "two_star"}:
                rate_vs_scored = np.nan
            rows.append(
                {
                    "proposal_view": view,
                    "stage": stage,
                    "count": int(count),
                    "rate_vs_generated": float(count / max(n_generated, 1)) if n_generated > 0 else np.nan,
                    "rate_vs_scored": rate_vs_scored,
                }
            )
    return pd.DataFrame(rows)


def _build_f5_metrics_rows(
    *,
    scored_df: pd.DataFrame,
    process_counts_df: pd.DataFrame,
    proposal_views: List[str],
    property_name: str,
    target_value: float,
    target_mode: str,
    epsilon: float,
    elapsed_sec: float,
    model_size: str,
    scoring_view: str,
    candidate_source: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    df = scored_df.copy()
    if "proposal_view" not in df.columns:
        df["proposal_view"] = "all"
    for view in ordered_views(proposal_views):
        view_df = df[df["proposal_view"] == view].copy()
        if process_counts_df.empty:
            continue
        process_view = process_counts_df[process_counts_df["proposal_view"] == view].copy()
        if process_view.empty:
            continue
        counts = {str(row["stage"]): int(row["count"]) for _, row in process_view.iterrows()}
        scored_mask = _candidate_scored_mask(view_df)
        scored_view = view_df.loc[scored_mask].copy()
        preds = pd.to_numeric(scored_view.get("prediction", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=np.float32)
        target_excess = pd.to_numeric(scored_view.get("target_excess", pd.Series(dtype=float)), errors="coerce")
        target_violation = pd.to_numeric(scored_view.get("target_violation", pd.Series(dtype=float)), errors="coerce")
        ood_prop = pd.to_numeric(scored_view.get("ood_prop", pd.Series(dtype=float)), errors="coerce")
        novelty = float(scored_view.get("is_novel", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if len(scored_view) else 0.0
        uniqueness = float(scored_view.get("is_unique_within_view", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if len(scored_view) else 0.0
        sa_pass_rate = float(scored_view.get("sa_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if len(scored_view) else 0.0
        property_hits = int(counts.get("property_hit", 0))
        fair_hits = int(counts.get("fair_hit", 0))
        n_generated = int(counts.get("generated_raw", 0))
        n_valid = int(counts.get("valid", 0))
        n_two_star = int(counts.get("two_star", 0))
        n_scored = int(counts.get("scored", 0))
        fair_hit_mask = scored_view.get("fair_hit", pd.Series([False] * len(scored_view), index=scored_view.index)).fillna(False).astype(bool)

        rows.append(
            {
                "method": "Multi_View_Foundation",
                "representation": _view_to_representation(view),
                "proposal_view": view,
                "scoring_view": scoring_view,
                "model_size": model_size,
                "property": property_name,
                "target_value": target_value,
                "target_mode": target_mode,
                "epsilon": epsilon,
                "candidate_source": candidate_source,
                "n_generated": n_generated,
                "n_generated_raw": n_generated,
                "n_convertible": int(counts.get("convertible", 0)),
                "n_valid": n_valid,
                "n_two_star": n_two_star,
                "n_scored": n_scored,
                "n_hits": property_hits,
                "n_fair_hits": fair_hits,
                "success_rate": round(float(property_hits / max(n_scored, 1)), 4) if n_scored else 0.0,
                "fair_success_rate": round(float(fair_hits / max(n_scored, 1)), 4) if n_scored else 0.0,
                "validity": round(float(n_valid / max(n_generated, 1)), 4) if n_generated else 0.0,
                "validity_two_stars": round(float(n_two_star / max(n_generated, 1)), 4) if n_generated else 0.0,
                "uniqueness": round(uniqueness, 4),
                "novelty": round(novelty, 4),
                "sa_pass_rate": round(sa_pass_rate, 4),
                "avg_diversity": _compute_pairwise_tanimoto_diversity(scored_view.loc[fair_hit_mask, "smiles"].astype(str).tolist()),
                "mean_target_excess": round(float(target_excess.mean()), 6) if target_excess.notna().any() else np.nan,
                "mean_target_violation": round(float(target_violation.mean()), 6) if target_violation.notna().any() else np.nan,
                "mean_ood_prop": round(float(ood_prop.mean()), 6) if ood_prop.notna().any() else np.nan,
                **_achievement_rates(preds, target_value),
                "sampling_time_sec": round(elapsed_sec, 2),
                "valid_per_compute": round(float(n_scored / max(elapsed_sec, 1e-9)), 4) if n_scored else 0.0,
                "hits_per_compute": round(float(property_hits / max(elapsed_sec, 1e-9)), 4) if property_hits else 0.0,
            }
        )

    if not rows:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows)
    total_generated = int(metrics_df["n_generated"].sum())
    total_scored = int(metrics_df["n_scored"].sum())
    total_hits = int(metrics_df["n_hits"].sum())
    total_fair_hits = int(metrics_df["n_fair_hits"].sum())
    aggregate_row = {
        "method": "Multi_View_Foundation",
        "representation": "All_Views",
        "proposal_view": "all",
        "scoring_view": scoring_view,
        "model_size": model_size,
        "property": property_name,
        "target_value": target_value,
        "target_mode": target_mode,
        "epsilon": epsilon,
        "candidate_source": candidate_source,
        "n_generated": total_generated,
        "n_generated_raw": int(metrics_df["n_generated_raw"].sum()) if "n_generated_raw" in metrics_df.columns else total_generated,
        "n_convertible": int(metrics_df["n_convertible"].sum()) if "n_convertible" in metrics_df.columns else np.nan,
        "n_valid": int(metrics_df["n_valid"].sum()),
        "n_two_star": int(metrics_df["n_two_star"].sum()) if "n_two_star" in metrics_df.columns else np.nan,
        "n_scored": total_scored,
        "n_hits": total_hits,
        "n_fair_hits": total_fair_hits,
        "success_rate": round(float(total_hits / max(total_scored, 1)), 4) if total_scored else 0.0,
        "fair_success_rate": round(float(total_fair_hits / max(total_scored, 1)), 4) if total_scored else 0.0,
        "validity": round(float(metrics_df["n_valid"].sum() / max(total_generated, 1)), 4) if total_generated else 0.0,
        "validity_two_stars": round(float(metrics_df["n_two_star"].sum() / max(total_generated, 1)), 4) if total_generated and "n_two_star" in metrics_df.columns else np.nan,
        "uniqueness": round(float(metrics_df["uniqueness"].mean()), 4) if "uniqueness" in metrics_df.columns else np.nan,
        "novelty": round(float(metrics_df["novelty"].mean()), 4) if "novelty" in metrics_df.columns else np.nan,
        "sa_pass_rate": round(float(metrics_df["sa_pass_rate"].mean()), 4) if "sa_pass_rate" in metrics_df.columns else np.nan,
        "avg_diversity": round(float(metrics_df["avg_diversity"].mean()), 4) if "avg_diversity" in metrics_df.columns and metrics_df["avg_diversity"].notna().any() else np.nan,
        "mean_target_excess": round(float(metrics_df["mean_target_excess"].mean()), 6) if "mean_target_excess" in metrics_df.columns and metrics_df["mean_target_excess"].notna().any() else np.nan,
        "mean_target_violation": round(float(metrics_df["mean_target_violation"].mean()), 6) if "mean_target_violation" in metrics_df.columns and metrics_df["mean_target_violation"].notna().any() else np.nan,
        "mean_ood_prop": round(float(metrics_df["mean_ood_prop"].mean()), 6) if "mean_ood_prop" in metrics_df.columns and metrics_df["mean_ood_prop"].notna().any() else np.nan,
        "sampling_time_sec": round(elapsed_sec, 2),
        "valid_per_compute": round(float(total_scored / max(elapsed_sec, 1e-9)), 4) if total_scored else 0.0,
        "hits_per_compute": round(float(total_hits / max(elapsed_sec, 1e-9)), 4) if total_hits else 0.0,
    }
    for key, value in _achievement_rates(
        pd.to_numeric(
            scored_df.loc[_candidate_scored_mask(scored_df), "prediction"] if "prediction" in scored_df.columns else pd.Series(dtype=float),
            errors="coerce",
        ).dropna().to_numpy(dtype=np.float32),
        target_value,
    ).items():
        aggregate_row[key] = value
    return pd.concat([metrics_df, pd.DataFrame([aggregate_row])], ignore_index=True)


def _plot_f5_design_process(
    *,
    process_counts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    property_name: str,
    figures_dir: Path,
) -> None:
    if plt is None or process_counts_df.empty or metrics_df.empty:
        return
    property_label = property_display_name(property_name)
    stages = [
        "generated_raw",
        "convertible",
        "valid",
        "two_star",
        "scored",
        "property_hit",
        "fair_hit",
    ]
    stage_labels = {
        "generated_raw": "Generated",
        "convertible": "Convertible",
        "valid": "Valid",
        "two_star": "Two-star",
        "scored": "Scored",
        "property_hit": "Directional hit",
        "fair_hit": "Fair hit",
    }
    metric_rows = metrics_df[metrics_df["proposal_view"].astype(str) != "all"].copy()
    process_rows = process_counts_df[process_counts_df["stage"].isin(stages)].copy()
    views = ordered_views(process_rows["proposal_view"].tolist())
    if not views:
        return

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.6))
    ax0, ax1, ax2 = axes
    x = np.arange(len(stages), dtype=np.int64)

    for view in views:
        sub = process_rows[process_rows["proposal_view"] == view].copy()
        counts = [int(sub.loc[sub["stage"] == stage, "count"].iloc[0]) if (sub["stage"] == stage).any() else 0 for stage in stages]
        ax0.plot(
            x,
            counts,
            marker="o",
            linewidth=2.2,
            markersize=6.0,
            color=view_color(view),
            label=view_label(view),
        )
        rate_vals = [
            float(sub.loc[sub["stage"] == stage, "rate_vs_generated"].iloc[0]) if (sub["stage"] == stage).any() else np.nan
            for stage in stages
        ]
        ax1.plot(
            x,
            rate_vals,
            marker="o",
            linewidth=2.2,
            markersize=6.0,
            color=view_color(view),
            label=view_label(view),
        )

    ax0.set_xticks(x)
    ax0.set_xticklabels([stage_labels[stage] for stage in stages], rotation=30, ha="right")
    ax0.set_ylabel("Candidate count")
    ax0.set_title("Design process counts")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(loc="best")

    ax1.set_xticks(x)
    ax1.set_xticklabels([stage_labels[stage] for stage in stages], rotation=30, ha="right")
    ax1.set_ylabel("Fraction of generated")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("Normalized design process")
    ax1.grid(axis="y", alpha=0.25)

    summary = metric_rows.copy()
    if not summary.empty:
        xpos = np.arange(len(summary), dtype=np.int64)
        ax2.bar(
            xpos,
            summary["fair_success_rate"].to_numpy(dtype=float),
            width=0.6,
            color=[view_color(v) for v in summary["proposal_view"].tolist()],
            alpha=0.82,
            label="F5 fair hit rate",
        )
        ax2.set_xticks(xpos)
        ax2.set_xticklabels([view_label(v) for v in summary["proposal_view"].tolist()], rotation=20, ha="right")
        ax2.set_ylabel("Hit rate")
        ax2.set_ylim(0.0, max(1.0, float(summary["fair_success_rate"].max() + 0.05)))
        ax2.set_title("Per-view benchmark hit rates")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(loc="best")
    else:
        ax2.text(0.5, 0.5, "No per-view metrics", ha="center", va="center")
        ax2.set_axis_off()

    fig.suptitle(f"F5 Benchmark Design Process: {property_label}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save_figure_png(fig, figures_dir / f"figure_f5_design_process_{property_name}")
    plt.close(fig)


def main(args):
    args.property = _normalize_property_name(args.property)
    config = load_config(args.config)
    f5_cfg = _f5_cfg(config)
    prop_cfg = config.get("property", {}) if isinstance(config.get("property", {}), dict) else {}
    include_multiview_mean = _to_bool(prop_cfg.get("enable_multiview_mean_head", False), False)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step5_foundation_inverse")
    property_step_dirs = ensure_step_dirs(results_dir, "step5_foundation_inverse", args.property)
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")
    save_config(config, property_step_dirs["files_dir"] / "config_used.yaml")

    encoder_view = _select_encoder_view(config, args.encoder_view)
    proposal_views = _select_proposal_views(config, args.proposal_views, fallback_view=encoder_view)
    committee_properties = _resolve_committee_properties(config, args.property, args.committee_properties)

    encoder_device = _resolve_view_device(config, encoder_view)
    assets = _load_view_assets(config=config, view=encoder_view, device=encoder_device)

    property_model_mode = args.property_model_mode
    if property_model_mode is None:
        property_model_mode = str(f5_cfg.get("property_model_mode", "single"))
    property_model_mode = str(property_model_mode).strip().lower()
    if property_model_mode not in {"single", "all"}:
        raise ValueError("property_model_mode must be one of: single|all")
    if property_model_mode == "all" and args.property_model_path:
        raise ValueError("--property_model_path is only supported when property_model_mode=single")

    model_path = args.property_model_path
    if model_path is None:
        model_path = _default_property_model_path(results_dir, args.property, encoder_view)
    else:
        model_path = _resolve_path(model_path)

    if not Path(model_path).exists():
        discovered = _discover_property_model_paths(
            results_dir,
            args.property,
            include_multiview_mean=include_multiview_mean,
        )
        fallback = discovered.get(encoder_view) or discovered.get("smiles")
        if fallback is not None and Path(fallback).exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Property model not found: {model_path}")

    model = _load_property_model(Path(model_path))
    target_value = _resolve_target_value(args.target, args.property, f5_cfg)
    target_mode = _resolve_target_mode(args.target_mode, args.property, f5_cfg)
    epsilon = _resolve_epsilon(args.epsilon, args.property, f5_cfg)
    sampling_mode = _resolve_sampling_mode(args, f5_cfg)
    train_smiles_path = _resolve_path(config["paths"]["polymer_file"])
    training_set = _load_training_smiles(train_smiles_path)

    t0 = time.time()
    source_meta = {"candidate_source": sampling_mode, "sampling_mode": sampling_mode}
    if sampling_mode == "fixed_budget":
        sampled = _sample_fixed_budget_candidates(
            config=config,
            args=args,
            proposal_views=proposal_views,
            scoring_view=encoder_view,
            scoring_assets=assets,
            scoring_device=encoder_device,
            property_model=model,
            training_set=training_set,
            target_value=target_value,
            epsilon=epsilon,
            target_mode=target_mode,
        )
    else:
        sampled = _resample_candidates_until_target(
            config=config,
            args=args,
            proposal_views=proposal_views,
            scoring_view=encoder_view,
            scoring_assets=assets,
            scoring_device=encoder_device,
            property_model=model,
            training_set=training_set,
            target_value=target_value,
            epsilon=epsilon,
            target_mode=target_mode,
        )
    smiles_list = sampled["generated_smiles"]
    structurally_valid_smiles = sampled["structurally_valid_smiles"]
    scored_df = sampled["scored_df"]
    source_meta.update(
        {
            "sampling_target": int(sampled["sampling_target"]),
            "sampling_target_total": int(sampled.get("sampling_target_total", sampled["sampling_target"])),
            "sampling_num_per_batch": int(sampled["sampling_num_per_batch"]),
            "sampling_batch_size": int(sampled["sampling_batch_size"]),
            "sampling_max_batches": int(sampled["sampling_max_batches"]),
            "proposal_views": sampled["proposal_views"],
            "proposal_views_remaining": sampled.get("proposal_views_remaining", []),
            "proposal_views_disabled": sampled.get("proposal_views_disabled", {}),
            "scoring_view": sampled["scoring_view"],
            "target_classes": sampled["target_classes"],
            "require_validity": bool(sampled["require_validity"]),
            "require_two_stars": bool(sampled["require_two_stars"]),
            "require_novel": bool(sampled["require_novel"]),
            "require_unique": bool(sampled["require_unique"]),
            "max_sa": sampled["max_sa"],
            "per_view_min_hits": int(sampled.get("per_view_min_hits", 0)),
            "per_view_quota_relax_after_batches": int(sampled.get("per_view_quota_relax_after_batches", 0)),
            "accepted_by_view": sampled.get("accepted_by_view", {}),
            "accepted_total": int(sampled.get("accepted_total", 0)),
            "completed_views": sampled.get("completed_views", []),
            "incomplete_views": sampled.get("incomplete_views", []),
            "per_view_stats": sampled.get("per_view_stats", {}),
            "stats": sampled["stats"],
            "sampling_completed": bool(sampled["completed"]),
        }
    )

    # Optional all-model aggregation:
    # score with all available F3 property heads across committee properties,
    # then use target-property committee mean/std for acceptance.
    if property_model_mode == "all" and not scored_df.empty:
        committee_models_used: Dict[str, Dict[str, str]] = {}
        committee_model_order: Dict[str, List[str]] = {}
        missing_properties: List[str] = []
        smiles_for_committee = scored_df["smiles"].astype(str).tolist()
        for committee_prop in committee_properties:
            all_model_paths = _discover_property_model_paths(
                results_dir,
                committee_prop,
                include_multiview_mean=include_multiview_mean,
            )
            if committee_prop == args.property and encoder_view not in all_model_paths and Path(model_path).exists():
                # Keep backward compatibility for primary scoring model location.
                all_model_paths[encoder_view] = Path(model_path)
            if not all_model_paths:
                if committee_prop == args.property:
                    raise FileNotFoundError(
                        f"No F3 property models found for target property={args.property} under {results_dir / 'step3_property'}"
                    )
                missing_properties.append(committee_prop)
                continue

            pred_pack = _build_prediction_columns_from_all_models(
                config=config,
                results_dir=results_dir,
                smiles_list=smiles_for_committee,
                model_paths=all_model_paths,
            )
            if not pred_pack["prediction_by_model"]:
                if committee_prop == args.property:
                    raise RuntimeError(
                        f"Failed to compute committee predictions for target property={args.property}."
                    )
                missing_properties.append(committee_prop)
                continue

            for model_name, pred_values in pred_pack["prediction_by_model"].items():
                scored_df[f"pred_{committee_prop}_{model_name}"] = pred_values
                if committee_prop == args.property:
                    # Preserve legacy target-property column names.
                    scored_df[f"prediction_{model_name}"] = pred_values

            scored_df[f"pred_{committee_prop}_mean"] = np.asarray(pred_pack["prediction_ensemble"], dtype=np.float32).reshape(-1)
            scored_df[f"pred_{committee_prop}_std"] = np.asarray(pred_pack["prediction_std"], dtype=np.float32).reshape(-1)
            scored_df[f"pred_{committee_prop}_n_models"] = np.asarray(pred_pack["prediction_valid_count"], dtype=np.int64).reshape(-1)
            committee_models_used[committee_prop] = pred_pack["models_used"]
            committee_model_order[committee_prop] = [str(x) for x in pred_pack.get("model_order", [])]

        target_mean_col = f"pred_{args.property}_mean"
        target_std_col = f"pred_{args.property}_std"
        target_count_col = f"pred_{args.property}_n_models"
        if target_mean_col not in scored_df.columns:
            raise RuntimeError(
                f"Missing target committee prediction column '{target_mean_col}'. "
                "Ensure step3 models exist for the target property."
            )

        ensemble_pred = pd.to_numeric(scored_df[target_mean_col], errors="coerce").to_numpy(dtype=np.float32)
        scored_df["prediction"] = ensemble_pred
        if target_std_col in scored_df.columns:
            scored_df["prediction_std"] = pd.to_numeric(scored_df[target_std_col], errors="coerce")
        else:
            scored_df["prediction_std"] = np.nan
        if target_count_col in scored_df.columns:
            scored_df["prediction_n_models"] = pd.to_numeric(scored_df[target_count_col], errors="coerce")
        else:
            scored_df["prediction_n_models"] = np.nan

        scored_df["abs_error"] = np.abs(ensemble_pred - target_value)
        scored_df["target_excess"] = _compute_target_excess(ensemble_pred, target_value, target_mode)
        scored_df["target_violation"] = _compute_target_violation(ensemble_pred, target_value, target_mode)
        valid_pred = np.isfinite(ensemble_pred)
        hit_mask = _compute_hits(ensemble_pred, target_value, epsilon, target_mode)
        hit_mask = np.logical_and(hit_mask, valid_pred)
        scored_df["property_hit"] = hit_mask
        scored_df["accepted"] = hit_mask

        source_meta.update(
            {
                "property_model_mode": "all",
                "prediction_aggregation": "mean",
                "include_multiview_mean_head": bool(include_multiview_mean),
                "committee_properties_requested": committee_properties,
                "committee_properties_scored": sorted(committee_models_used.keys()),
                "committee_properties_missing": missing_properties,
                "property_models_used": committee_models_used,
                "property_model_order": committee_model_order,
                "target_prediction_column": target_mean_col,
                "target_uncertainty_column": target_std_col,
            }
        )
    elif property_model_mode == "all":
        source_meta.update(
            {
                "property_model_mode": "all",
                "prediction_aggregation": "mean",
                "include_multiview_mean_head": bool(include_multiview_mean),
                "committee_properties_requested": committee_properties,
                "committee_properties_scored": [],
                "committee_properties_missing": committee_properties,
                "property_models_used": {},
                "property_model_order": {},
            }
        )
    else:
        source_meta.update(
            {
                "property_model_mode": "single",
                "include_multiview_mean_head": bool(include_multiview_mean),
                "property_models_used": {encoder_view: str(Path(model_path))},
            }
        )

    elapsed_sec = time.time() - t0
    if scored_df.empty:
        scored_df = pd.DataFrame(
            columns=[
                "smiles",
                "prediction",
                "abs_error",
                "target_excess",
                "target_violation",
                "property_hit",
                "accepted",
            ]
        )

    scored_mask = _candidate_scored_mask(scored_df)
    if "property" not in scored_df.columns:
        scored_df = scored_df.copy()
    scored_df["property"] = args.property
    scored_df = _augment_f5_posthoc_flags(scored_df)
    process_counts_df = _build_f5_process_counts(
        scored_df=scored_df,
        proposal_views=proposal_views,
        stats=source_meta.get("stats", {}),
    )
    view_compare_top_k = _to_int_or_none(args.top_k)
    if view_compare_top_k is None:
        view_compare_top_k = _to_int_or_none(f5_cfg.get("top_k"))
    if view_compare_top_k is None:
        view_compare_top_k = 100
    view_compare_top_k = max(int(view_compare_top_k), 1)
    metrics_df = _build_f5_metrics_rows(
        scored_df=scored_df,
        process_counts_df=process_counts_df,
        proposal_views=proposal_views,
        property_name=args.property,
        target_value=target_value,
        target_mode=target_mode,
        epsilon=epsilon,
        elapsed_sec=elapsed_sec,
        model_size=str(assets["model_size"]),
        scoring_view=encoder_view,
        candidate_source=str(source_meta.get("candidate_source", sampling_mode)),
    )
    aggregate_metrics = metrics_df[metrics_df["proposal_view"].astype(str) == "all"].copy()
    if aggregate_metrics.empty:
        aggregate_metrics = pd.DataFrame(
            [
                {
                    "proposal_view": "all",
                    "n_generated": len(smiles_list),
                    "n_valid": len(structurally_valid_smiles),
                    "n_scored": int(scored_mask.sum()),
                    "n_hits": int(scored_df.get("property_hit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                    "n_fair_hits": int(scored_df.get("fair_hit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                }
            ]
        )
    aggregate_row = aggregate_metrics.iloc[0].to_dict()

    save_csv(
        metrics_df,
        property_step_dirs["metrics_dir"] / "metrics_inverse.csv",
        legacy_paths=[
            step_dirs["metrics_dir"] / "metrics_inverse.csv",
            results_dir / "metrics_inverse.csv",
        ],
        index=False,
    )

    out_dir = step_dirs["step_dir"]
    files_dir = property_step_dirs["files_dir"]
    legacy_files_dir = step_dirs["files_dir"]
    accepted_df = scored_df[scored_df["accepted"] == True].copy() if "accepted" in scored_df.columns else scored_df.iloc[0:0].copy()  # noqa: E712
    if "property" not in accepted_df.columns:
        accepted_df = accepted_df.copy()
    accepted_df["property"] = args.property

    save_csv(
        process_counts_df,
        files_dir / f"design_process_counts_{args.property}.csv",
        legacy_paths=[
            legacy_files_dir / f"design_process_counts_{args.property}.csv",
            out_dir / f"design_process_counts_{args.property}.csv",
        ],
        index=False,
    )
    save_csv(
        process_counts_df,
        files_dir / "design_process_counts.csv",
        legacy_paths=[
            legacy_files_dir / "design_process_counts.csv",
            out_dir / "design_process_counts.csv",
        ],
        index=False,
    )

    save_csv(
        scored_df,
        files_dir / "candidate_scores.csv",
        legacy_paths=[
            legacy_files_dir / "candidate_scores.csv",
            out_dir / "candidate_scores.csv",
        ],
        index=False,
    )
    save_csv(
        scored_df,
        files_dir / f"candidate_scores_{args.property}.csv",
        legacy_paths=[
            legacy_files_dir / f"candidate_scores_{args.property}.csv",
            out_dir / f"candidate_scores_{args.property}.csv",
        ],
        index=False,
    )

    view_compare_outputs = None
    try:
        view_compare_outputs = analyze_view_compare(
            candidate_df=scored_df,
            property_name=args.property,
            target=float(target_value),
            target_mode=target_mode,
            epsilon=float(epsilon),
            top_k=int(view_compare_top_k),
            model_size=str(assets["model_size"]),
            scoring_view=encoder_view,
        )
        save_view_compare_outputs(
            analysis=view_compare_outputs,
            property_name=args.property,
            property_step_dirs=property_step_dirs,
            root_step_dirs=step_dirs,
            root_legacy_dirs=[out_dir],
        )
    except Exception as exc:
        print(f"[F5] Warning: view comparison export skipped for {args.property}: {exc}")
    save_csv(
        accepted_df,
        files_dir / "accepted_candidates.csv",
        legacy_paths=[
            legacy_files_dir / "accepted_candidates.csv",
            out_dir / "accepted_candidates.csv",
        ],
        index=False,
    )
    save_csv(
        accepted_df,
        files_dir / f"accepted_candidates_{args.property}.csv",
        legacy_paths=[
            legacy_files_dir / f"accepted_candidates_{args.property}.csv",
            out_dir / f"accepted_candidates_{args.property}.csv",
        ],
        index=False,
    )

    report_df, report_summary = _build_f5_accepted_polymer_report(
        accepted_df=accepted_df,
        property_name=args.property,
        target_value=target_value,
        target_mode=target_mode,
        epsilon=epsilon,
        sampling_target=int(source_meta.get("sampling_target", len(accepted_df))),
        sampling_target_total=int(source_meta.get("sampling_target_total", source_meta.get("sampling_target", len(accepted_df)))),
    )
    save_csv(
        report_df,
        files_dir / "accepted_polymer_report.csv",
        legacy_paths=[
            legacy_files_dir / "accepted_polymer_report.csv",
            out_dir / "accepted_polymer_report.csv",
        ],
        index=False,
    )
    save_csv(
        report_df,
        files_dir / f"accepted_polymer_report_{args.property}.csv",
        legacy_paths=[
            legacy_files_dir / f"accepted_polymer_report_{args.property}.csv",
            out_dir / f"accepted_polymer_report_{args.property}.csv",
        ],
        index=False,
    )
    save_json(
        report_summary,
        files_dir / "accepted_polymer_summary.json",
        legacy_paths=[
            legacy_files_dir / "accepted_polymer_summary.json",
            out_dir / "accepted_polymer_summary.json",
        ],
    )
    save_json(
        report_summary,
        files_dir / f"accepted_polymer_summary_{args.property}.json",
        legacy_paths=[
            legacy_files_dir / f"accepted_polymer_summary_{args.property}.json",
            out_dir / f"accepted_polymer_summary_{args.property}.json",
        ],
    )

    save_json(
        {
            **source_meta,
            "encoder_view": encoder_view,
            "proposal_views": proposal_views,
            "property": args.property,
            "target_value": target_value,
            "target_mode": target_mode,
            "epsilon": epsilon,
            "sampling_mode": sampling_mode,
            "view_compare_top_k": int(view_compare_top_k),
            "n_generated": int(aggregate_row.get("n_generated", 0)),
            "n_structurally_valid": int(aggregate_row.get("n_valid", 0)),
            "n_scored": int(aggregate_row.get("n_scored", 0)),
            "n_hits": int(aggregate_row.get("n_hits", 0)),
            "n_fair_hits": int(aggregate_row.get("n_fair_hits", 0)),
        },
        files_dir / "run_meta.json",
        legacy_paths=[
            legacy_files_dir / "run_meta.json",
            out_dir / "run_meta.json",
        ],
    )
    save_json(
        {
            **source_meta,
            "encoder_view": encoder_view,
            "proposal_views": proposal_views,
            "property": args.property,
            "target_value": target_value,
            "target_mode": target_mode,
            "epsilon": epsilon,
            "sampling_mode": sampling_mode,
            "view_compare_top_k": int(view_compare_top_k),
            "n_generated": int(aggregate_row.get("n_generated", 0)),
            "n_structurally_valid": int(aggregate_row.get("n_valid", 0)),
            "n_scored": int(aggregate_row.get("n_scored", 0)),
            "n_hits": int(aggregate_row.get("n_hits", 0)),
            "n_fair_hits": int(aggregate_row.get("n_fair_hits", 0)),
        },
        files_dir / f"run_meta_{args.property}.json",
        legacy_paths=[
            legacy_files_dir / f"run_meta_{args.property}.json",
            out_dir / f"run_meta_{args.property}.json",
        ],
    )

    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(f5_cfg.get("generate_figures", True), True)
    if generate_figures and plt is None:
        print("Warning: matplotlib unavailable; skipping F5 figures.")
        generate_figures = False
    if generate_figures:
        _plot_f5_diagnostics(
            scored_df=scored_df,
            property_name=args.property,
            target_value=target_value,
            target_mode=target_mode,
            epsilon=epsilon,
            source_meta=source_meta,
            figures_dir=property_step_dirs["figures_dir"],
        )
        _plot_f5_design_process(
            process_counts_df=process_counts_df,
            metrics_df=metrics_df,
            property_name=args.property,
            figures_dir=property_step_dirs["figures_dir"],
        )
        if view_compare_outputs is not None:
            plot_view_compare(
                analysis=view_compare_outputs,
                property_name=args.property,
                figures_dir=property_step_dirs["figures_dir"],
                figure_prefix="figure_f5_view_compare",
            )
        _plot_f5_accepted_polymer_overview(
            report_df=report_df,
            property_name=args.property,
            target_value=target_value,
            target_mode=target_mode,
            epsilon=epsilon,
            figures_dir=property_step_dirs["figures_dir"],
        )
        _plot_f5_accepted_polymer_gallery(
            report_df=report_df,
            property_name=args.property,
            figures_dir=property_step_dirs["figures_dir"],
            top_n=min(36, int(source_meta.get("sampling_target", 100))),
        )

    print(f"Saved metrics_inverse.csv to {property_step_dirs['step_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--encoder_view", type=str, default=None, choices=list(SUPPORTED_VIEWS))
    parser.add_argument("--proposal_views", type=str, default=None, help="Comma list or 'all' for F5 proposal generation views.")
    parser.add_argument("--property", type=str, required=True)
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--target_mode", type=str, default=None, choices=["window", "ge", "le"])
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--sampling_mode", type=str, default=None, choices=["fixed_budget", "resample"])
    parser.add_argument("--candidate_budget_per_view", type=int, default=None, help="Legacy fixed_budget-only candidate cap per view.")
    parser.add_argument("--sampling_target", type=int, default=None, help="Accepted-polymer target per proposal view in resample mode.")
    parser.add_argument("--sampling_num_per_batch", type=int, default=None)
    parser.add_argument("--sampling_batch_size", type=int, default=None)
    parser.add_argument("--sampling_max_batches", type=int, default=None)
    parser.add_argument("--sampling_temperature", type=float, default=None)
    parser.add_argument("--sampling_num_atoms", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None, help="Per-view top-k exported by F5 view comparison analysis.")
    parser.add_argument("--target_class", type=str, default=None)
    parser.add_argument("--max_sa", type=float, default=None)
    parser.add_argument(
        "--per_view_min_hits",
        type=int,
        default=None,
        help="Minimum accepted hits per proposal view during the initial balancing phase.",
    )
    parser.add_argument(
        "--per_view_quota_relax_after_batches",
        type=int,
        default=None,
        help="Batch index after which per-view minimum-hit quota is relaxed (0=auto).",
    )
    parser.add_argument("--property_model_mode", type=str, default=None, choices=["single", "all"])
    parser.add_argument("--committee_properties", type=str, default=None, help="Comma list for committee property exports; defaults to property.files.")
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    main(parser.parse_args())
