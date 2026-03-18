"""Shared per-view benchmark analysis for MVF F5."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.foundation_inverse import compute_property_hits
from src.utils.output_layout import save_csv, save_json
from src.utils.property_names import property_display_name
from src.utils.visualization import (
    normalize_view_name,
    ordered_views,
    save_figure_png,
    set_publication_style,
    view_color,
    view_label,
)

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:  # pragma: no cover
    from rdkit import Chem, rdBase
    from rdkit.Chem import Descriptors, rdMolDescriptors

    rdBase.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None
    rdMolDescriptors = None


POLYMER_MOTIF_SMARTS = {
    "polyimide": "[#6][CX3](=[OX1])[NX3][CX3](=[OX1])[#6]",
    "polyester": "[#6][CX3](=[OX1])[OX2][#6]",
    "polyamide": "[#6][CX3](=[OX1])[NX3;!$([N]([C](=O))[C](=O))][#6;!$([CX3](=[OX1]))]",
    "polyurethane": "[#6][OX2][CX3](=[OX1])[NX3][#6]",
    "polyether": "[#6;!$([CX3](=[OX1]))][OX2][#6;!$([CX3](=[OX1]))]",
    "polysiloxane": "[Si][OX2][Si]",
    "polycarbonate": "[#6][OX2][CX3](=[OX1])[OX2][#6]",
    "polysulfone": "[#6][SX4](=[OX1])(=[OX1])[#6]",
    "polyacrylate": "[#6]-[#6](=O)-[#8]",
    "polystyrene": "[#6]-[#6](c1ccccc1)-[#6]",
}

DESCRIPTOR_COLUMNS = [
    "mol_wt",
    "heavy_atoms",
    "ring_count",
    "aromatic_ring_count",
    "fraction_csp3",
    "rotatable_bonds",
    "tpsa",
    "logp",
    "sa_score",
]

if plt is not None:
    set_publication_style()


def _to_float_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid float value.")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _save_figure_png(fig, output_base: Path) -> None:
    save_figure_png(fig, output_base, font_size=16, dpi=600, legend_loc="best")


def _candidate_scored_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    if "is_scored" in df.columns:
        return df["is_scored"].fillna(False).astype(bool)
    return pd.to_numeric(df.get("prediction", pd.Series(dtype=float)), errors="coerce").notna()


def _canonicalize_smiles(smiles: str) -> str:
    text = str(smiles).strip()
    if not text:
        return ""
    if Chem is None:
        return text
    try:
        mol = Chem.MolFromSmiles(text)
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else text
    except Exception:
        return text


def _compute_target_excess(preds: np.ndarray, target: float, target_mode: str) -> np.ndarray:
    arr = np.asarray(preds, dtype=np.float32).reshape(-1)
    mode = str(target_mode).strip().lower()
    if mode == "le":
        return (float(target) - arr).astype(np.float32, copy=False)
    return (arr - float(target)).astype(np.float32, copy=False)


def _compute_target_violation(preds: np.ndarray, target: float, target_mode: str) -> np.ndarray:
    arr = np.asarray(preds, dtype=np.float32).reshape(-1)
    mode = str(target_mode).strip().lower()
    if mode == "ge":
        return np.maximum(0.0, float(target) - arr).astype(np.float32, copy=False)
    if mode == "le":
        return np.maximum(0.0, arr - float(target)).astype(np.float32, copy=False)
    return np.abs(arr - float(target)).astype(np.float32, copy=False)


def _match_polymer_class(smiles: str) -> str:
    text = str(smiles).strip()
    if not text or Chem is None:
        return "unknown"
    try:
        mol = Chem.MolFromSmiles(text)
    except Exception:
        mol = None
    if mol is None:
        return "unknown"
    for label, smarts in POLYMER_MOTIF_SMARTS.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
        except Exception:
            patt = None
        if patt is not None and mol.HasSubstructMatch(patt):
            return label
    return "other"


def _descriptor_row(smiles: str, sa_score: Any = np.nan) -> dict:
    row = {name: np.nan for name in DESCRIPTOR_COLUMNS}
    row["sa_score"] = _to_float_or_none(sa_score)
    text = str(smiles).strip()
    if not text or Chem is None or Descriptors is None or rdMolDescriptors is None:
        return row
    try:
        mol = Chem.MolFromSmiles(text)
    except Exception:
        mol = None
    if mol is None:
        return row
    try:
        row["mol_wt"] = float(Descriptors.MolWt(mol))
        row["heavy_atoms"] = float(mol.GetNumHeavyAtoms())
        row["ring_count"] = float(rdMolDescriptors.CalcNumRings(mol))
        row["aromatic_ring_count"] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
        row["fraction_csp3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
        row["rotatable_bonds"] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        row["tpsa"] = float(rdMolDescriptors.CalcTPSA(mol))
        row["logp"] = float(Descriptors.MolLogP(mol))
    except Exception:
        pass
    return row


def _build_descriptor_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for idx, row in df.iterrows():
        desc = _descriptor_row(row.get("smiles", ""), row.get("sa_score", np.nan))
        desc["proposal_view"] = normalize_view_name(row.get("proposal_view", ""))
        desc["smiles"] = str(row.get("smiles", ""))
        desc["index"] = idx
        rows.append(desc)
    return pd.DataFrame(rows)


def _compute_space_coordinates(descriptor_df: pd.DataFrame) -> pd.DataFrame:
    if descriptor_df.empty:
        return descriptor_df.copy()
    cols = [col for col in DESCRIPTOR_COLUMNS if col in descriptor_df.columns]
    out = descriptor_df.copy()
    out["pc1"] = np.nan
    out["pc2"] = np.nan
    if not cols:
        return out
    matrix = descriptor_df[cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = matrix.notna().all(axis=1)
    if int(valid_mask.sum()) < 2:
        return out
    x = matrix.loc[valid_mask].to_numpy(dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x = (x - mean) / std
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comp = x @ vt[:2].T
    out.loc[valid_mask, "pc1"] = comp[:, 0]
    out.loc[valid_mask, "pc2"] = comp[:, 1] if comp.shape[1] > 1 else 0.0
    return out


def _ensure_analysis_columns(df: pd.DataFrame, *, target: float, target_mode: str, epsilon: float) -> pd.DataFrame:
    data = df.copy()
    if "proposal_view" not in data.columns:
        data["proposal_view"] = "all"
    data["proposal_view"] = data["proposal_view"].astype(str).map(lambda x: normalize_view_name(x) or "all")
    if "canonical_smiles" not in data.columns:
        data["canonical_smiles"] = data["smiles"].astype(str).map(_canonicalize_smiles)
    scored_mask = _candidate_scored_mask(data)
    preds = pd.to_numeric(data.get("prediction", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float32)
    if "property_hit" not in data.columns:
        hits = compute_property_hits(preds, target_value=float(target), epsilon=float(epsilon), target_mode=str(target_mode))
        data["property_hit"] = np.logical_and(np.isfinite(preds), hits)
    else:
        data["property_hit"] = data["property_hit"].fillna(False).astype(bool)
    if "target_excess" not in data.columns:
        data["target_excess"] = _compute_target_excess(preds, target=float(target), target_mode=target_mode)
    if "target_violation" not in data.columns:
        data["target_violation"] = _compute_target_violation(preds, target=float(target), target_mode=target_mode)
    if "is_valid" not in data.columns:
        data["is_valid"] = data["smiles"].astype(str).map(lambda x: bool(Chem.MolFromSmiles(x)) if Chem is not None and x else False)
    if "is_two_star" not in data.columns:
        data["is_two_star"] = data["smiles"].astype(str).map(lambda x: int(x.count("*")) == 2)
    if "is_novel" not in data.columns:
        data["is_novel"] = False
    if "sa_pass" not in data.columns:
        data["sa_pass"] = False
    data["is_valid"] = data["is_valid"].fillna(False).astype(bool)
    data["is_two_star"] = data["is_two_star"].fillna(False).astype(bool)
    data["is_novel"] = data["is_novel"].fillna(False).astype(bool)
    data["sa_pass"] = data["sa_pass"].fillna(False).astype(bool)
    data["is_unique_within_view"] = False
    for _, idx in data.loc[scored_mask].groupby("proposal_view").groups.items():
        subset = data.loc[list(idx), "canonical_smiles"].astype(str)
        duplicated = subset.duplicated(keep="first")
        data.loc[list(idx), "is_unique_within_view"] = ~duplicated.to_numpy(dtype=bool)
    data["fair_hit"] = (
        scored_mask
        & data["property_hit"]
        & data["is_valid"]
        & data["is_two_star"]
        & data["is_novel"]
        & data["is_unique_within_view"]
        & data["sa_pass"]
    )
    if "polymer_class" not in data.columns:
        data["polymer_class"] = np.nan
    missing_class = data["polymer_class"].isna()
    if bool(missing_class.any()):
        data.loc[missing_class, "polymer_class"] = data.loc[missing_class, "smiles"].astype(str).map(_match_polymer_class)
    return data


def _rank_within_view(valid_df: pd.DataFrame, *, target_mode: str, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = valid_df.copy()
    df["property_rank_within_view"] = np.nan
    df["topk_selected_within_view"] = False
    top_frames: List[pd.DataFrame] = []
    mode = str(target_mode).strip().lower()
    for view in ordered_views(df["proposal_view"].tolist()):
        view_df = df[df["proposal_view"] == view].copy()
        if view_df.empty:
            continue
        if mode in {"ge", "le"}:
            sort_df = pd.DataFrame(
                {
                    "target_violation": pd.to_numeric(view_df.get("target_violation"), errors="coerce").fillna(np.inf),
                    "target_excess": pd.to_numeric(view_df.get("target_excess"), errors="coerce").fillna(-np.inf),
                    "abs_error": pd.to_numeric(view_df.get("abs_error"), errors="coerce").fillna(np.inf),
                },
                index=view_df.index,
            ).sort_values(
                ["target_violation", "target_excess", "abs_error"],
                ascending=[True, False, True],
                kind="mergesort",
            )
        else:
            sort_df = pd.DataFrame(
                {
                    "target_violation": pd.to_numeric(view_df.get("target_violation"), errors="coerce").fillna(np.inf),
                    "abs_error": pd.to_numeric(view_df.get("abs_error"), errors="coerce").fillna(np.inf),
                },
                index=view_df.index,
            ).sort_values(["target_violation", "abs_error"], ascending=[True, True], kind="mergesort")
        ranked_idx = sort_df.index.to_numpy()
        df.loc[ranked_idx, "property_rank_within_view"] = np.arange(1, len(ranked_idx) + 1, dtype=np.int64)
        keep_n = min(int(top_k), len(ranked_idx))
        top_idx = ranked_idx[:keep_n]
        df.loc[top_idx, "topk_selected_within_view"] = True
        top_frames.append(df.loc[top_idx].copy())
    top_df = pd.concat(top_frames, ignore_index=False).copy() if top_frames else df.iloc[0:0].copy()
    return df, top_df


def _metrics_rows(
    valid_df: pd.DataFrame,
    top_df: pd.DataFrame,
    *,
    property_name: str,
    target: float,
    target_mode: str,
    epsilon: float,
    top_k: int,
    model_size: str,
    scoring_view: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    for view in ordered_views(valid_df["proposal_view"].tolist()):
        view_df = valid_df[valid_df["proposal_view"] == view].copy()
        top_view = top_df[top_df["proposal_view"] == view].copy()
        n_scored = int(len(view_df))
        if n_scored == 0:
            continue
        n_hits = int(view_df["property_hit"].sum())
        n_fair = int(view_df["fair_hit"].sum())
        k_sel = int(len(top_view))
        rows.append(
            {
                "method": "Multi_View_Foundation",
                "representation": view_label(view),
                "proposal_view": view,
                "scoring_view": scoring_view,
                "model_size": model_size,
                "property": property_name,
                "target_value": float(target),
                "target_mode": target_mode,
                "epsilon": float(epsilon),
                "top_k": int(k_sel),
                "top_k_requested": int(top_k),
                "n_candidates_scored": n_scored,
                "n_hits": n_hits,
                "n_fair_hits": n_fair,
                "success_rate": round(float(n_hits / max(n_scored, 1)), 4),
                "fair_success_rate": round(float(n_fair / max(n_scored, 1)), 4),
                "top_k_hits": int(top_view["property_hit"].sum()) if k_sel else 0,
                "top_k_fair_hits": int(top_view["fair_hit"].sum()) if k_sel else 0,
                "top_k_hit_rate": round(float(top_view["property_hit"].mean()), 4) if k_sel else 0.0,
                "top_k_fair_hit_rate": round(float(top_view["fair_hit"].mean()), 4) if k_sel else 0.0,
                "mean_prediction": round(float(pd.to_numeric(view_df["prediction"], errors="coerce").mean()), 6),
                "median_prediction": round(float(pd.to_numeric(view_df["prediction"], errors="coerce").median()), 6),
                "mean_target_excess": round(float(pd.to_numeric(view_df["target_excess"], errors="coerce").mean()), 6),
                "mean_target_violation": round(float(pd.to_numeric(view_df["target_violation"], errors="coerce").mean()), 6),
                "novelty": round(float(view_df["is_novel"].mean()), 4),
                "uniqueness": round(float(view_df["is_unique_within_view"].mean()), 4),
                "sa_pass_rate": round(float(view_df["sa_pass"].mean()), 4),
                "validity": round(float(view_df["is_valid"].mean()), 4),
                "validity_two_stars": round(float(view_df["is_two_star"].mean()), 4),
            }
        )
    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df
    all_row = {
        "method": "Multi_View_Foundation",
        "representation": "All Views",
        "proposal_view": "all",
        "scoring_view": scoring_view,
        "model_size": model_size,
        "property": property_name,
        "target_value": float(target),
        "target_mode": target_mode,
        "epsilon": float(epsilon),
        "top_k": int(metrics_df["top_k"].sum()),
        "top_k_requested": int(top_k),
        "n_candidates_scored": int(metrics_df["n_candidates_scored"].sum()),
        "n_hits": int(metrics_df["n_hits"].sum()),
        "n_fair_hits": int(metrics_df["n_fair_hits"].sum()),
        "success_rate": round(float(metrics_df["n_hits"].sum() / max(metrics_df["n_candidates_scored"].sum(), 1)), 4),
        "fair_success_rate": round(float(metrics_df["n_fair_hits"].sum() / max(metrics_df["n_candidates_scored"].sum(), 1)), 4),
        "top_k_hits": int(metrics_df["top_k_hits"].sum()),
        "top_k_fair_hits": int(metrics_df["top_k_fair_hits"].sum()),
        "top_k_hit_rate": round(float(metrics_df["top_k_hits"].sum() / max(metrics_df["top_k"].sum(), 1)), 4),
        "top_k_fair_hit_rate": round(float(metrics_df["top_k_fair_hits"].sum() / max(metrics_df["top_k"].sum(), 1)), 4),
        "mean_prediction": round(float(metrics_df["mean_prediction"].mean()), 6),
        "median_prediction": round(float(metrics_df["median_prediction"].mean()), 6),
        "mean_target_excess": round(float(metrics_df["mean_target_excess"].mean()), 6),
        "mean_target_violation": round(float(metrics_df["mean_target_violation"].mean()), 6),
        "novelty": round(float(metrics_df["novelty"].mean()), 4),
        "uniqueness": round(float(metrics_df["uniqueness"].mean()), 4),
        "sa_pass_rate": round(float(metrics_df["sa_pass_rate"].mean()), 4),
        "validity": round(float(metrics_df["validity"].mean()), 4),
        "validity_two_stars": round(float(metrics_df["validity_two_stars"].mean()), 4),
    }
    return pd.concat([metrics_df, pd.DataFrame([all_row])], ignore_index=True)


def _descriptor_summary(valid_df: pd.DataFrame, top_df: pd.DataFrame) -> pd.DataFrame:
    frames: List[dict] = []
    for subset_name, subset_df in [("scored", valid_df), ("top_k", top_df)]:
        desc = _build_descriptor_frame(subset_df)
        if desc.empty:
            continue
        for view in ordered_views(desc["proposal_view"].tolist()):
            view_desc = desc[desc["proposal_view"] == view]
            if view_desc.empty:
                continue
            row = {"subset": subset_name, "proposal_view": view}
            for col in DESCRIPTOR_COLUMNS:
                values = pd.to_numeric(view_desc[col], errors="coerce")
                row[f"{col}_mean"] = round(float(values.mean()), 6) if values.notna().any() else np.nan
                row[f"{col}_median"] = round(float(values.median()), 6) if values.notna().any() else np.nan
            frames.append(row)
    return pd.DataFrame(frames)


def _class_distribution(top_df: pd.DataFrame) -> pd.DataFrame:
    if top_df.empty:
        return pd.DataFrame(columns=["proposal_view", "polymer_class", "count", "fraction"])
    rows: List[dict] = []
    for view in ordered_views(top_df["proposal_view"].tolist()):
        sub = top_df[top_df["proposal_view"] == view].copy()
        total = len(sub)
        counts = sub["polymer_class"].fillna("unknown").astype(str).value_counts()
        for cls, count in counts.items():
            rows.append(
                {
                    "proposal_view": view,
                    "polymer_class": cls,
                    "count": int(count),
                    "fraction": round(float(count / max(total, 1)), 6),
                }
            )
    return pd.DataFrame(rows)


def analyze_view_compare(
    *,
    candidate_df: pd.DataFrame,
    property_name: str,
    target: float,
    target_mode: str,
    epsilon: float,
    top_k: int,
    model_size: str,
    scoring_view: str,
) -> Dict[str, pd.DataFrame]:
    df = _ensure_analysis_columns(candidate_df, target=float(target), target_mode=target_mode, epsilon=float(epsilon))
    valid_df = df.loc[_candidate_scored_mask(df)].copy()
    if valid_df.empty:
        raise RuntimeError("No scored candidates available for view comparison analysis.")
    valid_df, top_df = _rank_within_view(valid_df, target_mode=target_mode, top_k=top_k)
    valid_df["property"] = property_name
    top_df["property"] = property_name
    metrics_df = _metrics_rows(
        valid_df,
        top_df,
        property_name=property_name,
        target=float(target),
        target_mode=target_mode,
        epsilon=float(epsilon),
        top_k=int(top_k),
        model_size=model_size,
        scoring_view=scoring_view,
    )
    descriptor_summary_df = _descriptor_summary(valid_df, top_df)
    class_df = _class_distribution(top_df)
    space_df = _compute_space_coordinates(_build_descriptor_frame(top_df if len(top_df) else valid_df))
    return {
        "valid_df": valid_df,
        "top_df": top_df,
        "metrics_df": metrics_df,
        "descriptor_summary_df": descriptor_summary_df,
        "class_df": class_df,
        "space_df": space_df,
    }


def save_view_compare_outputs(
    *,
    analysis: Dict[str, pd.DataFrame],
    property_name: str,
    property_step_dirs: dict,
    root_step_dirs: dict,
    root_legacy_dirs: Optional[List[Path]] = None,
) -> None:
    valid_df = analysis["valid_df"]
    top_df = analysis["top_df"]
    metrics_df = analysis["metrics_df"]
    descriptor_summary_df = analysis["descriptor_summary_df"]
    class_df = analysis["class_df"]
    space_df = analysis["space_df"]
    legacy_files = [root_step_dirs["files_dir"]]
    legacy_metrics = [root_step_dirs["metrics_dir"]]
    if root_legacy_dirs:
        legacy_files.extend(root_legacy_dirs)
        legacy_metrics.extend(root_legacy_dirs)
    save_csv(
        valid_df.sort_values(["proposal_view", "property_rank_within_view"], kind="mergesort"),
        property_step_dirs["files_dir"] / "view_compare_scores.csv",
        legacy_paths=[path / "view_compare_scores.csv" for path in legacy_files],
        index=False,
    )
    save_csv(
        valid_df.sort_values(["proposal_view", "property_rank_within_view"], kind="mergesort"),
        property_step_dirs["files_dir"] / f"view_compare_scores_{property_name}.csv",
        legacy_paths=[path / f"view_compare_scores_{property_name}.csv" for path in legacy_files],
        index=False,
    )
    save_csv(
        top_df.sort_values(["proposal_view", "property_rank_within_view"], kind="mergesort"),
        property_step_dirs["files_dir"] / "view_compare_topk.csv",
        legacy_paths=[path / "view_compare_topk.csv" for path in legacy_files],
        index=False,
    )
    save_csv(
        top_df.sort_values(["proposal_view", "property_rank_within_view"], kind="mergesort"),
        property_step_dirs["files_dir"] / f"view_compare_topk_{property_name}.csv",
        legacy_paths=[path / f"view_compare_topk_{property_name}.csv" for path in legacy_files],
        index=False,
    )
    save_csv(
        metrics_df,
        property_step_dirs["metrics_dir"] / "metrics_view_compare.csv",
        legacy_paths=[path / "metrics_view_compare.csv" for path in legacy_metrics],
        index=False,
    )
    save_csv(
        metrics_df,
        property_step_dirs["metrics_dir"] / f"metrics_view_compare_{property_name}.csv",
        legacy_paths=[path / f"metrics_view_compare_{property_name}.csv" for path in legacy_metrics],
        index=False,
    )
    save_csv(
        descriptor_summary_df,
        property_step_dirs["files_dir"] / f"view_compare_descriptor_summary_{property_name}.csv",
        legacy_paths=[path / f"view_compare_descriptor_summary_{property_name}.csv" for path in legacy_files],
        index=False,
    )
    save_csv(
        class_df,
        property_step_dirs["files_dir"] / f"view_compare_class_distribution_{property_name}.csv",
        legacy_paths=[path / f"view_compare_class_distribution_{property_name}.csv" for path in legacy_files],
        index=False,
    )
    save_csv(
        space_df,
        property_step_dirs["files_dir"] / f"view_compare_space_{property_name}.csv",
        legacy_paths=[path / f"view_compare_space_{property_name}.csv" for path in legacy_files],
        index=False,
    )
    run_meta = {
        "property": property_name,
        "analysis_mode": "property_view_compare",
        "selection_mode": "property_only",
        "proposal_views": ordered_views(valid_df["proposal_view"].tolist()),
        "n_scored": int(len(valid_df)),
        "n_topk": int(len(top_df)),
    }
    save_json(
        run_meta,
        property_step_dirs["files_dir"] / "view_compare_run_meta.json",
        legacy_paths=[path / "view_compare_run_meta.json" for path in legacy_files],
    )


def plot_view_compare(
    *,
    analysis: Dict[str, pd.DataFrame],
    property_name: str,
    figures_dir: Path,
    figure_prefix: str = "figure_f5_view_compare",
) -> None:
    if plt is None:
        return
    valid_df = analysis["valid_df"]
    top_df = analysis["top_df"]
    metrics_df = analysis["metrics_df"]
    class_df = analysis["class_df"]
    if metrics_df.empty:
        return
    views = ordered_views(metrics_df.loc[metrics_df["proposal_view"] != "all", "proposal_view"].tolist())
    if not views:
        return
    metric_rows = metrics_df[metrics_df["proposal_view"] != "all"].set_index("proposal_view").reindex(views).reset_index()
    colors = [view_color(v) for v in metric_rows["proposal_view"].tolist()]

    fig, axes = plt.subplots(2, 2, figsize=(18.0, 13.0))
    ax0, ax1, ax2, ax3 = axes.reshape(-1)
    xpos = np.arange(len(metric_rows), dtype=np.int64)

    ax0.bar(xpos, metric_rows["fair_success_rate"].to_numpy(dtype=float), color=colors, alpha=0.85)
    ax0.set_xticks(xpos)
    ax0.set_xticklabels([view_label(v) for v in metric_rows["proposal_view"].tolist()], rotation=20, ha="right")
    ax0.set_ylabel("Fair hit rate")
    ax0.set_ylim(0.0, max(1.0, float(metric_rows["fair_success_rate"].max() + 0.05)))
    ax0.grid(axis="y", alpha=0.25)

    box_data = []
    box_colors = []
    box_labels = []
    for view in views:
        vals = pd.to_numeric(valid_df.loc[valid_df["proposal_view"] == view, "prediction"], errors="coerce").dropna()
        if vals.empty:
            continue
        box_data.append(vals.to_numpy(dtype=np.float32))
        box_colors.append(view_color(view))
        box_labels.append(view_label(view))
    if box_data:
        box = ax1.boxplot(box_data, patch_artist=True, labels=box_labels)
        for patch, color in zip(box["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax1.tick_params(axis="x", rotation=20)
        ax1.set_ylabel(f"Predicted {property_display_name(property_name)}")
        ax1.grid(axis="y", alpha=0.25)
    else:
        ax1.text(0.5, 0.5, "No finite predictions", ha="center", va="center")
        ax1.set_axis_off()

    space_source = top_df if len(top_df) >= 5 else valid_df
    space_df = _compute_space_coordinates(_build_descriptor_frame(space_source))
    if not space_df.empty and space_df["pc1"].notna().any() and space_df["pc2"].notna().any():
        for view in views:
            sub = space_df[space_df["proposal_view"] == view]
            if sub.empty:
                continue
            ax2.scatter(sub["pc1"], sub["pc2"], s=36, alpha=0.75, color=view_color(view), label=view_label(view))
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="best")
    else:
        ax2.text(0.5, 0.5, "Chemical-space PCA unavailable", ha="center", va="center")
        ax2.set_axis_off()

    if not class_df.empty:
        class_order = sorted(class_df["polymer_class"].dropna().astype(str).unique().tolist())
        bottom = np.zeros((len(views),), dtype=np.float32)
        for cls in class_order:
            values = []
            for view in views:
                match = class_df[(class_df["proposal_view"] == view) & (class_df["polymer_class"] == cls)]
                values.append(float(match["fraction"].iloc[0]) if not match.empty else 0.0)
            vals = np.asarray(values, dtype=np.float32)
            ax3.bar(xpos, vals, bottom=bottom, alpha=0.85, label=cls)
            bottom += vals
        ax3.set_xticks(xpos)
        ax3.set_xticklabels([view_label(v) for v in views], rotation=20, ha="right")
        ax3.set_ylabel("Top-k class fraction")
        ax3.set_ylim(0.0, 1.0)
        ax3.grid(axis="y", alpha=0.25)
        ax3.legend(loc="best", fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No class composition available", ha="center", va="center")
        ax3.set_axis_off()

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / f"{figure_prefix}_{property_name}")
    plt.close(fig)
