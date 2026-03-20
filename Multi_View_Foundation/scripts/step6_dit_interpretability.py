#!/usr/bin/env python
"""F6: Multi-method interpretability of the shared SMILES DiT property scorer."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.utils.foundation_assets import (
    TorchPropertyPredictor,
    default_property_model_path,
    load_property_model,
    load_view_assets,
    resolve_path as foundation_resolve_path,
    resolve_view_device,
)
from src.utils.output_layout import ensure_step_dirs, save_csv, save_json
from src.utils.property_names import (
    normalize_property_name as shared_normalize_property_name,
    property_column_candidates,
    property_display_name,
)
from src.utils.runtime import to_bool as _to_bool, to_int_or_none as _to_int_or_none
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

if plt is not None:
    set_publication_style()


OUTCOME_GROUPS = ("fair_hit", "property_hit_only", "near_miss")
OUTCOME_LABELS = {
    "fair_hit": "Fair hit",
    "property_hit_only": "Property-only hit",
    "near_miss": "Near miss",
}
METHODS_DEFAULT = ("gradient_x_hidden", "integrated_gradients", "attention_rollout")
METHOD_LABELS = {
    "gradient_x_hidden": "Grad x Hidden",
    "integrated_gradients": "Integrated Gradients",
    "attention_rollout": "Attention Rollout",
}


def _normalize_property_name(value) -> str:
    return shared_normalize_property_name(value)


def _dit_cfg(config: dict) -> dict:
    return dict(config.get("dit_interpretability", {}) or {})


def _parse_methods(value: Optional[object]) -> List[str]:
    if value is None:
        raw = list(METHODS_DEFAULT)
    elif isinstance(value, str):
        raw = [x.strip() for x in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(x).strip() for x in value]
    else:
        raw = [str(value).strip()]
    methods: List[str] = []
    valid = set(METHODS_DEFAULT)
    for item in raw:
        if not item:
            continue
        key = str(item).strip().lower()
        if key not in valid:
            raise ValueError(f"Unsupported interpretability method '{item}'. Supported: {sorted(valid)}")
        if key not in methods:
            methods.append(key)
    return methods or list(METHODS_DEFAULT)


def _save_figure(fig, output_base: Path) -> None:
    save_figure_png(fig, output_base, font_size=16, dpi=600, legend_loc="best")


def _resolve_view_compare_scores_path(results_dir: Path, property_name: str) -> Path:
    candidates: List[Path] = []
    aliases = [_normalize_property_name(property_name), *property_column_candidates(property_name)]
    for alias in aliases:
        if not alias:
            continue
        candidates.extend(
            [
                results_dir / "step5_foundation_inverse" / alias / "files" / f"view_compare_scores_{alias}.csv",
                results_dir / "step5_foundation_inverse" / alias / "files" / "view_compare_scores.csv",
                results_dir / "step5_foundation_inverse" / "files" / f"view_compare_scores_{alias}.csv",
                results_dir / "step5_foundation_inverse" / f"view_compare_scores_{alias}.csv",
                results_dir / "step5_foundation_inverse" / alias / "files" / f"candidate_scores_{alias}.csv",
                results_dir / "step5_foundation_inverse" / alias / "files" / "candidate_scores.csv",
                results_dir / "step5_foundation_inverse" / "files" / f"candidate_scores_{alias}.csv",
                results_dir / "step5_foundation_inverse" / "files" / "candidate_scores.csv",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Interpretability input not found for property={property_name}. searched={[str(p) for p in candidates]}")


def _pool_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return hidden_states[:, 0]
    if pooling == "max":
        mask = attention_mask.unsqueeze(-1).float()
        masked = hidden_states * mask + (1.0 - mask) * (-1e9)
        return masked.max(dim=1)[0]
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def _token_strings(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> List[str]:
    ids = input_ids.detach().cpu().tolist()
    mask = attention_mask.detach().cpu().tolist()
    tokens = []
    for token_id, keep in zip(ids, mask):
        if int(keep) != 1:
            continue
        tokens.append(str(getattr(tokenizer, "id_to_token", {}).get(int(token_id), "[UNK]")))
    return tokens


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _normalized_entropy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    probs = np.clip(arr / total, 1e-12, 1.0)
    return float((-np.sum(probs * np.log(probs))) / math.log(len(probs)))


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    rank_a = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    rank_b = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(rank_a) < 1e-12 or np.std(rank_b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _normalize_token_scores(scores: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.float()
    scores = scores * mask
    denom = scores.sum().clamp(min=1e-9)
    return scores / denom


def _select_cases(df: pd.DataFrame, *, max_samples_per_group: int) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    data = df.copy()
    data["proposal_view"] = data["proposal_view"].astype(str).map(lambda x: normalize_view_name(x) or "all")
    data["property_rank_within_view"] = pd.to_numeric(data.get("property_rank_within_view"), errors="coerce")
    data["target_violation"] = pd.to_numeric(data.get("target_violation"), errors="coerce")
    data["abs_error"] = pd.to_numeric(data.get("abs_error"), errors="coerce")
    for view in ordered_views(data["proposal_view"].tolist()):
        view_df = data[data["proposal_view"] == view].copy()
        if view_df.empty:
            continue
        selectors = {
            "fair_hit": view_df["fair_hit"].fillna(False).astype(bool),
            "property_hit_only": view_df["property_hit"].fillna(False).astype(bool) & ~view_df["fair_hit"].fillna(False).astype(bool),
            "near_miss": ~view_df["property_hit"].fillna(False).astype(bool),
        }
        for outcome_group, mask in selectors.items():
            sub = view_df.loc[mask].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(
                ["target_violation", "property_rank_within_view", "abs_error"],
                ascending=[True, True, True],
                kind="mergesort",
            ).head(int(max_samples_per_group))
            sub["outcome_group"] = outcome_group
            rows.append(sub)
    if not rows:
        return data.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def _load_property_scorer(config: dict, results_dir: Path, property_name: str, encoder_view: str, property_model_path: Optional[str]):
    device = resolve_view_device(config, encoder_view)
    assets = load_view_assets(config=config, view=encoder_view, device=device)
    model_path = Path(property_model_path) if property_model_path else default_property_model_path(results_dir, property_name, encoder_view)
    if not model_path.is_absolute():
        model_path = foundation_resolve_path(str(model_path))
    predictor = load_property_model(model_path)
    if not isinstance(predictor, TorchPropertyPredictor):
        raise TypeError(f"F6 interpretability requires MVF torch MLP checkpoint, got {type(predictor).__name__}")
    predictor.apply_backbone_to_assets(assets)
    predictor.model.to(device)
    predictor.model.eval()
    mean = torch.tensor(predictor.mean, dtype=torch.float32, device=device)
    scale = torch.tensor(predictor.scale, dtype=torch.float32, device=device)
    tokenizer = assets["tokenizer"]
    pad_id = int(getattr(tokenizer, "pad_token_id"))
    mask_id = int(getattr(tokenizer, "mask_token_id"))
    bos_id = int(getattr(tokenizer, "bos_token_id"))
    eos_id = int(getattr(tokenizer, "eos_token_id"))
    return {
        "assets": assets,
        "predictor": predictor,
        "device": device,
        "mean": mean,
        "scale": scale,
        "model_path": model_path,
        "pad_id": pad_id,
        "mask_id": mask_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
    }


def _encode_case(smiles: str, scorer_pack: dict) -> dict:
    assets = scorer_pack["assets"]
    tokenizer = assets["tokenizer"]
    encoded = tokenizer.encode(str(smiles), add_special_tokens=True, padding=True, return_attention_mask=True)
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=scorer_pack["device"])
    attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=scorer_pack["device"])
    timesteps = torch.full((1,), int(assets["timestep"]), dtype=torch.long, device=scorer_pack["device"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "timesteps": timesteps,
        "tokens": _token_strings(input_ids.squeeze(0), attention_mask.squeeze(0), tokenizer),
    }


def _forward_from_hidden(hidden: torch.Tensor, scorer_pack: dict, attention_mask: torch.Tensor) -> torch.Tensor:
    pooled = _pool_hidden(hidden, attention_mask, scorer_pack["assets"]["pooling"])
    scaled = (pooled - scorer_pack["mean"]) / scorer_pack["scale"]
    return scorer_pack["predictor"].model(scaled).reshape(-1)


def _forward_from_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor, timesteps: torch.Tensor, scorer_pack: dict) -> torch.Tensor:
    hidden = scorer_pack["assets"]["backbone"].get_hidden_states(input_ids, timesteps, attention_mask)
    return _forward_from_hidden(hidden, scorer_pack, attention_mask)


def _forward_from_token_embeddings(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    timesteps: torch.Tensor,
    scorer_pack: dict,
) -> torch.Tensor:
    backbone = scorer_pack["assets"]["backbone"]
    batch_size, seq_len, _ = token_embeddings.shape
    positions = torch.arange(seq_len, device=token_embeddings.device).unsqueeze(0).expand(batch_size, -1)
    pos_emb = backbone.position_embedding(positions)
    if timesteps.dim() == 2:
        timesteps = timesteps.squeeze(-1)
    time_emb = backbone.timestep_embedding(timesteps).unsqueeze(1)
    x = token_embeddings + pos_emb + time_emb
    x = backbone.embedding_dropout(x)
    for layer in backbone.layers:
        x = layer(x, attention_mask)
    x = backbone.final_norm(x)
    return _forward_from_hidden(x, scorer_pack, attention_mask)


def _valid_token_positions(case_pack: dict, scorer_pack: dict) -> List[int]:
    mask = case_pack["attention_mask"].squeeze(0).detach().cpu().numpy().astype(np.int64)
    ids = case_pack["input_ids"].squeeze(0).detach().cpu().numpy().astype(np.int64)
    positions: List[int] = []
    for pos, keep in enumerate(mask.tolist()):
        if keep != 1:
            continue
        token_id = int(ids[pos])
        if token_id in {scorer_pack["pad_id"], scorer_pack["bos_id"], scorer_pack["eos_id"]}:
            continue
        positions.append(pos)
    return positions


def _method_gradient_x_hidden(case_pack: dict, scorer_pack: dict) -> dict:
    backbone = scorer_pack["assets"]["backbone"]
    predictor = scorer_pack["predictor"]
    backbone.zero_grad(set_to_none=True)
    predictor.model.zero_grad(set_to_none=True)
    hidden = backbone.get_hidden_states(case_pack["input_ids"], case_pack["timesteps"], case_pack["attention_mask"])
    hidden.retain_grad()
    pred = _forward_from_hidden(hidden, scorer_pack, case_pack["attention_mask"])
    pred.sum().backward()
    scores = (hidden.grad * hidden).sum(dim=-1).abs().squeeze(0)
    scores_norm = _normalize_token_scores(scores, case_pack["attention_mask"].squeeze(0))
    return {
        "prediction": float(pred.detach().cpu().numpy().reshape(-1)[0]),
        "scores_raw": scores.detach().cpu().numpy(),
        "scores_norm": scores_norm.detach().cpu().numpy(),
    }


def _method_integrated_gradients(case_pack: dict, scorer_pack: dict, steps: int) -> dict:
    backbone = scorer_pack["assets"]["backbone"]
    input_ids = case_pack["input_ids"]
    attention_mask = case_pack["attention_mask"]
    timesteps = case_pack["timesteps"]
    actual_emb = backbone.token_embedding(input_ids).detach()
    baseline_ids = torch.full_like(input_ids, scorer_pack["pad_id"])
    baseline_emb = backbone.token_embedding(baseline_ids).detach()
    delta = actual_emb - baseline_emb
    total_grad = torch.zeros_like(actual_emb)
    for alpha in torch.linspace(0.0, 1.0, steps=max(int(steps), 2), device=actual_emb.device):
        emb = (baseline_emb + alpha * delta).detach().requires_grad_(True)
        backbone.zero_grad(set_to_none=True)
        scorer_pack["predictor"].model.zero_grad(set_to_none=True)
        pred = _forward_from_token_embeddings(emb, attention_mask, timesteps, scorer_pack)
        grad = torch.autograd.grad(pred.sum(), emb, retain_graph=False)[0]
        total_grad += grad.detach()
    ig = delta * (total_grad / max(int(steps), 2))
    scores = ig.abs().sum(dim=-1).squeeze(0)
    scores_norm = _normalize_token_scores(scores, attention_mask.squeeze(0))
    pred_final = _forward_from_input_ids(input_ids, attention_mask, timesteps, scorer_pack)
    return {
        "prediction": float(pred_final.detach().cpu().numpy().reshape(-1)[0]),
        "scores_raw": scores.detach().cpu().numpy(),
        "scores_norm": scores_norm.detach().cpu().numpy(),
    }


def _method_attention_rollout(case_pack: dict, scorer_pack: dict) -> dict:
    backbone = scorer_pack["assets"]["backbone"]
    hidden, attentions = backbone.get_hidden_states_and_attentions(
        case_pack["input_ids"],
        case_pack["timesteps"],
        case_pack["attention_mask"],
    )
    pred = _forward_from_hidden(hidden, scorer_pack, case_pack["attention_mask"])
    if not attentions:
        scores = torch.zeros_like(case_pack["attention_mask"].squeeze(0), dtype=torch.float32)
    else:
        seq_len = case_pack["input_ids"].size(1)
        rollout = torch.eye(seq_len, device=case_pack["input_ids"].device, dtype=torch.float32)
        for attn in attentions:
            attn_mean = attn.mean(dim=1).squeeze(0)
            eye = torch.eye(seq_len, device=attn_mean.device, dtype=attn_mean.dtype)
            aug = attn_mean + eye
            aug = aug / aug.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            rollout = aug @ rollout
        mask = case_pack["attention_mask"].squeeze(0).float()
        valid_rows = rollout[mask == 1]
        if valid_rows.numel() == 0:
            scores = torch.zeros_like(mask)
        else:
            scores = valid_rows.mean(dim=0) * mask
    scores_norm = _normalize_token_scores(scores, case_pack["attention_mask"].squeeze(0))
    return {
        "prediction": float(pred.detach().cpu().numpy().reshape(-1)[0]),
        "scores_raw": scores.detach().cpu().numpy(),
        "scores_norm": scores_norm.detach().cpu().numpy(),
    }


def _top_positions(scores_norm: np.ndarray, valid_positions: Sequence[int], top_k: int) -> List[int]:
    if not valid_positions:
        return []
    ranked = sorted(valid_positions, key=lambda pos: float(scores_norm[pos]), reverse=True)
    return ranked[: max(int(top_k), 1)]


def _mask_positions(input_ids: torch.Tensor, positions: Iterable[int], token_id: int) -> torch.Tensor:
    perturbed = input_ids.clone()
    for pos in positions:
        perturbed[:, int(pos)] = int(token_id)
    return perturbed


def _keep_only_positions(input_ids: torch.Tensor, keep_positions: Iterable[int], scorer_pack: dict) -> torch.Tensor:
    keep_set = {int(p) for p in keep_positions}
    perturbed = input_ids.clone()
    seq_len = perturbed.size(1)
    for pos in range(seq_len):
        token_id = int(perturbed[0, pos].item())
        if token_id in {scorer_pack["pad_id"], scorer_pack["bos_id"], scorer_pack["eos_id"]}:
            continue
        if pos not in keep_set:
            perturbed[:, pos] = int(scorer_pack["mask_id"])
    return perturbed


def _faithfulness_metrics(
    *,
    scores_norm: np.ndarray,
    case_pack: dict,
    scorer_pack: dict,
    original_prediction: float,
    top_k: int,
) -> dict:
    valid_positions = _valid_token_positions(case_pack, scorer_pack)
    selected = _top_positions(scores_norm, valid_positions, top_k=top_k)
    if not selected:
        return {
            "faithfulness_top_k": 0,
            "comprehensiveness_drop": np.nan,
            "sufficiency_gap": np.nan,
            "top_token_indices": "",
            "top_token_strings": "",
        }
    masked_ids = _mask_positions(case_pack["input_ids"], selected, scorer_pack["mask_id"])
    keep_ids = _keep_only_positions(case_pack["input_ids"], selected, scorer_pack)
    with torch.no_grad():
        pred_masked = _forward_from_input_ids(masked_ids, case_pack["attention_mask"], case_pack["timesteps"], scorer_pack)
        pred_keep = _forward_from_input_ids(keep_ids, case_pack["attention_mask"], case_pack["timesteps"], scorer_pack)
    masked_value = float(pred_masked.detach().cpu().numpy().reshape(-1)[0])
    keep_value = float(pred_keep.detach().cpu().numpy().reshape(-1)[0])
    top_tokens = [case_pack["tokens"][pos] for pos in selected if pos < len(case_pack["tokens"])]
    return {
        "faithfulness_top_k": int(len(selected)),
        "comprehensiveness_drop": float(original_prediction - masked_value),
        "sufficiency_gap": float(original_prediction - keep_value),
        "top_token_indices": ",".join(str(int(pos)) for pos in selected),
        "top_token_strings": ",".join(str(tok) for tok in top_tokens),
    }


def _attribute_case(case_pack: dict, scorer_pack: dict, *, methods: Sequence[str], ig_steps: int, faithfulness_top_k: int) -> Dict[str, dict]:
    outputs: Dict[str, dict] = {}
    for method in methods:
        if method == "gradient_x_hidden":
            result = _method_gradient_x_hidden(case_pack, scorer_pack)
        elif method == "integrated_gradients":
            result = _method_integrated_gradients(case_pack, scorer_pack, steps=ig_steps)
        elif method == "attention_rollout":
            result = _method_attention_rollout(case_pack, scorer_pack)
        else:
            raise ValueError(f"Unsupported method: {method}")
        result.update(
            _faithfulness_metrics(
                scores_norm=np.asarray(result["scores_norm"], dtype=np.float32),
                case_pack=case_pack,
                scorer_pack=scorer_pack,
                original_prediction=float(result["prediction"]),
                top_k=int(faithfulness_top_k),
            )
        )
        outputs[method] = result
    return outputs


def _build_interpretability_frames(
    selected_df: pd.DataFrame,
    scorer_pack: dict,
    *,
    methods: Sequence[str],
    ig_steps: int,
    faithfulness_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    token_rows: List[dict] = []
    case_rows: List[dict] = []
    agreement_rows: List[dict] = []

    for _, row in selected_df.iterrows():
        smiles = str(row.get("smiles", "")).strip()
        if not smiles:
            continue
        try:
            case_pack = _encode_case(smiles, scorer_pack)
            attr_by_method = _attribute_case(
                case_pack,
                scorer_pack,
                methods=methods,
                ig_steps=int(ig_steps),
                faithfulness_top_k=int(faithfulness_top_k),
            )
        except Exception as exc:
            case_rows.append(
                {
                    "proposal_view": normalize_view_name(row.get("proposal_view", "")),
                    "outcome_group": row.get("outcome_group", ""),
                    "smiles": smiles,
                    "method": "error",
                    "interpretability_error": str(exc),
                }
            )
            continue

        for method, attr in attr_by_method.items():
            scores_norm = np.asarray(attr["scores_norm"], dtype=np.float32)
            top_token_share = float(np.sort(scores_norm[np.isfinite(scores_norm)])[-3:].sum()) if np.isfinite(scores_norm).any() else 0.0
            entropy = _normalized_entropy(scores_norm)
            case_rows.append(
                {
                    "proposal_view": normalize_view_name(row.get("proposal_view", "")),
                    "outcome_group": row.get("outcome_group", ""),
                    "smiles": smiles,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "prediction": _safe_float(attr["prediction"]),
                    "property_hit": bool(row.get("property_hit", False)),
                    "fair_hit": bool(row.get("fair_hit", False)),
                    "target_violation": _safe_float(row.get("target_violation")),
                    "top_token_share": _safe_float(top_token_share),
                    "normalized_entropy": _safe_float(entropy),
                    "faithfulness_top_k": int(attr["faithfulness_top_k"]),
                    "comprehensiveness_drop": _safe_float(attr["comprehensiveness_drop"]),
                    "sufficiency_gap": _safe_float(attr["sufficiency_gap"]),
                    "top_token_indices": attr["top_token_indices"],
                    "top_token_strings": attr["top_token_strings"],
                }
            )
            for idx, (token, score_norm) in enumerate(zip(case_pack["tokens"], scores_norm[: len(case_pack["tokens"])])):
                token_rows.append(
                    {
                        "proposal_view": normalize_view_name(row.get("proposal_view", "")),
                        "outcome_group": row.get("outcome_group", ""),
                        "smiles": smiles,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "token_position": int(idx),
                        "token": str(token),
                        "token_score_norm": _safe_float(score_norm),
                        "prediction": _safe_float(attr["prediction"]),
                        "property_hit": bool(row.get("property_hit", False)),
                        "fair_hit": bool(row.get("fair_hit", False)),
                    }
                )

        for method_a, method_b in itertools.combinations(methods, 2):
            scores_a = np.asarray(attr_by_method[method_a]["scores_norm"], dtype=np.float32)[: len(case_pack["tokens"])]
            scores_b = np.asarray(attr_by_method[method_b]["scores_norm"], dtype=np.float32)[: len(case_pack["tokens"])]
            valid_positions = _valid_token_positions(case_pack, scorer_pack)
            top_a = set(_top_positions(scores_a, valid_positions, top_k=int(faithfulness_top_k)))
            top_b = set(_top_positions(scores_b, valid_positions, top_k=int(faithfulness_top_k)))
            union = len(top_a | top_b)
            overlap = float(len(top_a & top_b) / union) if union else np.nan
            agreement_rows.append(
                {
                    "proposal_view": normalize_view_name(row.get("proposal_view", "")),
                    "outcome_group": row.get("outcome_group", ""),
                    "smiles": smiles,
                    "method_a": method_a,
                    "method_b": method_b,
                    "method_pair": f"{method_a}__{method_b}",
                    "topk_overlap": overlap,
                    "spearman_token_score": _safe_float(_safe_spearman(scores_a, scores_b)),
                }
            )

    return pd.DataFrame(token_rows), pd.DataFrame(case_rows), pd.DataFrame(agreement_rows)


def _summarize_tokens(token_df: pd.DataFrame) -> pd.DataFrame:
    if token_df.empty:
        return pd.DataFrame(columns=["proposal_view", "outcome_group", "method", "token", "token_score_norm_mean", "token_score_norm_sum", "occurrences"])
    summary = (
        token_df.groupby(["proposal_view", "outcome_group", "method", "method_label", "token"], as_index=False)
        .agg(
            token_score_norm_mean=("token_score_norm", "mean"),
            token_score_norm_sum=("token_score_norm", "sum"),
            occurrences=("smiles", "count"),
        )
        .sort_values(
            ["proposal_view", "outcome_group", "method", "token_score_norm_mean", "token_score_norm_sum"],
            ascending=[True, True, True, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return summary


def _metrics_rows(
    case_df: pd.DataFrame,
    token_summary_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    *,
    property_name: str,
    model_size: str,
    encoder_view: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    if case_df.empty:
        return pd.DataFrame()
    for view in ordered_views(case_df["proposal_view"].tolist()):
        for outcome_group in OUTCOME_GROUPS:
            for method in METHODS_DEFAULT:
                sub = case_df[
                    (case_df["proposal_view"] == view)
                    & (case_df["outcome_group"] == outcome_group)
                    & (case_df["method"] == method)
                ].copy()
                if sub.empty:
                    continue
                tok_sub = token_summary_df[
                    (token_summary_df["proposal_view"] == view)
                    & (token_summary_df["outcome_group"] == outcome_group)
                    & (token_summary_df["method"] == method)
                ].copy()
                agr_sub = agreement_df[
                    (agreement_df["proposal_view"] == view)
                    & (agreement_df["outcome_group"] == outcome_group)
                    & (agreement_df["method_a"] == method)
                ].copy()
                top_tokens = ",".join(tok_sub.head(5)["token"].astype(str).tolist()) if not tok_sub.empty else ""
                rows.append(
                    {
                        "method": "Multi_View_Foundation",
                        "representation": view_label(view),
                        "proposal_view": view,
                        "encoder_view": encoder_view,
                        "model_size": model_size,
                        "property": property_name,
                        "outcome_group": outcome_group,
                        "interpretability_method": method,
                        "interpretability_method_label": METHOD_LABELS[method],
                        "n_cases": int(len(sub)),
                        "mean_prediction": round(float(pd.to_numeric(sub["prediction"], errors="coerce").mean()), 6),
                        "mean_target_violation": round(float(pd.to_numeric(sub["target_violation"], errors="coerce").mean()), 6),
                        "mean_top_token_share": round(float(pd.to_numeric(sub["top_token_share"], errors="coerce").mean()), 6),
                        "mean_normalized_entropy": round(float(pd.to_numeric(sub["normalized_entropy"], errors="coerce").mean()), 6),
                        "mean_comprehensiveness_drop": round(float(pd.to_numeric(sub["comprehensiveness_drop"], errors="coerce").mean()), 6),
                        "mean_sufficiency_gap": round(float(pd.to_numeric(sub["sufficiency_gap"], errors="coerce").mean()), 6),
                        "mean_method_pair_overlap": round(float(pd.to_numeric(agr_sub["topk_overlap"], errors="coerce").mean()), 6) if not agr_sub.empty else np.nan,
                        "mean_method_pair_spearman": round(float(pd.to_numeric(agr_sub["spearman_token_score"], errors="coerce").mean()), 6) if not agr_sub.empty else np.nan,
                        "top_tokens": top_tokens,
                    }
                )
    return pd.DataFrame(rows)


def _plot_token_heatmaps(
    token_summary_df: pd.DataFrame,
    *,
    property_name: str,
    figures_dir: Path,
    max_tokens_in_figure: int,
    methods: Sequence[str],
) -> None:
    if plt is None or token_summary_df.empty:
        return
    active_groups = [g for g in OUTCOME_GROUPS if not token_summary_df[token_summary_df["outcome_group"] == g].empty]
    if not active_groups:
        return
    global_max = pd.to_numeric(token_summary_df["token_score_norm_mean"], errors="coerce").max()
    vmax = float(global_max) if np.isfinite(global_max) and float(global_max) > 0.0 else 1.0
    fig, axes = plt.subplots(len(methods), len(active_groups), figsize=(5.2 * len(active_groups), 4.6 * len(methods)), squeeze=False)
    for row_idx, method in enumerate(methods):
        for col_idx, outcome_group in enumerate(active_groups):
            ax = axes[row_idx][col_idx]
            group_df = token_summary_df[
                (token_summary_df["outcome_group"] == outcome_group) & (token_summary_df["method"] == method)
            ].copy()
            views = ordered_views(group_df["proposal_view"].tolist())
            token_order = (
                group_df.groupby("token", as_index=False)["token_score_norm_mean"]
                .mean()
                .sort_values("token_score_norm_mean", ascending=False)
                .head(int(max_tokens_in_figure))["token"]
                .astype(str)
                .tolist()
            )
            if not views or not token_order:
                ax.set_axis_off()
                continue
            matrix = np.zeros((len(views), len(token_order)), dtype=np.float32)
            for i, view in enumerate(views):
                for j, token in enumerate(token_order):
                    match = group_df[(group_df["proposal_view"] == view) & (group_df["token"] == token)]
                    matrix[i, j] = float(match["token_score_norm_mean"].iloc[0]) if not match.empty else 0.0
            im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=vmax)
            ax.set_yticks(np.arange(len(views)))
            ax.set_yticklabels([view_label(v) for v in views])
            ax.set_xticks(np.arange(len(token_order)))
            ax.set_xticklabels(token_order, rotation=45, ha="right")
            ax.set_ylabel(METHOD_LABELS[method])
            if row_idx == 0:
                ax.set_title(OUTCOME_LABELS[outcome_group])
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Mean normalized token attribution")
    fig.tight_layout()
    _save_figure(fig, figures_dir / f"figure_f6_token_saliency_{property_name}")
    plt.close(fig)


def _plot_faithfulness_summary(
    case_df: pd.DataFrame,
    *,
    property_name: str,
    figures_dir: Path,
    methods: Sequence[str],
) -> None:
    if plt is None or case_df.empty:
        return
    summary = (
        case_df.groupby(["method", "outcome_group"], as_index=False)
        .agg(
            comprehensiveness_drop=("comprehensiveness_drop", "mean"),
            sufficiency_gap=("sufficiency_gap", "mean"),
        )
    )
    active_groups = [g for g in OUTCOME_GROUPS if not summary[summary["outcome_group"] == g].empty]
    if not active_groups:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))
    for panel_idx, (ax, value_col) in enumerate(zip(axes, ["comprehensiveness_drop", "sufficiency_gap"])):
        width = 0.22
        xpos = np.arange(len(methods), dtype=np.float32)
        for offset_idx, outcome_group in enumerate(active_groups):
            sub = summary[summary["outcome_group"] == outcome_group].set_index("method").reindex(methods).reset_index()
            vals = pd.to_numeric(sub[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            ax.bar(
                xpos + (offset_idx - (len(active_groups) - 1) / 2.0) * width,
                vals,
                width=width,
                alpha=0.82,
                label=OUTCOME_LABELS[outcome_group],
                color=[view_color("smiles"), view_color("selfies"), view_color("graph")][offset_idx % 3],
            )
        ax.set_xticks(xpos)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
        if value_col == "comprehensiveness_drop":
            ax.set_ylabel("Mean comprehensiveness drop")
        else:
            ax.set_ylabel("Mean sufficiency gap")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=12)
        ax.text(0.01, 0.99, f"({chr(ord('A') + panel_idx)})", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, figures_dir / f"figure_f6_faithfulness_{property_name}")
    plt.close(fig)


def _load_json_dict(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    return next((path for path in candidates if path.exists()), None)


def _filter_property_rows_if_available(
    df: pd.DataFrame,
    *,
    property_name: str,
    source_path: Path,
    label: str,
) -> pd.DataFrame:
    if "property" not in df.columns:
        return df
    prop = _normalize_property_name(property_name)
    prop_series = df["property"].astype(str).map(_normalize_property_name)
    match_mask = prop_series == prop
    if bool(match_mask.any()):
        return df.loc[match_mask].copy()
    seen = [x for x in sorted(prop_series.unique().tolist()) if x]
    seen_preview = ",".join(seen[:6]) if seen else "(empty)"
    raise RuntimeError(
        f"{label} does not contain rows for property={property_name}: {source_path}. "
        f"Found properties={seen_preview}."
    )


def _read_required_csv(path: Path, *, label: str, property_name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(
            f"{label} is unreadable or empty: {path}. "
            "Run F6 first to generate interpretability artifacts before regenerating figures."
        ) from exc
    if df.empty:
        raise RuntimeError(
            f"{label} is empty: {path}. "
            "Run F6 first to generate interpretability artifacts before regenerating figures."
        )
    return _filter_property_rows_if_available(
        df,
        property_name=property_name,
        source_path=path,
        label=label,
    )


def _load_existing_f6_artifacts(
    *,
    step_dirs: dict,
    property_step_dirs: dict,
    property_name: str,
) -> dict:
    prop = _normalize_property_name(property_name)
    token_summary_path = _first_existing_path(
        [
            property_step_dirs["files_dir"] / f"dit_token_summary_{prop}.csv",
            step_dirs["files_dir"] / f"dit_token_summary_{prop}.csv",
        ]
    )
    if token_summary_path is None:
        raise FileNotFoundError(
            f"No dit_token_summary CSV found for property={property_name}. "
            "Run F6 first or provide the expected interpretability outputs."
        )
    case_path = _first_existing_path(
        [
            property_step_dirs["files_dir"] / f"dit_case_attributions_{prop}.csv",
            step_dirs["files_dir"] / f"dit_case_attributions_{prop}.csv",
        ]
    )
    if case_path is None:
        raise FileNotFoundError(
            f"No dit_case_attributions CSV found for property={property_name}. "
            "Run F6 first or provide the expected interpretability outputs."
        )

    run_meta = {}
    run_meta_path = ""
    for candidate in [
        property_step_dirs["files_dir"] / f"run_meta_{prop}.json",
        property_step_dirs["files_dir"] / "run_meta.json",
        step_dirs["files_dir"] / f"run_meta_{prop}.json",
        step_dirs["files_dir"] / "run_meta.json",
    ]:
        payload = _load_json_dict(candidate)
        if payload:
            run_meta = payload
            run_meta_path = str(candidate)
            break

    token_summary_df = _read_required_csv(
        token_summary_path,
        label="dit_token_summary",
        property_name=property_name,
    )
    case_df = _read_required_csv(
        case_path,
        label="dit_case_attributions",
        property_name=property_name,
    )
    return {
        "token_summary_df": token_summary_df,
        "case_df": case_df,
        "run_meta": run_meta,
        "run_meta_path": run_meta_path,
    }


def _resolve_existing_f6_methods(
    *,
    args_methods: Optional[str],
    run_meta: dict,
    token_summary_df: pd.DataFrame,
    case_df: pd.DataFrame,
) -> List[str]:
    if args_methods is not None:
        return _parse_methods(args_methods)
    if run_meta.get("methods") is not None:
        return _parse_methods(run_meta.get("methods"))
    if run_meta.get("attribution_method") is not None:
        return _parse_methods(run_meta.get("attribution_method"))
    discovered = []
    for df in [token_summary_df, case_df]:
        if "method" not in df.columns:
            continue
        for method in df["method"].dropna().astype(str).tolist():
            key = method.strip().lower()
            if key in METHOD_LABELS and key not in discovered:
                discovered.append(key)
    return discovered or list(METHODS_DEFAULT)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--property", type=str, default=None)
    parser.add_argument("--encoder_view", type=str, default=None)
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated interpretability methods.")
    parser.add_argument("--ig_steps", type=int, default=None)
    parser.add_argument("--faithfulness_top_k", type=int, default=None)
    parser.add_argument("--max_samples_per_group", type=int, default=None)
    parser.add_argument("--max_tokens_in_figure", type=int, default=None)
    parser.add_argument("--figures_only", action="store_true", help="Regenerate F6 figures from existing interpretability artifacts.")
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    return parser


def main(args) -> None:
    config = load_config(args.config)
    cfg_f5 = config.get("foundation_inverse", {}) or {}
    cfg_f6 = _dit_cfg(config)

    results_dir = foundation_resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step6_dit_interpretability")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")

    property_name = _normalize_property_name(args.property or cfg_f6.get("property") or cfg_f5.get("property") or "Tg")
    property_step_dirs = ensure_step_dirs(results_dir, "step6_dit_interpretability", property_name)
    save_config(config, property_step_dirs["files_dir"] / "config_used.yaml")

    encoder_view = str(args.encoder_view or cfg_f6.get("encoder_view") or cfg_f5.get("encoder_view") or "smiles").strip() or "smiles"
    if normalize_view_name(encoder_view) != "smiles":
        raise ValueError("F6 interpretability currently supports the shared smiles scorer only.")

    methods = _parse_methods(args.methods if args.methods is not None else cfg_f6.get("methods"))
    ig_steps = _to_int_or_none(args.ig_steps)
    if ig_steps is None:
        ig_steps = _to_int_or_none(cfg_f6.get("ig_steps"))
    if ig_steps is None:
        ig_steps = 16
    faithfulness_top_k = _to_int_or_none(args.faithfulness_top_k)
    if faithfulness_top_k is None:
        faithfulness_top_k = _to_int_or_none(cfg_f6.get("faithfulness_top_k"))
    if faithfulness_top_k is None:
        faithfulness_top_k = 3
    max_samples_per_group = _to_int_or_none(args.max_samples_per_group)
    if max_samples_per_group is None:
        max_samples_per_group = _to_int_or_none(cfg_f6.get("max_samples_per_group"))
    if max_samples_per_group is None:
        max_samples_per_group = 24
    max_tokens_in_figure = _to_int_or_none(args.max_tokens_in_figure)
    if max_tokens_in_figure is None:
        max_tokens_in_figure = _to_int_or_none(cfg_f6.get("max_tokens_in_figure"))
    if max_tokens_in_figure is None:
        max_tokens_in_figure = 16

    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(cfg_f6.get("generate_figures", True), True)
    if args.figures_only:
        artifacts = _load_existing_f6_artifacts(
            step_dirs=step_dirs,
            property_step_dirs=property_step_dirs,
            property_name=property_name,
        )
        token_summary_df = artifacts["token_summary_df"]
        case_df = artifacts["case_df"]
        run_meta = artifacts["run_meta"]
        methods = _resolve_existing_f6_methods(
            args_methods=args.methods,
            run_meta=run_meta,
            token_summary_df=token_summary_df,
            case_df=case_df,
        )
        max_tokens_for_figures = _to_int_or_none(args.max_tokens_in_figure)
        if max_tokens_for_figures is None:
            max_tokens_for_figures = _to_int_or_none(run_meta.get("max_tokens_in_figure"))
        if max_tokens_for_figures is None:
            max_tokens_for_figures = int(max_tokens_in_figure)
        if generate_figures and plt is not None:
            _plot_token_heatmaps(
                token_summary_df,
                property_name=property_name,
                figures_dir=property_step_dirs["figures_dir"],
                max_tokens_in_figure=int(max_tokens_for_figures),
                methods=methods,
            )
            _plot_faithfulness_summary(
                case_df,
                property_name=property_name,
                figures_dir=property_step_dirs["figures_dir"],
                methods=methods,
            )
        print(f"Saved F6 figures to {property_step_dirs['figures_dir']}")
        return

    candidate_scores_path = _resolve_view_compare_scores_path(results_dir, property_name)
    df = pd.read_csv(candidate_scores_path)
    if "property" in df.columns:
        prop_mask = df["property"].astype(str).map(_normalize_property_name) == property_name
        if bool(prop_mask.any()):
            df = df.loc[prop_mask].copy()
    if "smiles" not in df.columns:
        raise ValueError(f"Interpretability input must include 'smiles': {candidate_scores_path}")
    if "proposal_view" not in df.columns:
        raise ValueError(f"Interpretability input must include 'proposal_view': {candidate_scores_path}")
    if "prediction" not in df.columns:
        raise ValueError(f"Interpretability input must include 'prediction': {candidate_scores_path}")

    df["proposal_view"] = df["proposal_view"].astype(str).map(lambda x: normalize_view_name(x) or "all")
    df = df.loc[pd.to_numeric(df["prediction"], errors="coerce").notna()].copy()
    if "fair_hit" not in df.columns:
        df["fair_hit"] = False
    if "property_hit" not in df.columns:
        df["property_hit"] = False
    selected_df = _select_cases(df, max_samples_per_group=int(max_samples_per_group))
    if selected_df.empty:
        raise RuntimeError(f"No candidate cases available for F6 interpretability in {candidate_scores_path}")

    scorer_pack = _load_property_scorer(
        config=config,
        results_dir=results_dir,
        property_name=property_name,
        encoder_view=encoder_view,
        property_model_path=args.property_model_path or cfg_f6.get("property_model_path"),
    )
    token_df, case_df, agreement_df = _build_interpretability_frames(
        selected_df,
        scorer_pack,
        methods=methods,
        ig_steps=int(ig_steps),
        faithfulness_top_k=int(faithfulness_top_k),
    )
    token_summary_df = _summarize_tokens(token_df)
    model_size = str(config.get(f"{encoder_view}_encoder", {}).get("model_size", "base"))
    metrics_df = _metrics_rows(
        case_df,
        token_summary_df,
        agreement_df,
        property_name=property_name,
        model_size=model_size,
        encoder_view=encoder_view,
    )

    save_csv(
        selected_df,
        property_step_dirs["files_dir"] / f"interpretability_selected_cases_{property_name}.csv",
        legacy_paths=[step_dirs["files_dir"] / f"interpretability_selected_cases_{property_name}.csv"],
        index=False,
    )
    save_csv(
        case_df,
        property_step_dirs["files_dir"] / f"dit_case_attributions_{property_name}.csv",
        legacy_paths=[step_dirs["files_dir"] / f"dit_case_attributions_{property_name}.csv"],
        index=False,
    )
    save_csv(
        token_df,
        property_step_dirs["files_dir"] / f"dit_token_attributions_{property_name}.csv",
        legacy_paths=[step_dirs["files_dir"] / f"dit_token_attributions_{property_name}.csv"],
        index=False,
    )
    save_csv(
        token_summary_df,
        property_step_dirs["files_dir"] / f"dit_token_summary_{property_name}.csv",
        legacy_paths=[step_dirs["files_dir"] / f"dit_token_summary_{property_name}.csv"],
        index=False,
    )
    save_csv(
        agreement_df,
        property_step_dirs["files_dir"] / f"dit_method_agreement_{property_name}.csv",
        legacy_paths=[step_dirs["files_dir"] / f"dit_method_agreement_{property_name}.csv"],
        index=False,
    )
    save_csv(
        metrics_df,
        property_step_dirs["metrics_dir"] / "metrics_dit_interpretability.csv",
        legacy_paths=[step_dirs["metrics_dir"] / "metrics_dit_interpretability.csv"],
        index=False,
    )
    save_csv(
        metrics_df,
        property_step_dirs["metrics_dir"] / f"metrics_dit_interpretability_{property_name}.csv",
        legacy_paths=[step_dirs["metrics_dir"] / f"metrics_dit_interpretability_{property_name}.csv"],
        index=False,
    )

    run_meta = {
        "property": property_name,
        "encoder_view": encoder_view,
        "candidate_scores_path": str(candidate_scores_path),
        "property_model_path": str(scorer_pack["model_path"]),
        "methods": list(methods),
        "ig_steps": int(ig_steps),
        "faithfulness_top_k": int(faithfulness_top_k),
        "max_samples_per_group": int(max_samples_per_group),
        "max_tokens_in_figure": int(max_tokens_in_figure),
        "proposal_views": ordered_views(selected_df["proposal_view"].tolist()),
        "outcome_groups": list(OUTCOME_GROUPS),
        "n_selected_cases": int(len(selected_df)),
        "n_interpreted_cases": int(len(case_df)),
        "analysis_mode": "smiles_dit_multi_method_interpretability",
        "attribution_method": ",".join(methods),
        "faithfulness_metrics": ["comprehensiveness_drop", "sufficiency_gap"],
    }
    save_json(
        run_meta,
        property_step_dirs["files_dir"] / "run_meta.json",
        legacy_paths=[step_dirs["files_dir"] / "run_meta.json"],
    )
    save_json(
        run_meta,
        property_step_dirs["files_dir"] / f"run_meta_{property_name}.json",
        legacy_paths=[step_dirs["files_dir"] / f"run_meta_{property_name}.json"],
    )

    if generate_figures and plt is not None:
        _plot_token_heatmaps(
            token_summary_df,
            property_name=property_name,
            figures_dir=property_step_dirs["figures_dir"],
            max_tokens_in_figure=int(max_tokens_in_figure),
            methods=methods,
        )
        _plot_faithfulness_summary(
            case_df,
            property_name=property_name,
            figures_dir=property_step_dirs["figures_dir"],
            methods=methods,
        )

    print(f"Saved metrics_dit_interpretability.csv to {property_step_dirs['step_dir']}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
