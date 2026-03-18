#!/usr/bin/env python
"""Compatibility utility: regenerate F5 view-comparison outputs from candidate scores."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.view_compare import analyze_view_compare, plot_view_compare, save_view_compare_outputs
from src.utils.config import load_config, save_config
from src.utils.output_layout import ensure_step_dirs
from src.utils.property_names import normalize_property_name
from step5_foundation_inverse import _resolve_path


def _resolve_candidate_scores_path(results_dir: Path, property_name: str) -> Path:
    alias = normalize_property_name(property_name)
    candidates = [
        results_dir / "step5_foundation_inverse" / alias / "files" / f"candidate_scores_{alias}.csv",
        results_dir / "step5_foundation_inverse" / alias / "files" / "candidate_scores.csv",
        results_dir / "step5_foundation_inverse" / "files" / f"candidate_scores_{alias}.csv",
        results_dir / "step5_foundation_inverse" / "files" / "candidate_scores.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"F5 candidate scores not found for property={property_name}. searched={[str(p) for p in candidates]}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--property", type=str, default=None)
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--target_mode", type=str, default=None, choices=["window", "ge", "le"])
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    return parser


def main(args) -> None:
    config = load_config(args.config)
    cfg_f5 = config.get("foundation_inverse", {}) or {}
    results_dir = _resolve_path(config["paths"]["results_dir"])
    property_name = normalize_property_name(args.property or cfg_f5.get("property") or "Tg")
    target = args.target if args.target is not None else float((cfg_f5.get("targets") or {}).get(property_name, cfg_f5.get("target")))
    target_mode = args.target_mode or str((cfg_f5.get("target_modes") or {}).get(property_name, cfg_f5.get("target_mode", "ge"))).strip()
    epsilon = args.epsilon if args.epsilon is not None else float((cfg_f5.get("epsilons") or {}).get(property_name, cfg_f5.get("epsilon", 0.0)))
    top_k = int(args.top_k if args.top_k is not None else cfg_f5.get("top_k", 100))
    encoder_view = str(cfg_f5.get("encoder_view", "smiles")).strip() or "smiles"
    model_size = str(config.get(f"{encoder_view}_encoder", {}).get("model_size", "base"))

    candidate_scores_path = _resolve_candidate_scores_path(results_dir, property_name)
    df = pd.read_csv(candidate_scores_path)
    if "property" in df.columns:
        prop_mask = df["property"].astype(str).map(normalize_property_name) == property_name
        if bool(prop_mask.any()):
            df = df.loc[prop_mask].copy()

    root_step_dirs = ensure_step_dirs(results_dir, "step5_foundation_inverse")
    property_step_dirs = ensure_step_dirs(results_dir, "step5_foundation_inverse", property_name)
    save_config(config, property_step_dirs["files_dir"] / "config_used.yaml")
    analysis = analyze_view_compare(
        candidate_df=df,
        property_name=property_name,
        target=float(target),
        target_mode=target_mode,
        epsilon=float(epsilon),
        top_k=top_k,
        model_size=model_size,
        scoring_view=encoder_view,
    )
    save_view_compare_outputs(
        analysis=analysis,
        property_name=property_name,
        property_step_dirs=property_step_dirs,
        root_step_dirs=root_step_dirs,
        root_legacy_dirs=[root_step_dirs["step_dir"]],
    )
    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = True
    if generate_figures:
        plot_view_compare(
            analysis=analysis,
            property_name=property_name,
            figures_dir=property_step_dirs["figures_dir"],
            figure_prefix="figure_f5_view_compare",
        )
    print(f"Regenerated F5 view comparison outputs in {property_step_dirs['step_dir']}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
