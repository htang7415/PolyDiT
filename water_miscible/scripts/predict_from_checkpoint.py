#!/usr/bin/env python
"""Predict chi or water miscibility from a saved water_miscible checkpoint."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from run_five_view_tasks import (
    MLP,
    TASK_KINDS,
    add_polymer_ids,
    align_embeddings_and_raw,
    apply_model_size_override,
    build_finetune_predictor,
    build_embedding_and_raw_caches,
    chi_feature_matrix,
    device_from_config,
    load_view_assets,
    load_yaml,
    predict_joint_model,
    predict_model,
    resolve_path,
)


def make_scaler(checkpoint: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = int(scaler.mean_.shape[0])
    return scaler


def prepare_input(df: pd.DataFrame, task_name: str, config: dict) -> pd.DataFrame:
    if "SMILES" not in df.columns:
        raise ValueError("Input CSV must contain a SMILES column.")
    out = df.copy()
    if task_name == "chi_regression":
        defaults = config.get("data", {})
        if "temperature" not in out.columns:
            out["temperature"] = float(defaults.get("default_temperature", 293.15))
        if "phi" not in out.columns:
            out["phi"] = float(defaults.get("default_phi", 0.2))
        out["temperature"] = pd.to_numeric(out["temperature"], errors="coerce")
        out["phi"] = pd.to_numeric(out["phi"], errors="coerce")
        if out[["temperature", "phi"]].isna().any(axis=None):
            raise ValueError("temperature and phi must be numeric for chi_regression.")
        out["chi"] = 0.0
        out["water_miscible"] = 0
    else:
        out["water_miscible"] = 0
        out["temperature"] = float(config.get("data", {}).get("default_temperature", 293.15))
        out["phi"] = float(config.get("data", {}).get("default_phi", 0.2))
        out["chi"] = 0.0
    return add_polymer_ids(out)


def predict(args) -> None:
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    task_name = args.task or str(checkpoint.get("task", ""))
    view = args.view or str(checkpoint.get("view", ""))
    if task_name not in TASK_KINDS:
        raise ValueError("--task must be chi_regression or water_classification, or checkpoint must store a valid task.")
    if not view:
        raise ValueError("--view is required when checkpoint does not store one.")

    config = apply_model_size_override(load_yaml(resolve_path(args.config)), args.model_size)
    device = args.device or device_from_config(config)
    assets = load_view_assets(config, view, device=device)

    df = prepare_input(pd.read_csv(resolve_path(args.input_csv)), task_name, config)
    cache_dir = resolve_path(args.cache_dir) if args.cache_dir else checkpoint_path.parent / "prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = f"_{args.cache_tag.strip()}" if args.cache_tag and args.cache_tag.strip() else ""
    smiles_values = df["SMILES"].astype(str).tolist()
    embedding_map, raw_map = build_embedding_and_raw_caches(
        smiles_values=smiles_values,
        assets=assets,
        embedding_cache_path=cache_dir / f"{view}{cache_tag}_embeddings.npz",
        raw_cache_path=cache_dir / f"{view}{cache_tag}_raw_inputs.npz",
        device=device,
        force_rebuild=bool(args.fresh_cache),
    )
    aligned_df, embeddings, raw_inputs = align_embeddings_and_raw(df, embedding_map, raw_map, assets)
    if aligned_df.empty:
        raise RuntimeError("No input rows could be encoded for the requested view.")

    params = dict(checkpoint["params"])
    scaler = make_scaler(checkpoint)
    task_kind = TASK_KINDS[task_name]
    if task_name == "chi_regression":
        X = chi_feature_matrix(aligned_df, embeddings)
        aux_features = aligned_df[["temperature", "phi"]].to_numpy(dtype=np.float32)
    else:
        X = embeddings.astype(np.float32, copy=False)
        aux_features = None

    if bool(checkpoint.get("joint_training", False)):
        predictor = build_finetune_predictor(assets=assets, scaler=scaler, params=params, device=device)
        predictor.head.load_state_dict(checkpoint["model_state_dict"])
        backbone_state = checkpoint.get("backbone_state_dict") or {}
        if backbone_state:
            predictor.backbone.load_state_dict(backbone_state, strict=False)
        pred = predict_joint_model(
            predictor,
            raw_inputs,
            np.arange(len(aligned_df), dtype=np.int64),
            view_type=assets.view_type,
            device=device,
            task=task_kind,
            aux_features=aux_features,
            batch_size=int(args.batch_size),
        )
    else:
        model = MLP(
            int(checkpoint.get("input_dim", X.shape[1])),
            params["hidden_sizes"],
            dropout=float(params.get("dropout", 0.0)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        pred = predict_model(model, scaler, X, device=device, task=task_kind, batch_size=int(args.batch_size))

    out = aligned_df.copy()
    if task_name == "water_classification":
        out["class_prob"] = pred
        out["class_pred"] = (out["class_prob"] >= 0.5).astype(int)
    else:
        out["chi_pred"] = pred
    output_path = resolve_path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="water_miscible/configs/config_water.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--task", choices=sorted(TASK_KINDS), default=None)
    parser.add_argument("--view", default=None)
    parser.add_argument("--model_size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--cache_tag", default="predict")
    parser.add_argument("--fresh_cache", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    predict(parse_args())
