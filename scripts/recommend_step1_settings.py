#!/usr/bin/env python
"""Recommend Step1 optimization settings from current system and GPU."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from shared.step1_recommendations import (
    detect_gpu_info,
    recommend_step1_optimization,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _infer_pipeline_kind(config_path: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    lowered = str(config_path).lower()
    if "group_selfies" in lowered:
        return "group_selfies"
    if "graph" in lowered:
        return "graph"
    return "sequence"


def _choose_world_size(user_world_size: int, gpu_info: Dict[str, Any]) -> int:
    if user_world_size and user_world_size > 0:
        return int(user_world_size)
    env_world_size = int(os.environ.get("WORLD_SIZE", "0") or 0)
    if env_world_size > 0:
        return env_world_size
    if gpu_info["cuda_available"]:
        return max(1, int(gpu_info["device_count"]))
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend Step1 settings from current system/GPU."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml (e.g., Bi_Diffusion_SMILES/configs/config.yaml)",
    )
    parser.add_argument(
        "--pipeline-kind",
        type=str,
        default="auto",
        choices=["auto", "sequence", "group_selfies", "graph"],
        help="Pipeline type; default auto-infers from config path.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=0,
        help="Total distributed world size. Defaults to WORLD_SIZE env, then GPU count, then 1.",
    )
    parser.add_argument(
        "--reference-world-size",
        type=int,
        default=4,
        help="Reference world size used to preserve baseline global batch when scaling out.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index used for capability detection.",
    )
    parser.add_argument(
        "--model-num-layers",
        type=int,
        default=0,
        help="Optional active model depth; if set, enables auto_batch_law evaluation by depth.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_yaml(config_path)
    pipeline_kind = _infer_pipeline_kind(config_path, args.pipeline_kind)
    gpu_info = detect_gpu_info(args.device_index)
    world_size = _choose_world_size(args.world_size, gpu_info)

    rec = recommend_step1_optimization(
        config=config,
        pipeline_kind=pipeline_kind,
        world_size=world_size,
        reference_world_size=args.reference_world_size,
        gpu_info=gpu_info,
        model_num_layers=(args.model_num_layers if args.model_num_layers > 0 else None),
    )

    print("System summary:")
    print(
        yaml.safe_dump(
            {"gpu": gpu_info, "system": rec.get("system", {}), "world_size": world_size},
            sort_keys=False,
        ).strip()
    )
    print("")
    print("Recommended Step1 optimization overrides:")
    print(yaml.safe_dump({"optimization": rec["recommended_optimization"]}, sort_keys=False).strip())
    print("")
    print("Recommended Step1 training_backbone overrides:")
    print(
        yaml.safe_dump(
            {"training_backbone": rec.get("recommended_training_backbone", {})},
            sort_keys=False,
        ).strip()
    )
    print("")
    print("Batch scaling summary:")
    print(
        yaml.safe_dump(
            {
                "original_per_rank_batch_size": rec.get("original_per_rank_batch_size"),
                "per_rank_batch_size": rec["per_rank_batch_size"],
                "memory_aware_batch_meta": rec.get("memory_aware_batch_meta"),
                "current_grad_accumulation_steps": rec["current_grad_accumulation_steps"],
                "current_global_batch_size": rec["current_global_batch_size"],
                "reference_world_size": rec["reference_world_size"],
                "target_global_batch_source": rec["target_global_batch_source"],
                "cpu_oom_guard_meta": rec.get("cpu_oom_guard_meta"),
                "model_num_layers": rec["model_num_layers"],
                "h100_preset_active": rec.get("h100_preset_active", False),
                "compute_optimal_scaling_active": rec.get("compute_optimal_scaling_active", False),
                "compute_optimal_source": rec.get("compute_optimal_source"),
                "compute_optimal_batch_ratio": rec.get("compute_optimal_batch_ratio"),
                "compute_optimal_meta": rec.get("compute_optimal_meta"),
                "recommended_target_global_batch_size": rec[
                    "recommended_target_global_batch_size"
                ],
                "recommended_achieved_global_batch_size": rec.get(
                    "recommended_achieved_global_batch_size"
                ),
                "expected_grad_accumulation_steps_after_target": rec[
                    "expected_grad_accumulation_steps_after_target"
                ],
            },
            sort_keys=False,
        ).strip()
    )


if __name__ == "__main__":
    main()
