"""Step1 optimization recommendation utilities based on runtime system/GPU."""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, Optional

import torch


def detect_gpu_info(device_index: int = 0) -> Dict[str, Any]:
    """Detect runtime CUDA/GPU capabilities used for Step1 recommendations."""
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_count": 0,
            "name": None,
            "memory_gb": None,
            "per_device_memory_gb": [],
            "min_memory_gb": None,
            "max_memory_gb": None,
            "total_visible_memory_gb": None,
            "capability": None,
            "is_hopper": False,
            "supports_fp8": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS"),
        }

    count = torch.cuda.device_count()
    if count <= 0:
        return {
            "cuda_available": False,
            "device_count": 0,
            "name": None,
            "memory_gb": None,
            "per_device_memory_gb": [],
            "min_memory_gb": None,
            "max_memory_gb": None,
            "total_visible_memory_gb": None,
            "capability": None,
            "is_hopper": False,
            "supports_fp8": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS"),
        }

    idx = max(0, min(int(device_index), count - 1))
    per_device_memory_gb = []
    device_names = []
    for i in range(count):
        props_i = torch.cuda.get_device_properties(i)
        device_names.append(str(props_i.name))
        per_device_memory_gb.append(round(props_i.total_memory / (1024**3), 2))

    props = torch.cuda.get_device_properties(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    name = str(props.name)
    is_hopper = major >= 9 or "H100" in name.upper()
    supports_fp8 = bool(
        hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")
    )

    return {
        "cuda_available": True,
        "device_count": int(count),
        "name": name,
        "memory_gb": round(props.total_memory / (1024**3), 2),
        "per_device_memory_gb": per_device_memory_gb,
        "min_memory_gb": min(per_device_memory_gb) if per_device_memory_gb else None,
        "max_memory_gb": max(per_device_memory_gb) if per_device_memory_gb else None,
        "total_visible_memory_gb": round(sum(per_device_memory_gb), 2) if per_device_memory_gb else None,
        "device_names": device_names,
        "capability": f"{major}.{minor}",
        "is_hopper": bool(is_hopper),
        "supports_fp8": bool(supports_fp8),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS"),
    }


def _read_proc_meminfo_value_kb(key: str) -> Optional[int]:
    """Read one memory key from /proc/meminfo (Linux)."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith(f"{key}:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except (FileNotFoundError, ValueError, OSError):
        return None
    return None


def detect_system_info() -> Dict[str, Any]:
    """Detect CPU and RAM resources visible to this training process."""
    cpu_count = os.cpu_count() or 1
    mem_total_kb = _read_proc_meminfo_value_kb("MemTotal")
    mem_available_kb = _read_proc_meminfo_value_kb("MemAvailable")
    total_ram_gb = round(mem_total_kb / (1024**2), 2) if mem_total_kb else None
    available_ram_gb = round(mem_available_kb / (1024**2), 2) if mem_available_kb else None
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1") or 1)
    slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    return {
        "cpu_count": int(cpu_count),
        "local_world_size": int(max(1, local_world_size)),
        "slurm_cpus_per_task": int(max(0, slurm_cpus_per_task)),
        "slurm_job_id": slurm_job_id,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
    }


def _recommend_memory_safe_per_rank_batch(
    per_rank_batch: int,
    config: Dict[str, Any],
    optimization: Dict[str, Any],
    gpu: Dict[str, Any],
    active_model_cfg: Dict[str, Any],
) -> tuple[int, Dict[str, Any]]:
    """Apply a memory-aware per-rank batch cap to reduce GPU OOM risk."""
    cfg = optimization.get("memory_aware_batch_sizing", {})
    if not isinstance(cfg, dict):
        return per_rank_batch, {"enabled": False, "reason": "invalid_config"}
    if not bool(cfg.get("enabled", True)):
        return per_rank_batch, {"enabled": False, "reason": "disabled"}
    if not bool(gpu.get("cuda_available", False)):
        return per_rank_batch, {"enabled": False, "reason": "no_cuda"}

    gpu_mem_gb = gpu.get("min_memory_gb", gpu.get("memory_gb"))
    if gpu_mem_gb is None or float(gpu_mem_gb) <= 0:
        return per_rank_batch, {"enabled": False, "reason": "unknown_gpu_memory"}

    reference_gpu_mem_gb = float(cfg.get("reference_gpu_memory_gb", 80.0))
    reference_seq_len = int(cfg.get("reference_seq_len", 128))
    allow_upscale = bool(cfg.get("allow_upscale", False))
    min_batch = int(cfg.get("min_per_rank_batch_size", 4))
    max_batch = cfg.get("max_per_rank_batch_size", None)
    batch_round_multiple = int(cfg.get("batch_round_multiple", 8))
    use_model_shape_scaling = bool(cfg.get("use_model_shape_scaling", False))
    reference_layers = float(cfg.get("reference_num_layers", 12))
    reference_hidden = float(cfg.get("reference_hidden_size", 768))

    seq_len = int(config.get("tokenizer", {}).get("max_length", reference_seq_len))
    seq_scale = float(reference_seq_len) / max(1.0, float(seq_len))
    memory_scale = float(gpu_mem_gb) / max(1.0e-8, reference_gpu_mem_gb)
    model_scale = 1.0
    if use_model_shape_scaling:
        model_layers = float(active_model_cfg.get("num_layers", reference_layers))
        model_hidden = float(active_model_cfg.get("hidden_size", reference_hidden))
        numerator = reference_layers * reference_hidden
        denominator = max(1.0e-8, model_layers * model_hidden)
        model_scale = numerator / denominator

    scale = memory_scale * seq_scale * model_scale
    if not allow_upscale:
        scale = min(1.0, scale)

    proposed = int(math.floor(float(per_rank_batch) * scale))
    proposed = max(1, proposed)
    if batch_round_multiple > 1 and proposed >= batch_round_multiple:
        proposed = (proposed // batch_round_multiple) * batch_round_multiple
    proposed = max(1, proposed)
    proposed = max(min_batch, proposed)

    if max_batch is not None:
        proposed = min(proposed, int(max_batch))
    if not allow_upscale:
        proposed = min(proposed, int(per_rank_batch))

    meta = {
        "enabled": True,
        "gpu_memory_gb": float(gpu_mem_gb),
        "reference_gpu_memory_gb": float(reference_gpu_mem_gb),
        "seq_len": int(seq_len),
        "reference_seq_len": int(reference_seq_len),
        "memory_scale": float(memory_scale),
        "seq_scale": float(seq_scale),
        "model_scale": float(model_scale),
        "combined_scale": float(scale),
        "original_per_rank_batch_size": int(per_rank_batch),
        "recommended_per_rank_batch_size": int(proposed),
        "batch_round_multiple": int(batch_round_multiple),
        "allow_upscale": bool(allow_upscale),
    }
    return proposed, meta


def _recommend_cpu_dataloader_guards(
    optimization: Dict[str, Any],
    system_info: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Recommend conservative per-rank DataLoader settings for CPU OOM avoidance."""
    cfg = optimization.get("cpu_oom_guard", {})
    if not isinstance(cfg, dict):
        return {}, {"enabled": False, "reason": "invalid_config"}
    if not bool(cfg.get("enabled", True)):
        return {}, {"enabled": False, "reason": "disabled"}

    local_world_size = int(system_info.get("local_world_size", 1) or 1)
    local_world_size = max(1, local_world_size)
    cpu_count = int(system_info.get("cpu_count", 1) or 1)
    slurm_cpus_per_task = int(system_info.get("slurm_cpus_per_task", 0) or 0)
    available_ram_gb = system_info.get("available_ram_gb")

    if slurm_cpus_per_task > 0:
        per_rank_cpu_budget = max(1, slurm_cpus_per_task // local_world_size)
    else:
        per_rank_cpu_budget = max(1, cpu_count // local_world_size)
    per_rank_worker_cap = max(1, per_rank_cpu_budget - 2)
    max_step1_workers = int(cfg.get("max_step1_num_workers", 12))
    max_step1_workers = max(1, max_step1_workers)

    current_workers = int(optimization.get("step1_num_workers", optimization.get("num_workers", 4)))
    if current_workers <= 0:
        recommended_workers = min(per_rank_worker_cap, max_step1_workers)
    else:
        recommended_workers = min(current_workers, per_rank_worker_cap, max_step1_workers)

    current_prefetch = int(optimization.get("prefetch_factor", 2))
    recommended_prefetch = max(1, current_prefetch)
    persistent_workers = bool(
        optimization.get("step1_persistent_workers", optimization.get("persistent_workers", False))
    )

    if available_ram_gb is not None:
        per_rank_available_ram = float(available_ram_gb) / local_world_size
        low_ram_threshold = float(cfg.get("low_ram_threshold_gb_per_rank", 12.0))
        if per_rank_available_ram < low_ram_threshold:
            recommended_workers = min(recommended_workers, 2)
            recommended_prefetch = 1
            persistent_workers = False
    else:
        per_rank_available_ram = None

    rec = {
        "step1_num_workers": int(max(1, recommended_workers)),
        "prefetch_factor": int(max(1, recommended_prefetch)),
        "step1_persistent_workers": bool(persistent_workers and recommended_workers > 0),
    }
    meta = {
        "enabled": True,
        "cpu_count": int(cpu_count),
        "local_world_size": int(local_world_size),
        "per_rank_cpu_budget": int(per_rank_cpu_budget),
        "per_rank_worker_cap": int(per_rank_worker_cap),
        "max_step1_num_workers": int(max_step1_workers),
        "per_rank_available_ram_gb": per_rank_available_ram,
        "requested_workers": int(current_workers),
        "recommended_workers": int(rec["step1_num_workers"]),
        "requested_prefetch_factor": int(current_prefetch),
        "recommended_prefetch_factor": int(rec["prefetch_factor"]),
    }
    return rec, meta


def maybe_apply_cuda_oom_env(config: Dict[str, Any]) -> Optional[str]:
    """Set a fragmentation-friendly CUDA allocator config unless already defined."""
    optimization = config.get("optimization", {})
    guard_cfg = optimization.get("gpu_oom_guard", {})
    if isinstance(guard_cfg, dict) and not bool(guard_cfg.get("enabled", True)):
        return None
    alloc_conf = (
        guard_cfg.get("cuda_alloc_conf", "expandable_segments:True,max_split_size_mb:128")
        if isinstance(guard_cfg, dict)
        else "expandable_segments:True,max_split_size_mb:128"
    )
    if not alloc_conf:
        return None
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(alloc_conf)
    return str(alloc_conf)


def _resolve_model_config(
    config: Dict[str, Any],
    model_num_layers: Optional[int],
    model_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve active model config for scaling-law calculations."""
    if isinstance(model_config, dict) and model_config:
        return model_config

    if model_num_layers is not None:
        size_cfgs = config.get("model_sizes", {})
        if isinstance(size_cfgs, dict):
            target_layers = int(model_num_layers)
            for _, size_cfg in size_cfgs.items():
                if not isinstance(size_cfg, dict):
                    continue
                if int(size_cfg.get("num_layers", -1)) == target_layers:
                    return size_cfg

    return config.get("backbone", {})


def _estimate_scaling_params(
    pipeline_kind: str,
    model_cfg: Dict[str, Any],
    compute_cfg: Dict[str, Any],
) -> float:
    """Estimate nanochat-style scaling parameters from active model shape."""
    h = int(model_cfg.get("hidden_size", 0) or 0)
    L = int(model_cfg.get("num_layers", 0) or 0)
    f = int(model_cfg.get("ffn_hidden_size", 0) or 0)

    if h <= 0 or L <= 0:
        return 0.0
    if f <= 0:
        f = 4 * h

    # Matches nanochat intent: transformer matrices dominate scaling count.
    transformer_matrices = float(L * (4 * h * h + 2 * h * f))
    scaling_key = str(
        compute_cfg.get("scaling_params_key", "transformer_matrices_lm_head")
    ).lower()
    include_lm_head = bool(compute_cfg.get("include_lm_head", True))

    if not include_lm_head or scaling_key in {"transformer", "transformer_matrices"}:
        return transformer_matrices

    if pipeline_kind == "graph":
        atom_vocab = int(compute_cfg.get("graph_atom_vocab_size", 11))
        edge_vocab = int(compute_cfg.get("graph_edge_vocab_size", 6))
        lm_head = float(h * atom_vocab)
        # Graph edge head: Linear(2h->h) + Linear(h->edge_vocab), weights only.
        edge_head = float(2 * h * h + h * edge_vocab)
        return transformer_matrices + lm_head + edge_head

    vocab = int(compute_cfg.get("lm_head_vocab_size", 256))
    lm_head = float(h * vocab)
    return transformer_matrices + lm_head


def recommend_step1_optimization(
    config: Dict[str, Any],
    pipeline_kind: str,
    world_size: int,
    reference_world_size: int,
    gpu_info: Optional[Dict[str, Any]] = None,
    model_num_layers: Optional[int] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build Step1 optimization recommendations for a given runtime profile."""
    optimization = config.get("optimization", {})
    train_cfg = config.get("training_backbone", {})
    gpu = gpu_info or detect_gpu_info(0)

    system_info = detect_system_info()
    active_model_cfg = _resolve_model_config(
        config=config,
        model_num_layers=model_num_layers,
        model_config=model_config,
    )

    per_rank_batch = int(train_cfg.get("batch_size", 128))
    original_per_rank_batch = per_rank_batch
    per_rank_batch, memory_batch_meta = _recommend_memory_safe_per_rank_batch(
        per_rank_batch=per_rank_batch,
        config=config,
        optimization=optimization,
        gpu=gpu,
        active_model_cfg=active_model_cfg,
    )
    cpu_oom_rec, cpu_oom_meta = _recommend_cpu_dataloader_guards(
        optimization=optimization,
        system_info=system_info,
    )

    grad_accum = int(optimization.get("gradient_accumulation_steps", 1))
    world = max(1, int(world_size))
    reference_world = max(1, min(world, int(reference_world_size)))

    current_global_batch = per_rank_batch * grad_accum * world
    target_global_batch = per_rank_batch * grad_accum * reference_world
    target_batch_source = "reference_world_size"

    # Optional H100 preset for 8xH100 jobs (nanochat-inspired defaults).
    h100_cfg = optimization.get("h100_8x_preset", {})
    h100_preset_active = False
    if isinstance(h100_cfg, dict) and bool(h100_cfg.get("enabled", False)):
        min_world_size = int(h100_cfg.get("min_world_size", 8))
        h100_preset_active = gpu.get("is_hopper", False) and world >= max(1, min_world_size)

    # Nanochat-style auto batch law:
    # total_batch ~= B_ref * (depth / depth_ref)^(2/3), optionally scaled.
    auto_batch_cfg = optimization.get("auto_batch_law", {})
    law_cfg = None
    if h100_preset_active:
        law_cfg = {
            "reference_total_batch_size": h100_cfg.get("reference_total_batch_size", 8192),
            "reference_num_layers": h100_cfg.get("reference_num_layers", 12),
            "depth_exponent": h100_cfg.get("depth_exponent", 2.0 / 3.0),
            "batch_lr_scale": h100_cfg.get("batch_lr_scale", 1.0),
        }
        target_batch_source = "h100_8x_preset"
    elif isinstance(auto_batch_cfg, dict) and bool(auto_batch_cfg.get("enabled", False)):
        law_cfg = auto_batch_cfg
        target_batch_source = "auto_batch_law"

    depth = model_num_layers
    if depth is None:
        depth = active_model_cfg.get("num_layers", config.get("backbone", {}).get("num_layers"))
    depth = int(depth or 0)

    if isinstance(law_cfg, dict):
        if depth > 0:
            batch_ref = float(law_cfg.get("reference_total_batch_size", 6144))
            depth_ref = float(law_cfg.get("reference_num_layers", 12))
            depth_exponent = float(law_cfg.get("depth_exponent", 2.0 / 3.0))
            batch_lr_scale = float(law_cfg.get("batch_lr_scale", 1.0))
            scaled = batch_lr_scale * batch_ref * ((depth / max(1.0, depth_ref)) ** depth_exponent)
            target_global_batch = max(1, int(round(scaled)))

    expected_grad_accum = max(
        1, math.ceil(target_global_batch / max(1, per_rank_batch * world))
    )
    achieved_global_batch = per_rank_batch * world * expected_grad_accum

    rec_opt: Dict[str, Any] = {
        "target_global_batch_size": int(target_global_batch),
    }
    rec_train: Dict[str, Any] = {}
    if cpu_oom_rec:
        rec_opt = {**rec_opt, **cpu_oom_rec}
    if per_rank_batch != original_per_rank_batch:
        rec_train["batch_size"] = int(per_rank_batch)

    compute_optimal_source = None
    compute_optimal_batch_ratio = None
    compute_optimal_meta: Dict[str, Any] = {}

    # Compute-optimal law:
    # - `fixed_budget`: keep fixed total tokens/sequences via max_steps ~ 1 / global_batch
    # - `token_param_ratio`: nanochat-style horizon via target_tokens = ratio * scaling_params
    # Optional hybrid floor:
    # - enforce a minimum effective epoch budget on top of token:param objective.
    # Learning-rate scaling is optional and remains config-driven.
    compute_cfg = None
    if h100_preset_active:
        h100_compute_cfg = h100_cfg.get("compute_optimal", {})
        if isinstance(h100_compute_cfg, dict) and bool(h100_compute_cfg.get("enabled", False)):
            compute_cfg = h100_compute_cfg
            compute_optimal_source = "h100_8x_preset.compute_optimal"

    if compute_cfg is None:
        global_compute_cfg = optimization.get("compute_optimal_scaling", {})
        if isinstance(global_compute_cfg, dict) and bool(global_compute_cfg.get("enabled", False)):
            compute_cfg = global_compute_cfg
            compute_optimal_source = "optimization.compute_optimal_scaling"

    if isinstance(compute_cfg, dict):
        ref_batch = float(compute_cfg.get("reference_total_batch_size", 0))
        if ref_batch <= 0:
            ref_batch = float(law_cfg.get("reference_total_batch_size", 0)) if isinstance(law_cfg, dict) else 0.0
        if ref_batch <= 0:
            ref_batch = float(target_global_batch)

        objective = str(compute_cfg.get("objective", "fixed_budget")).lower()
        ref_steps = int(compute_cfg.get("reference_max_steps", train_cfg.get("max_steps", 300000)))
        ref_lr = float(
            compute_cfg.get("reference_learning_rate", train_cfg.get("learning_rate", 3.0e-4))
        )
        lr_power = float(compute_cfg.get("lr_power", 0.5))
        min_lr = float(compute_cfg.get("min_learning_rate", 0.0))
        max_lr_raw = compute_cfg.get("max_learning_rate", None)
        max_lr = float(max_lr_raw) if max_lr_raw is not None else None
        step_round_multiple = int(compute_cfg.get("step_round_multiple", 10))

        warmup_fraction = float(compute_cfg.get("warmup_fraction", 0.01))
        min_warmup_steps = int(compute_cfg.get("min_warmup_steps", 200))
        max_warmup_steps = int(compute_cfg.get("max_warmup_steps", 20000))

        if ref_steps <= 0:
            ref_steps = int(train_cfg.get("max_steps", 300000) or 300000)

        use_achieved_batch = bool(compute_cfg.get("use_achieved_global_batch", True))
        batch_for_compute = achieved_global_batch if use_achieved_batch else target_global_batch

        compute_optimal_batch_ratio = float(batch_for_compute) / max(1.0, ref_batch)
        train_samples_est = float(compute_cfg.get("train_num_samples_estimate", 0))
        if train_samples_est <= 0:
            train_samples_est = float(config.get("data", {}).get("train_num_samples_estimate", 0))
        train_fraction = float(config.get("data", {}).get("train_fraction", 1.0))
        if train_fraction <= 0:
            train_fraction = 1.0
        effective_train_samples = (
            train_samples_est * min(1.0, max(0.0, train_fraction))
            if train_samples_est > 0
            else 0.0
        )
        min_effective_epochs = float(compute_cfg.get("min_effective_epochs", 0.0))

        if objective in {"token_param", "token_param_ratio", "nanochat_token_param_ratio"}:
            target_param_data_ratio = float(compute_cfg.get("target_param_data_ratio", 10.5))
            tokens_per_sample = float(compute_cfg.get("tokens_per_sample_estimate", 36.5228))
            tokens_per_sample = max(1.0e-8, tokens_per_sample)
            scaling_params = _estimate_scaling_params(
                pipeline_kind=pipeline_kind,
                model_cfg=active_model_cfg,
                compute_cfg=compute_cfg,
            )
            target_tokens = target_param_data_ratio * scaling_params
            total_batch_tokens = float(batch_for_compute) * tokens_per_sample
            recommended_steps = int(target_tokens // max(1.0e-8, total_batch_tokens))
            compute_optimal_meta = {
                "target_tokens": float(target_tokens),
                "total_batch_tokens": float(total_batch_tokens),
                "scaling_params": float(scaling_params),
                "target_param_data_ratio": float(target_param_data_ratio),
                "tokens_per_sample_estimate": float(tokens_per_sample),
                "compute_objective": "token_param_ratio",
            }
        else:
            recommended_steps = int(round(ref_steps / max(1.0e-8, compute_optimal_batch_ratio)))
            compute_optimal_meta = {"compute_objective": "fixed_budget"}

        if min_effective_epochs > 0 and effective_train_samples > 0:
            epoch_floor_steps = int(
                math.ceil((min_effective_epochs * effective_train_samples) / max(1.0e-8, float(batch_for_compute)))
            )
            recommended_steps = max(recommended_steps, epoch_floor_steps)
            compute_optimal_meta["min_effective_epochs"] = float(min_effective_epochs)
            compute_optimal_meta["effective_train_samples"] = float(effective_train_samples)
            compute_optimal_meta["epoch_floor_steps"] = int(epoch_floor_steps)

        min_steps = int(compute_cfg.get("min_steps", 1))
        max_steps_cap = int(compute_cfg.get("max_steps_cap", 0))
        recommended_steps = max(min_steps, recommended_steps)
        if max_steps_cap > 0:
            recommended_steps = min(max_steps_cap, recommended_steps)
        recommended_steps = max(1, recommended_steps)
        if step_round_multiple > 1:
            recommended_steps = max(
                step_round_multiple,
                int(round(recommended_steps / step_round_multiple) * step_round_multiple),
            )

        use_batch_scaled_lr = bool(compute_cfg.get("use_batch_scaled_learning_rate", True))
        if use_batch_scaled_lr:
            recommended_lr = ref_lr * (compute_optimal_batch_ratio ** lr_power)
            recommended_lr = max(min_lr, recommended_lr)
            if max_lr is not None and max_lr > 0:
                recommended_lr = min(max_lr, recommended_lr)
        else:
            recommended_lr = float(train_cfg.get("learning_rate", ref_lr))

        recommended_warmup = int(round(recommended_steps * warmup_fraction))
        recommended_warmup = max(min_warmup_steps, recommended_warmup)
        if max_warmup_steps > 0:
            recommended_warmup = min(max_warmup_steps, recommended_warmup)
        recommended_warmup = min(recommended_warmup, max(1, recommended_steps - 1))

        rec_train = {
            **rec_train,
            "max_steps": int(recommended_steps),
            "warmup_steps": int(recommended_warmup),
            "learning_rate": float(recommended_lr),
        }

    if pipeline_kind in {"sequence", "group_selfies"}:
        rec_opt["dynamic_padding"] = True
    if pipeline_kind == "sequence":
        rec_opt["length_bucket_sampler"] = True
        rec_opt["bucket_size_multiplier"] = int(
            optimization.get("bucket_size_multiplier", 50) or 50
        )

    if h100_preset_active:
        rec_opt["auto_batch_law"] = {
            "enabled": True,
            "reference_total_batch_size": float(h100_cfg.get("reference_total_batch_size", 8192)),
            "reference_num_layers": float(h100_cfg.get("reference_num_layers", 12)),
            "depth_exponent": float(h100_cfg.get("depth_exponent", 2.0 / 3.0)),
            "batch_lr_scale": float(h100_cfg.get("batch_lr_scale", 1.0)),
        }
        preset_optimizer = h100_cfg.get("optimizer", {})
        if isinstance(preset_optimizer, dict) and preset_optimizer:
            rec_opt["optimizer"] = copy.deepcopy(preset_optimizer)

    if world > 1 and gpu.get("is_hopper", False):
        rec_opt["compile_in_ddp"] = True

    if gpu.get("is_hopper", False) and gpu.get("supports_fp8", False):
        rec_opt["fp8_phase2_eval"] = {
            "enabled": True,
            "dtype": "float8_e4m3fn",
            "num_steps": 100,
            "warmup_steps": 10,
        }

    return {
        "world_size": int(world),
        "pipeline_kind": str(pipeline_kind),
        "original_per_rank_batch_size": int(original_per_rank_batch),
        "per_rank_batch_size": int(per_rank_batch),
        "memory_aware_batch_meta": memory_batch_meta,
        "current_grad_accumulation_steps": int(grad_accum),
        "current_global_batch_size": int(current_global_batch),
        "reference_world_size": int(reference_world),
        "target_global_batch_source": target_batch_source,
        "model_num_layers": int(model_num_layers) if model_num_layers is not None else None,
        "h100_preset_active": bool(h100_preset_active),
        "recommended_target_global_batch_size": int(target_global_batch),
        "recommended_achieved_global_batch_size": int(achieved_global_batch),
        "expected_grad_accumulation_steps_after_target": int(expected_grad_accum),
        "recommended_optimization": rec_opt,
        "compute_optimal_scaling_active": bool(compute_optimal_source),
        "compute_optimal_source": compute_optimal_source,
        "compute_optimal_batch_ratio": compute_optimal_batch_ratio,
        "compute_optimal_meta": compute_optimal_meta,
        "cpu_oom_guard_meta": cpu_oom_meta,
        "recommended_training_backbone": rec_train,
        "gpu": gpu,
        "system": system_info,
    }


def _merge_recommended_value(
    current_value: Any, recommended_value: Any, override_existing: bool
) -> tuple[Any, bool]:
    """Merge one optimization key and report if it changed."""
    if isinstance(recommended_value, dict):
        merged = copy.deepcopy(current_value) if isinstance(current_value, dict) else {}
        changed = not isinstance(current_value, dict)
        for key, value in recommended_value.items():
            if override_existing or key not in merged:
                if merged.get(key) != value:
                    merged[key] = copy.deepcopy(value)
                    changed = True
        return merged, changed

    if override_existing:
        changed = current_value != recommended_value
        return copy.deepcopy(recommended_value), changed

    if current_value is None:
        return copy.deepcopy(recommended_value), True

    return current_value, False


def apply_auto_step1_recommendations(
    config: Dict[str, Any],
    pipeline_kind: str,
    world_size: int,
    model_num_layers: Optional[int] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Apply optional runtime recommendations from optimization.auto_recommend_from_system."""
    opt_config = config.setdefault("optimization", {})
    auto_cfg = opt_config.get("auto_recommend_from_system", {})
    if not isinstance(auto_cfg, dict):
        return None
    if not bool(auto_cfg.get("enabled", False)):
        return None

    reference_world_size = int(auto_cfg.get("reference_world_size", 4))
    if reference_world_size <= 0:
        raise ValueError(
            "optimization.auto_recommend_from_system.reference_world_size must be > 0."
        )
    device_index = int(auto_cfg.get("device_index", 0))
    override_existing = bool(auto_cfg.get("override_existing", True))

    gpu_info = detect_gpu_info(device_index=device_index)
    summary = recommend_step1_optimization(
        config=config,
        pipeline_kind=pipeline_kind,
        world_size=world_size,
        reference_world_size=reference_world_size,
        gpu_info=gpu_info,
        model_num_layers=model_num_layers,
        model_config=model_config,
    )

    recommended_opt = summary["recommended_optimization"]
    applied_opt: Dict[str, Any] = {}
    for key, value in recommended_opt.items():
        merged, changed = _merge_recommended_value(
            current_value=opt_config.get(key),
            recommended_value=value,
            override_existing=override_existing,
        )
        if changed:
            opt_config[key] = merged
            applied_opt[key] = copy.deepcopy(merged)

    train_cfg = config.setdefault("training_backbone", {})
    recommended_train = summary.get("recommended_training_backbone", {})
    applied_train: Dict[str, Any] = {}
    for key, value in recommended_train.items():
        merged, changed = _merge_recommended_value(
            current_value=train_cfg.get(key),
            recommended_value=value,
            override_existing=override_existing,
        )
        if changed:
            train_cfg[key] = merged
            applied_train[key] = copy.deepcopy(merged)

    summary["applied_optimization"] = applied_opt
    summary["applied_training_backbone"] = applied_train
    summary["override_existing"] = bool(override_existing)
    summary["device_index"] = int(device_index)
    return summary
