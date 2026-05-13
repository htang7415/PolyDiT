#!/usr/bin/env python
"""Run the baseline scaling pipeline for Step 1 and Step 2 only."""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.model_scales import (
    estimate_params,
    get_model_config,
    get_results_dir,
    get_training_config,
)
from src.utils.scaling_logger import ScalingLogger


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def model_type_for_project() -> str:
    return "graph" if project_root().name == "Bi_Diffusion_graph" else "sequence"


def run_step(script_name: str, model_size: str, extra_args: str = "") -> tuple[int, str, str]:
    cmd = f"python scripts/{script_name} --model_size {model_size} {extra_args}"
    print(f"\nRunning: {cmd}")
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"STDERR:\n{result.stderr}")
    return result.returncode, result.stdout, result.stderr


def check_backbone_checkpoint(results_dir: Path) -> bool:
    checkpoint_dir = results_dir / "step1_backbone" / "checkpoints"
    return any(
        (checkpoint_dir / name).exists()
        for name in ("backbone_best.pt", "graph_backbone_best.pt")
    )


def extract_step1_metrics(results_dir: Path) -> dict:
    metrics = {}
    metrics_file = results_dir / "step1_backbone" / "metrics" / "backbone_training_history.json"
    if metrics_file.exists():
        with metrics_file.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        if history.get("val_losses"):
            metrics["final_val_loss"] = history["val_losses"][-1]
            metrics["best_val_loss"] = min(history["val_losses"])
        if history.get("train_losses"):
            metrics["final_train_loss"] = history["train_losses"][-1]
    return metrics


def extract_step2_metrics(results_dir: Path) -> dict:
    metrics = {}
    metrics_file = results_dir / "step2_sampling" / "metrics" / "sampling_generative_metrics.csv"
    if metrics_file.exists():
        import pandas as pd

        df = pd.read_csv(metrics_file)
        if len(df) > 0:
            row = df.iloc[0]
            metrics["validity"] = row.get("validity", None)
            metrics["uniqueness"] = row.get("uniqueness", None)
            metrics["novelty"] = row.get("novelty", None)
            metrics["diversity"] = row.get("avg_diversity", None)
    return metrics


def load_vocab_size(config: dict, model_type: str) -> int:
    tokenizer_name = "graph_tokenizer.json" if model_type == "graph" else "tokenizer.json"
    tokenizer_path = Path(config["paths"]["results_dir"]) / tokenizer_name
    if not tokenizer_path.exists():
        print(f"Warning: tokenizer not found at {tokenizer_path}, using default vocab_size=100")
        return 100
    with tokenizer_path.open("r", encoding="utf-8") as handle:
        tokenizer_data = json.load(handle)
    if model_type == "graph" and "atom_vocab" in tokenizer_data:
        return len(tokenizer_data["atom_vocab"])
    if "token_to_id" in tokenizer_data:
        return len(tokenizer_data["token_to_id"])
    if isinstance(tokenizer_data, dict):
        return len(tokenizer_data)
    return 100


def archive_scaling_results(results_dir: Path) -> None:
    src = results_dir / "scaling_results.json"
    if not src.exists():
        return
    dest = results_dir / "scaling_results_base.json"
    if dest.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = results_dir / f"scaling_results_base_{timestamp}.json"
    shutil.copy2(src, dest)
    print(f"Archived scaling results to: {dest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline scaling steps 1-2 with logging")
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=["small", "medium", "large", "xl"],
        help="Model size preset",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples for step 2")
    parser.add_argument("--skip_step1", action="store_true", help="Skip step 1 backbone training")
    parser.add_argument("--skip_step2", action="store_true", help="Skip step 2 sampling")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_type = model_type_for_project()

    results_dir = Path(get_results_dir(args.model_size, config["paths"]["results_dir"]))
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = ScalingLogger(results_dir, args.model_size)
    model_config = get_model_config(args.model_size, config, model_type=model_type)
    training_config = get_training_config(args.model_size, config, model_type=model_type)
    vocab_size = load_vocab_size(config, model_type)
    num_params = estimate_params(model_config, vocab_size, model_type=model_type)
    logger.log_model_config(model_config, training_config, num_params)

    print("=" * 60)
    print(f"SCALING EXPERIMENT: {args.model_size.upper()} ({model_type})")
    print("=" * 60)
    print(f"Parameters: ~{num_params:,}")
    print(f"Results dir: {results_dir}")
    print("=" * 60)

    if not args.skip_step1:
        logger.start_step("step1_backbone")
        ret, _stdout, stderr = run_step("step1_train_backbone.py", args.model_size)
        if ret != 0:
            logger.log_error("step1_backbone", f"Step failed with return code {ret}\nSTDERR: {stderr}")
            logger.finalize()
            return 1
        logger.end_step("step1_backbone", extract_step1_metrics(results_dir))
    else:
        logger.end_step("step1_backbone", status="skipped")

    if not args.skip_step2:
        if not check_backbone_checkpoint(results_dir):
            print(f"ERROR: Backbone checkpoint not found in {results_dir}/step1_backbone/checkpoints/")
            logger.log_error("step2_sampling", "Missing backbone checkpoint")
            logger.finalize()
            return 1
        logger.start_step("step2_sampling")
        ret, _stdout, stderr = run_step(
            "step2_sample_and_evaluate.py",
            args.model_size,
            f"--num_samples {args.num_samples}",
        )
        if ret != 0:
            logger.log_error("step2_sampling", f"Step failed with return code {ret}\nSTDERR: {stderr}")
            logger.finalize()
            return 1
        logger.end_step("step2_sampling", extract_step2_metrics(results_dir))
    else:
        logger.end_step("step2_sampling", status="skipped")

    logger.finalize()
    archive_scaling_results(results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
