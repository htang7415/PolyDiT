"""Trainer for graph diffusion backbone model."""

import warnings
import time
import math
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
from shared.optimizer_setup import build_step1_backbone_optimizer

NAT_TO_BPB = 1.0 / math.log(2.0)


def _to_float(value, name: str) -> float:
    """Convert config value to float with a clear error on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric, got {value!r} ({type(value).__name__})")


def _to_int(value, name: str) -> int:
    """Convert config value to int with a clear error on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be integer-like, got {value!r} ({type(value).__name__})")


def _is_cuda_device(device) -> bool:
    """Return True if the provided device resolves to CUDA."""
    try:
        return torch.device(device).type == 'cuda'
    except (TypeError, ValueError):
        return str(device).startswith('cuda')


def _supports_torch_compile(device) -> bool:
    """Return True if torch.compile can safely run on the current GPU."""
    if not _is_cuda_device(device) or not torch.cuda.is_available():
        return False
    try:
        dev = torch.device(device)
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        major, _minor = torch.cuda.get_device_capability(index)
    except Exception:
        return False
    return major >= 7


class GraphBackboneTrainer:
    """Trainer for graph-based discrete masking diffusion backbone.

    Handles training with (X, E, M) graph tensors instead of sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        output_dir: str = 'results',
        step_dir: str = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        local_rank: Optional[int] = None
    ):
        """Initialize trainer.

        Args:
            model: Graph diffusion model (backbone + diffusion process).
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            config: Training configuration.
            device: Device for training.
            output_dir: Output directory for shared artifacts (checkpoints).
            step_dir: Step-specific output directory for metrics/figures.
            distributed: Whether to use DistributedDataParallel.
            rank: Global rank.
            world_size: Total number of ranks.
            local_rank: Local rank (GPU index).
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main_process = (not self.distributed) or self.rank == 0
        self.output_dir = Path(output_dir)
        self.step_dir = Path(step_dir) if step_dir else self.output_dir

        # Create output directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.step_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Optimization config
        opt_config = config.get('optimization', {})
        self.use_amp = opt_config.get('use_amp', False) and _is_cuda_device(device)
        self.compile_model = opt_config.get('compile_model', False)
        self.compile_in_ddp = bool(opt_config.get('compile_in_ddp', False))
        self.grad_accum_steps = opt_config.get('gradient_accumulation_steps', 1)
        self.fp8_phase2_eval_cfg = opt_config.get('fp8_phase2_eval', {})

        # Enable cuDNN benchmark for consistent input sizes
        if opt_config.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True

        # Suppress SequentialLR deprecation warning
        warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Compile model for faster execution (guard against older GPUs)
        if self.compile_model and self.distributed and not self.compile_in_ddp:
            warnings.warn(
                "torch.compile disabled for DDP by config. "
                "Set optimization.compile_in_ddp=true to enable."
            )
            self.compile_model = False
        if self.compile_model and _is_cuda_device(device):
            if not _supports_torch_compile(device):
                warnings.warn("torch.compile disabled: GPU compute capability < 7.0")
                self.compile_model = False
            else:
                print("Compiling model with torch.compile()...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
        if self.distributed:
            self.model = self._wrap_ddp(self.model)

        # Training config
        train_config = config['training_backbone']
        self.learning_rate = _to_float(train_config['learning_rate'], 'learning_rate')
        self.weight_decay = _to_float(train_config['weight_decay'], 'weight_decay')
        self.warmup_steps = _to_int(train_config['warmup_steps'], 'warmup_steps')
        self.max_steps = _to_int(train_config['max_steps'], 'max_steps')
        self.gradient_clip_norm = _to_float(train_config['gradient_clip_norm'], 'gradient_clip_norm')
        self.eval_every = _to_int(train_config['eval_every'], 'eval_every')
        self.save_every = _to_int(train_config['save_every'], 'save_every')
        self.num_epochs = _to_int(train_config.get('num_epochs', 50), 'num_epochs')

        ckpt_cfg = config.get('checkpointing', {})
        self.save_best_only = ckpt_cfg.get('save_best_only', True)
        self.save_last = ckpt_cfg.get('save_last', False)
        self.save_periodic = ckpt_cfg.get('save_periodic', False)

        # Initialize optimizer (AdamW or Muon+AdamW multi-group, config-driven).
        self.optimizer, optimizer_info = build_step1_backbone_optimizer(
            model=self.model,
            optimization_config=opt_config,
            base_learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.is_main_process and optimizer_info.get("type") == "muon_adamw":
            print(
                "Using Muon+AdamW optimizer groups: "
                f"muon_tensors={optimizer_info.get('num_muon_tensors')}, "
                f"adamw_tensors={optimizer_info.get('num_adamw_tensors')}, "
                f"adamw_lr={optimizer_info.get('adamw_lr'):.3e}, "
                f"muon_lr={optimizer_info.get('muon_lr'):.3e}"
            )

        # Initialize scheduler (warmup + cosine decay)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.train_node_losses = []
        self.train_edge_losses = []
        self.val_losses = []
        self.val_node_losses = []
        self.val_edge_losses = []
        self.val_steps = []
        self.learning_rates = []
        self.epoch_train_losses = []
        self.epoch_train_node_losses = []
        self.epoch_train_edge_losses = []
        self.epoch_val_losses = []
        self.epoch_val_node_losses = []
        self.epoch_val_edge_losses = []

    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        """Wrap model with DistributedDataParallel when enabled."""
        if not self.distributed or not dist.is_available() or not dist.is_initialized():
            return model
        if _is_cuda_device(self.device):
            device_index = torch.device(self.device).index
            return DDP(model, device_ids=[device_index], output_device=device_index)
        return DDP(model)

    def _get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get a clean state_dict for saving (strip DDP/compile wrappers)."""
        model = self.model
        if isinstance(model, DDP):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model.state_dict()

    def _reduce_mean(self, value: float) -> float:
        """Average a scalar across ranks when using DDP."""
        if not self.distributed or not dist.is_available() or not dist.is_initialized():
            return value
        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return (tensor / self.world_size).item()

    def _maybe_mark_cudagraph_step_begin(self) -> None:
        """Mark the beginning of a cudagraph step if supported."""
        if not self.compile_model or not _is_cuda_device(self.device):
            return

        compiler_mod = getattr(torch, "compiler", None)
        if compiler_mod is None:
            return

        mark_step = getattr(compiler_mod, "cudagraph_mark_step_begin", None)
        if mark_step is None:
            return

        mark_step()

    def train(self) -> Dict:
        """Run training loop.

        Returns:
            Training history.
        """
        if self.is_main_process:
            print(f"Starting graph diffusion training for {self.num_epochs} epochs...")
            print(f"Train batches: {len(self.train_dataloader)}")
            print(f"Val batches: {len(self.val_dataloader)}")

        for epoch in range(self.num_epochs):
            # Training epoch
            train_loss, node_loss, edge_loss = self._train_epoch(epoch)

            # Validation
            val_loss, val_node_loss, val_edge_loss = self._validate()
            self.epoch_train_losses.append(train_loss)
            self.epoch_train_node_losses.append(node_loss)
            self.epoch_train_edge_losses.append(edge_loss)
            self.epoch_val_losses.append(val_loss)
            self.epoch_val_node_losses.append(val_node_loss)
            self.epoch_val_edge_losses.append(val_edge_loss)

            # Save checkpoint
            self._save_checkpoint(val_loss, epoch)

            # Barrier after checkpoint to prevent rank drift
            if self.distributed and dist.is_available() and dist.is_initialized():
                dist.barrier()

            if self.is_main_process:
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f} (Node: {node_loss:.4f}, Edge: {edge_loss:.4f}) - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            if self.global_step >= self.max_steps:
                if self.is_main_process:
                    print(f"Reached max steps ({self.max_steps}), stopping training.")
                break

        # Save final checkpoint
        self._save_checkpoint(val_loss, epoch, final=True)

        # Save training history
        self._save_history()

        fp8_phase2_eval = self._run_fp8_phase2_eval()

        return {
            'train_losses': self.train_losses,
            'train_node_losses': self.train_node_losses,
            'train_edge_losses': self.train_edge_losses,
            'val_losses': self.val_losses,
            'val_node_losses': self.val_node_losses,
            'val_edge_losses': self.val_edge_losses,
            'learning_rates': self.learning_rates,
            'epoch_train_losses': self.epoch_train_losses,
            'epoch_val_losses': self.epoch_val_losses,
            'best_val_loss': self.best_val_loss,
            'fp8_phase2_eval': fp8_phase2_eval,
        }

    def _train_epoch(self, epoch: int):
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (total_loss, node_loss, edge_loss).
        """
        self.model.train()
        total_loss = 0.0
        total_node_loss = 0.0
        total_edge_loss = 0.0
        num_batches = 0

        if self.distributed and hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(epoch)
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", disable=not self.is_main_process)
        for batch_idx, batch in enumerate(pbar):
            if self.global_step >= self.max_steps:
                break

            is_last_batch = (batch_idx + 1) == len(self.train_dataloader)
            should_step = (
                ((batch_idx + 1) % self.grad_accum_steps == 0) or
                is_last_batch
            )
            loss, node_loss, edge_loss = self._train_step(batch, should_step=should_step)
            total_loss += loss
            total_node_loss += node_loss
            total_edge_loss += edge_loss
            num_batches += 1

            if self.is_main_process:
                self.train_losses.append(loss)
                self.train_node_losses.append(node_loss)
                self.train_edge_losses.append(edge_loss)
                self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'node': f'{node_loss:.4f}',
                    'edge': f'{edge_loss:.4f}'
                })

            if should_step:
                self.global_step += 1

                # Periodic validation
                if self.global_step > 0 and self.global_step % self.eval_every == 0:
                    val_loss, val_node, val_edge = self._validate()
                    self.model.train()
                    if self.is_main_process:
                        self.val_losses.append(val_loss)
                        self.val_node_losses.append(val_node)
                        self.val_edge_losses.append(val_edge)
                        self.val_steps.append(self.global_step)
                        self._save_checkpoint(val_loss, epoch)
                    if self.distributed and dist.is_available() and dist.is_initialized():
                        dist.barrier()

                # Periodic save
                if (not self.save_best_only and self.save_periodic and self.global_step > 0 and self.global_step % self.save_every == 0):
                    self._save_periodic_checkpoint(epoch)
                    if self.distributed and dist.is_available() and dist.is_initialized():
                        dist.barrier()

        # Barrier before final reduce to ensure all ranks exit loop together
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

        avg_loss = total_loss / max(num_batches, 1)
        avg_node = total_node_loss / max(num_batches, 1)
        avg_edge = total_edge_loss / max(num_batches, 1)

        return (
            self._reduce_mean(avg_loss),
            self._reduce_mean(avg_node),
            self._reduce_mean(avg_edge)
        )

    def _train_step(self, batch: Dict[str, torch.Tensor], should_step: bool):
        """Single training step for graph data.

        Args:
            batch: Batch with X, E, M tensors.
            should_step: Whether to apply optimizer/scheduler updates this micro-batch.

        Returns:
            Tuple of (total_loss, node_loss, edge_loss).
        """
        X = batch['X'].to(self.device)
        E = batch['E'].to(self.device)
        M = batch['M'].to(self.device)

        # Skip DDP gradient sync on accumulation microsteps to reduce communication overhead.
        sync_context = nullcontext()
        if self.distributed and isinstance(self.model, DDP) and not should_step:
            sync_context = self.model.no_sync()
        with sync_context:
            # Forward pass with AMP
            self._maybe_mark_cudagraph_step_begin()
            with autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                outputs = self.model(X, E, M)
                loss = outputs['loss'] / self.grad_accum_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

        if should_step:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm
            )

            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return (
            loss.item() * self.grad_accum_steps,
            outputs['node_loss'].item(),
            outputs['edge_loss'].item()
        )

    def _validate(self):
        """Run validation.

        Returns:
            Tuple of (total_loss, node_loss, edge_loss).
        """
        self.model.eval()
        total_loss = 0.0
        total_node_loss = 0.0
        total_edge_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                X = batch['X'].to(self.device)
                E = batch['E'].to(self.device)
                M = batch['M'].to(self.device)

                self._maybe_mark_cudagraph_step_begin()
                with autocast('cuda', dtype=torch.bfloat16, enabled=self.use_amp):
                    outputs = self.model(X, E, M)

                total_loss += outputs['loss'].item()
                total_node_loss += outputs['node_loss'].item()
                total_edge_loss += outputs['edge_loss'].item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_node = total_node_loss / max(num_batches, 1)
        avg_edge = total_edge_loss / max(num_batches, 1)

        return (
            self._reduce_mean(avg_loss),
            self._reduce_mean(avg_node),
            self._reduce_mean(avg_edge)
        )

    def _benchmark_autocast_eval(self, dtype: torch.dtype, num_steps: int, warmup_steps: int) -> Dict[str, float]:
        """Benchmark graph eval throughput/loss under a target autocast dtype."""
        self.model.eval()
        losses: List[float] = []
        node_losses: List[float] = []
        edge_losses: List[float] = []
        elapsed_sec = 0.0
        node_count = 0

        total_iters = warmup_steps + num_steps
        data_iter = iter(self.val_dataloader)
        device_obj = torch.device(self.device)
        device_index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()

        with torch.no_grad():
            for step_idx in range(total_iters):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.val_dataloader)
                    batch = next(data_iter)

                X = batch['X'].to(self.device)
                E = batch['E'].to(self.device)
                M = batch['M'].to(self.device)

                torch.cuda.synchronize(device_index)
                start = time.perf_counter()
                with autocast('cuda', dtype=dtype, enabled=True):
                    outputs = self.model(X, E, M)
                torch.cuda.synchronize(device_index)
                elapsed = time.perf_counter() - start

                if step_idx >= warmup_steps:
                    elapsed_sec += elapsed
                    losses.append(float(outputs['loss'].item()))
                    node_losses.append(float(outputs['node_loss'].item()))
                    edge_losses.append(float(outputs['edge_loss'].item()))
                    node_count += int(M.sum().item())

        eps = 1e-12
        return {
            'mean_loss': float(np.mean(losses)) if losses else float('nan'),
            'mean_node_loss': float(np.mean(node_losses)) if node_losses else float('nan'),
            'mean_edge_loss': float(np.mean(edge_losses)) if edge_losses else float('nan'),
            'nodes_per_sec': node_count / max(elapsed_sec, eps),
            'steps_per_sec': num_steps / max(elapsed_sec, eps),
            'elapsed_sec': elapsed_sec,
            'nodes': float(node_count),
        }

    def _save_fp8_phase2_eval(self, metrics: Dict[str, float]) -> None:
        """Save FP8 phase-2 evaluation metrics."""
        if not self.is_main_process:
            return
        fp8_df = pd.DataFrame([metrics])
        fp8_df.to_csv(self.metrics_dir / 'fp8_phase2_eval.csv', index=False)

    def _run_fp8_phase2_eval(self) -> Optional[Dict[str, float]]:
        """Evaluate FP8 throughput/loss as an optional second phase."""
        cfg = self.fp8_phase2_eval_cfg if isinstance(self.fp8_phase2_eval_cfg, dict) else {}
        if not bool(cfg.get('enabled', False)):
            return None

        if not self.is_main_process:
            return None

        metrics: Dict[str, float] = {
            'enabled': True,
        }

        if not _is_cuda_device(self.device) or not torch.cuda.is_available():
            if self.is_main_process:
                print("Skipping FP8 phase-2 evaluation: CUDA is unavailable.")
            metrics['status'] = 'skipped_no_cuda'
            self._save_fp8_phase2_eval(metrics)
            return metrics

        dtype_name = str(cfg.get('dtype', 'float8_e4m3fn'))
        fp8_dtype = getattr(torch, dtype_name, None)
        if fp8_dtype is None:
            if self.is_main_process:
                print(f"Skipping FP8 phase-2 evaluation: torch.{dtype_name} is unavailable.")
            metrics['status'] = 'skipped_unsupported_dtype'
            metrics['requested_dtype'] = dtype_name
            self._save_fp8_phase2_eval(metrics)
            return metrics

        num_steps = max(1, int(cfg.get('num_steps', 100)))
        warmup_steps = max(0, int(cfg.get('warmup_steps', 10)))

        try:
            bf16_stats = self._benchmark_autocast_eval(torch.bfloat16, num_steps, warmup_steps)
            fp8_stats = self._benchmark_autocast_eval(fp8_dtype, num_steps, warmup_steps)
        except Exception as exc:
            if self.is_main_process:
                print(f"Skipping FP8 phase-2 evaluation due to runtime error: {exc}")
            metrics['status'] = 'skipped_runtime_error'
            metrics['error'] = str(exc)
            metrics['requested_dtype'] = dtype_name
            self._save_fp8_phase2_eval(metrics)
            return metrics

        metrics.update({
            'status': 'ok',
            'requested_dtype': dtype_name,
            'num_steps': float(num_steps),
            'warmup_steps': float(warmup_steps),
            'bf16_mean_loss': bf16_stats['mean_loss'],
            'bf16_nodes_per_sec': bf16_stats['nodes_per_sec'],
            'bf16_steps_per_sec': bf16_stats['steps_per_sec'],
            'fp8_mean_loss': fp8_stats['mean_loss'],
            'fp8_nodes_per_sec': fp8_stats['nodes_per_sec'],
            'fp8_steps_per_sec': fp8_stats['steps_per_sec'],
            'fp8_nodes_per_sec_speedup': fp8_stats['nodes_per_sec'] / max(bf16_stats['nodes_per_sec'], 1e-12),
            'fp8_loss_delta_vs_bf16': fp8_stats['mean_loss'] - bf16_stats['mean_loss'],
        })
        if self.is_main_process:
            print(
                "FP8 phase-2 evaluation complete: "
                f"speedup={metrics['fp8_nodes_per_sec_speedup']:.3f}x, "
                f"loss_delta={metrics['fp8_loss_delta_vs_bf16']:.6f}"
            )
        self._save_fp8_phase2_eval(metrics)
        return metrics

    def _save_checkpoint(self, val_loss: float, epoch: int, final: bool = False):
        """Save model checkpoint.

        Args:
            val_loss: Validation loss.
            epoch: Current epoch.
            final: Whether this is the final checkpoint.
        """
        if not self.is_main_process:
            return

        model_state = self._get_model_state()

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.checkpoint_dir / 'graph_backbone_best.pt')
            print(f"New best model saved with val_loss: {val_loss:.4f}")

        # Save final checkpoint
        if final and not self.save_best_only and self.save_last:
            torch.save(checkpoint, self.checkpoint_dir / 'graph_backbone_last.pt')

    def _save_periodic_checkpoint(self, epoch: int):
        """Save periodic checkpoint.

        Args:
            epoch: Current epoch.
        """
        if self.save_best_only or not self.save_periodic:
            return
        if not self.is_main_process:
            return

        model_state = self._get_model_state()

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_dir / f'graph_backbone_step_{self.global_step}.pt')

    def _save_history(self):
        """Save training history to CSV."""
        if not self.is_main_process:
            return
        # Round floats
        rounded_train_losses = [round(loss, 4) for loss in self.train_losses]
        rounded_node_losses = [round(loss, 4) for loss in self.train_node_losses]
        rounded_edge_losses = [round(loss, 4) for loss in self.train_edge_losses]
        rounded_learning_rates = [round(lr, 8) for lr in self.learning_rates]
        rounded_val_losses = [round(loss, 4) for loss in self.val_losses]
        rounded_train_bpb = [round(loss * NAT_TO_BPB, 4) for loss in self.train_losses]
        rounded_val_bpb = [round(loss * NAT_TO_BPB, 4) for loss in self.val_losses]

        # Create loss curve CSV
        train_df = pd.DataFrame({
            'step': list(range(len(self.train_losses))),
            'train_loss': rounded_train_losses,
            'train_bpb': rounded_train_bpb,
            'node_loss': rounded_node_losses,
            'edge_loss': rounded_edge_losses,
            'learning_rate': rounded_learning_rates
        })
        train_df.to_csv(self.metrics_dir / 'graph_backbone_loss_curve.csv', index=False)

        if self.val_losses and self.val_steps:
            paired_count = min(
                len(self.val_steps),
                len(self.val_losses),
                len(self.val_node_losses),
                len(self.val_edge_losses),
            )
            val_df = pd.DataFrame({
                'step': self.val_steps[:paired_count],
                'val_loss': rounded_val_losses[:paired_count],
                'val_bpb': rounded_val_bpb[:paired_count],
                'val_node_loss': [round(l, 4) for l in self.val_node_losses[:paired_count]],
                'val_edge_loss': [round(l, 4) for l in self.val_edge_losses[:paired_count]]
            })
            val_df.to_csv(self.metrics_dir / 'graph_backbone_val_loss.csv', index=False)

        if self.epoch_train_losses and self.epoch_val_losses:
            paired_count = min(len(self.epoch_train_losses), len(self.epoch_val_losses))
            epoch_df = pd.DataFrame({
                'epoch': list(range(1, paired_count + 1)),
                'train_loss': [round(loss, 4) for loss in self.epoch_train_losses[:paired_count]],
                'val_loss': [round(loss, 4) for loss in self.epoch_val_losses[:paired_count]],
                'train_bpb': [round(loss * NAT_TO_BPB, 4) for loss in self.epoch_train_losses[:paired_count]],
                'val_bpb': [round(loss * NAT_TO_BPB, 4) for loss in self.epoch_val_losses[:paired_count]],
                'train_node_loss': [round(loss, 4) for loss in self.epoch_train_node_losses[:paired_count]],
                'train_edge_loss': [round(loss, 4) for loss in self.epoch_train_edge_losses[:paired_count]],
                'val_node_loss': [round(loss, 4) for loss in self.epoch_val_node_losses[:paired_count]],
                'val_edge_loss': [round(loss, 4) for loss in self.epoch_val_edge_losses[:paired_count]],
            })
            epoch_df.to_csv(self.metrics_dir / 'graph_backbone_epoch_loss_curve.csv', index=False)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from step {self.global_step}")
