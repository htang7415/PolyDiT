"""Contrastive trainer for multi-view alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = _normalize(z1)
    z2 = _normalize(z2)
    logits = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return (loss_12 + loss_21) / 2.0


def _collate_batch(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    views = batch[0].keys() if batch else []
    out: Dict[str, torch.Tensor] = {}
    for view in views:
        stacked = np.stack([item[view] for item in batch])
        out[view] = torch.tensor(stacked, dtype=torch.float32)
    return out


def split_dataset(dataset: Dataset, val_ratio: float, seed: int = 42) -> Tuple[Dataset, Optional[Dataset]]:
    if val_ratio <= 0 or len(dataset) == 0:
        return dataset, None
    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


@dataclass
class TrainerConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 10
    temperature: float = 0.07
    view_dropout: float = 0.0
    val_ratio: float = 0.1
    log_every: int = 50


class ContrastiveTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        config: TrainerConfig,
        device: str,
        output_dir: Path,
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_dataset, self.val_dataset = split_dataset(dataset, config.val_ratio)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=_collate_batch,
            drop_last=True,
        )
        self.val_loader = None
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=_collate_batch,
                drop_last=False,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.history: List[Dict[str, float]] = []

    def _apply_view_dropout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.config.view_dropout <= 0:
            return batch
        keep = {}
        for view, emb in batch.items():
            if torch.rand(()) >= self.config.view_dropout:
                keep[view] = emb
        if len(keep) < 2:
            return batch
        return keep

    def _compute_batch_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch = self._apply_view_dropout(batch)
        projections = self.model.forward_multi(batch)
        views = list(projections.keys())
        if len(views) < 2:
            return torch.tensor(0.0, device=self.device)
        losses = []
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                losses.append(info_nce_loss(
                    projections[views[i]],
                    projections[views[j]],
                    self.config.temperature,
                ))
        return torch.stack(losses).mean()

    def _evaluate(self) -> Optional[float]:
        if self.val_loader is None:
            return None
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self._compute_batch_loss(batch)
                losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses)) if losses else None

    def train(self) -> None:
        best_val = float("inf")
        for epoch in range(1, self.config.max_epochs + 1):
            epoch_losses = []
            for step, batch in enumerate(self.train_loader, start=1):
                self.optimizer.zero_grad()
                loss = self._compute_batch_loss(batch)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                if step % self.config.log_every == 0:
                    print(f"Epoch {epoch} Step {step}: loss={loss.item():.4f}")

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            val_loss = self._evaluate()

            self.history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
            })

            self._save_checkpoint(epoch, train_loss, val_loss)
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                self._save_checkpoint(epoch, train_loss, val_loss, best=True)

        self._save_history()

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float], best: bool = False) -> None:
        ckpt = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        name = "alignment_best.pt" if best else "alignment_last.pt"
        torch.save(ckpt, self.output_dir / name)

    def _save_history(self) -> None:
        df = pd.DataFrame(self.history)
        df.to_csv(self.output_dir / "alignment_loss_curve.csv", index=False)


class EndToEndContrastiveTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        config: TrainerConfig,
        device: str,
        output_dir: Path,
        collate_fn,
        required_views: List[str],
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.required_views = required_views

        self.train_dataset, self.val_dataset = split_dataset(dataset, config.val_ratio)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, required_views),
            drop_last=False,
        )
        self.val_loader = None
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_fn(b, required_views),
                drop_last=False,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.history: List[Dict[str, float]] = []

    def _apply_view_dropout(self, projections: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.config.view_dropout <= 0:
            return projections
        keep = {}
        for view, emb in projections.items():
            if torch.rand(()) >= self.config.view_dropout:
                keep[view] = emb
        if len(keep) < 2:
            return projections
        return keep

    def _compute_batch_loss(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        if not batch:
            return None
        batch_on_device = {}
        for view, tensors in batch.items():
            batch_on_device[view] = {k: v.to(self.device) for k, v in tensors.items()}
        projections = self.model.forward_multi(batch_on_device)
        projections = self._apply_view_dropout(projections)
        views = list(projections.keys())
        if len(views) < 2:
            return None
        losses = []
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                losses.append(info_nce_loss(
                    projections[views[i]],
                    projections[views[j]],
                    self.config.temperature,
                ))
        return torch.stack(losses).mean()

    def _evaluate(self) -> Optional[float]:
        if self.val_loader is None:
            return None
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self._compute_batch_loss(batch)
                if loss is not None:
                    losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses)) if losses else None

    def train(self) -> None:
        best_val = float("inf")
        for epoch in range(1, self.config.max_epochs + 1):
            epoch_losses = []
            for step, batch in enumerate(self.train_loader, start=1):
                loss = self._compute_batch_loss(batch)
                if loss is None:
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                if step % self.config.log_every == 0:
                    print(f"Epoch {epoch} Step {step}: loss={loss.item():.4f}")

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            val_loss = self._evaluate()

            self.history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
            })

            self._save_checkpoint(epoch, train_loss, val_loss)
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                self._save_checkpoint(epoch, train_loss, val_loss, best=True)

        self._save_history()

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float], best: bool = False) -> None:
        ckpt = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        name = "alignment_best.pt" if best else "alignment_last.pt"
        torch.save(ckpt, self.output_dir / name)

    def _save_history(self) -> None:
        df = pd.DataFrame(self.history)
        df.to_csv(self.output_dir / "alignment_loss_curve.csv", index=False)
