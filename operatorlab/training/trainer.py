"""Main Trainer class for neural operator training.

Handles the training loop, validation, checkpointing, and logging.
AMP and distributed training are integrated (controlled by config flags).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from datetime import datetime, timezone

from operatorlab.models.base import NeuralOperator
from operatorlab.registry.experiment import _get_git_hash
from operatorlab.training.losses import OperatorLoss, h1_loss, relative_l2

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Results from a training run."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_l2_errors: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    total_time: float = 0.0

    @property
    def initial_train_loss(self) -> float:
        return self.train_losses[0] if self.train_losses else float("inf")

    @property
    def final_train_loss(self) -> float:
        return self.train_losses[-1] if self.train_losses else float("inf")


@dataclass
class EvalResult:
    """Results from an evaluation pass."""

    loss: float = 0.0
    l2_error: float = 0.0
    h1_error: float = 0.0
    n_samples: int = 0
    resolution: Optional[int] = None


class Trainer:
    """Neural operator trainer with AMP, gradient clipping, and checkpointing.

    Args:
        model: Neural operator model to train.
        optimizer: PyTorch optimizer.
        scheduler: Optional learning rate scheduler.
        loss_fn: Combined loss function.
        device: Device string ('cpu', 'cuda', 'cuda:0', etc.).
        use_amp: Whether to use automatic mixed precision.
        grad_clip: Maximum gradient norm for clipping (0 to disable).
        log_every: Log training metrics every N steps.
        checkpoint_dir: Directory for saving checkpoints.
        registry: Optional experiment registry for logging.
    """

    def __init__(
        self,
        model: NeuralOperator,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        loss_fn: Optional[OperatorLoss] = None,
        device: str = "cpu",
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_every: int = 50,
        checkpoint_dir: Path = Path("checkpoints/"),
        registry: object | None = None,
        config: Optional[dict] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or OperatorLoss()
        self.device = device
        self.use_amp = use_amp and device != "cpu"
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.registry = registry
        self.config = config

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Ensure checkpoint dir exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
    ) -> TrainingResult:
        """Run the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            n_epochs: Number of training epochs.

        Returns:
            TrainingResult with loss curves and best metrics.
        """
        result = TrainingResult()
        start_time = time.time()

        for epoch in range(1, n_epochs + 1):
            # Train one epoch
            train_loss = self._train_epoch(train_loader, epoch)
            result.train_losses.append(train_loss)

            # Validate
            eval_result = self.evaluate(val_loader)
            result.val_losses.append(eval_result.loss)
            result.val_l2_errors.append(eval_result.l2_error)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log epoch summary
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | val_l2=%.6f | lr=%.2e",
                epoch, n_epochs, train_loss, eval_result.loss,
                eval_result.l2_error, lr,
            )

            # Save checkpoint if best
            if eval_result.loss < result.best_val_loss:
                result.best_val_loss = eval_result.loss
                result.best_epoch = epoch
                self._save_checkpoint(epoch, eval_result, "best.pt", train_loss=train_loss)

            # Always save last checkpoint
            self._save_checkpoint(epoch, eval_result, "last.pt", train_loss=train_loss)

            # Log to registry
            if self.registry is not None and hasattr(self.registry, "log_metrics"):
                self.registry.log_metrics(epoch, {
                    "train_loss": train_loss,
                    "val_loss": eval_result.loss,
                    "val_l2": eval_result.l2_error,
                    "lr": lr,
                })

        result.total_time = time.time() - start_time
        logger.info(
            "Training complete in %.1fs | best_val_loss=%.6f at epoch %d",
            result.total_time, result.best_val_loss, result.best_epoch,
        )
        return result

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Train for one epoch.

        Args:
            loader: Training data loader.
            epoch: Current epoch number (for logging).

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (a, u, grid) in enumerate(pbar):
            a = a.to(self.device)       # (batch, H, W, in_ch)
            u = u.to(self.device)       # (batch, H, W, out_ch)
            grid = grid.to(self.device)  # (H, W, 2) or (batch, H, W, 2)

            # Add batch dim to grid if needed
            if grid.ndim == 3:
                grid = grid.unsqueeze(0).expand(a.shape[0], -1, -1, -1)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    pred = self.model(a, grid)    # (batch, H, W, out_ch)
                    loss = self.loss_fn(pred, u)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(a, grid)
                loss = self.loss_fn(pred, u)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip,
                    )
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % self.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        resolution: Optional[int] = None,
    ) -> EvalResult:
        """Evaluate the model on a dataset.

        Args:
            loader: Data loader for evaluation.
            resolution: Optional resolution override for testing (unused here,
                        handled by the dataset).

        Returns:
            EvalResult with loss and L2 error metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_l2 = 0.0
        total_h1 = 0.0
        n_batches = 0

        for a, u, grid in loader:
            a = a.to(self.device)
            u = u.to(self.device)
            grid = grid.to(self.device)

            if grid.ndim == 3:
                grid = grid.unsqueeze(0).expand(a.shape[0], -1, -1, -1)

            pred = self.model(a, grid)
            loss = self.loss_fn(pred, u)
            l2 = relative_l2(pred, u)
            h1 = h1_loss(pred, u)

            total_loss += loss.item()
            total_l2 += l2.item()
            total_h1 += h1.item()
            n_batches += 1

        n = max(n_batches, 1)
        return EvalResult(
            loss=total_loss / n,
            l2_error=total_l2 / n,
            h1_error=total_h1 / n,
            n_samples=len(loader.dataset),
            resolution=resolution,
        )

    def _save_checkpoint(
        self,
        epoch: int,
        eval_result: EvalResult,
        filename: str,
        train_loss: float = 0.0,
    ) -> None:
        """Save a training checkpoint with full metadata and config.

        Args:
            epoch: Current epoch number.
            eval_result: Current evaluation results.
            filename: Checkpoint filename (e.g. 'best.pt', 'last.pt').
            train_loss: Training loss for the current epoch.
        """
        path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "config": self.config or {},
            "metrics": {
                "train_loss": train_loss,
                "val_loss": eval_result.loss,
                "val_l2": eval_result.l2_error,
                "val_h1": eval_result.h1_error,
            },
            "git_hash": _get_git_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint, path)
        logger.debug("Saved checkpoint to %s", path)
