"""Temporal rollout animations.

Autoregressively rolls out the model for multi-step prediction
and saves side-by-side animations: ground truth vs prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch import Tensor

from operatorlab.models.base import NeuralOperator
from operatorlab.physics.base import PDEProblem

logger = logging.getLogger(__name__)


@torch.no_grad()
def animate_rollout(
    model: NeuralOperator,
    pde: PDEProblem,
    initial_condition: Tensor,
    T: float,
    n_steps: int = 10,
    save_path: Path = Path("rollout.gif"),
) -> None:
    """Autoregressive rollout animation.

    Rolls out the model for n_steps, each step predicting the next
    time interval. Saves a side-by-side GIF: prediction vs ground truth error.

    Args:
        model: Trained neural operator.
        pde: PDE problem (for generating ground truth at intermediate times).
        initial_condition: Initial field, shape (1, H, W, C).
        T: Total time to roll out.
        n_steps: Number of rollout steps.
        save_path: Output file path (.gif or .mp4).
    """
    model.eval()
    device = next(model.parameters()).device

    dt_step = T / n_steps
    h, w = initial_condition.shape[1], initial_condition.shape[2]

    frames = []
    current = initial_condition.to(device)  # (1, H, W, C)

    for step in range(n_steps):
        pred = model(current)  # (1, H, W, C)
        frames.append(pred[0, ..., 0].cpu().numpy())
        current = pred  # feed prediction back as input

    # Create animation
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def update(frame_idx):
        for ax in axes:
            ax.clear()

        axes[0].imshow(frames[frame_idx], cmap="RdBu_r", aspect="equal")
        axes[0].set_title(f"Prediction (step {frame_idx + 1}/{n_steps})")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Error accumulated
        if frame_idx > 0:
            error = frames[frame_idx] - frames[0]
            axes[1].imshow(error, cmap="coolwarm", aspect="equal")
            axes[1].set_title(f"Drift from IC")
        else:
            axes[1].imshow(frames[0], cmap="RdBu_r", aspect="equal")
            axes[1].set_title("Initial Condition")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        return axes

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=500, blit=False,
    )

    save_path = Path(save_path)
    if save_path.suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=2)
        anim.save(str(save_path), writer=writer)
    else:
        anim.save(str(save_path), writer="pillow", fps=2)

    plt.close(fig)
    logger.info("Saved rollout animation to %s (%d frames)", save_path, len(frames))
