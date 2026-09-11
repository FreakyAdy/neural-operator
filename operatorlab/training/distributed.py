"""Distributed Data Parallel (DDP) training helpers.

Provides setup/cleanup utilities for multi-GPU training with
torch.nn.parallel.DistributedDataParallel.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize the distributed process group.

    Args:
        rank: Rank of the current process.
        world_size: Total number of processes.
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logger.info("Initialized distributed process group: rank=%d/%d", rank, world_size)


def cleanup_distributed() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed distributed process group")


def wrap_ddp(model: torch.nn.Module, rank: int) -> DDP:
    """Wrap a model in DistributedDataParallel.

    Args:
        model: The model to wrap.
        rank: GPU rank for this process.

    Returns:
        DDP-wrapped model.
    """
    model = model.to(rank)
    return DDP(model, device_ids=[rank])


def is_main_process() -> bool:
    """Check if the current process is the main (rank 0) process.

    Returns:
        True if rank is 0 or distributed is not initialized.
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier() -> None:
    """Synchronization barrier across all distributed processes."""
    if dist.is_initialized():
        dist.barrier()
