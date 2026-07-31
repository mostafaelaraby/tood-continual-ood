"""
Distributed training utilities for DDP (DistributedDataParallel).

Usage with torchrun:
    torchrun --nproc_per_node=NUM_GPUS train_cl.py [args]

Each worker process has LOCAL_RANK / RANK / WORLD_SIZE set automatically by
torchrun before the script starts, so setup_distributed() is safe to call at
the top of main().
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Parallel wrapper types — kept in one place so every isinstance check
# stays in sync when new wrapper types are added.
# ---------------------------------------------------------------------------
_PARALLEL_TYPES = (
    torch.nn.DataParallel,
    torch.nn.parallel.DistributedDataParallel,
)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the raw module, stripping DataParallel or DDP wrappers."""
    return model.module if isinstance(model, _PARALLEL_TYPES) else model


# ---------------------------------------------------------------------------
# Process-group helpers
# ---------------------------------------------------------------------------


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank of the current process (0 in single-GPU mode)."""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Total number of processes (1 in single-GPU mode)."""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """True only on the rank-0 process (the one that owns logging/checkpointing)."""
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------


def setup_distributed() -> tuple[int, int, int]:
    """Initialise the NCCL process group from environment variables set by torchrun.

    Returns:
        (local_rank, global_rank, world_size)

    If the environment does not contain LOCAL_RANK (i.e. single-GPU launch),
    returns (0, 0, 1) without touching torch.distributed.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Guard: make sure the requested local_rank is actually visible to CUDA.
    # This catches the common mistake of launching N workers while
    # CUDA_VISIBLE_DEVICES only exposes fewer than N GPUs.
    n_visible = torch.cuda.device_count()
    if local_rank >= n_visible:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
        raise RuntimeError(
            f"local_rank={local_rank} but only {n_visible} GPU(s) are visible "
            f"(CUDA_VISIBLE_DEVICES={visible}). Make sure "
            f"CUDA_VISIBLE_DEVICES lists at least {world_size} GPUs, or "
            f"reduce --nproc_per_node to match the number of available GPUs."
        )

    if world_size <= 1:
        # torchrun with --nproc_per_node=1 still sets LOCAL_RANK=0.
        # No process group needed.
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return local_rank, rank, world_size


def cleanup_distributed() -> None:
    """Destroy the process group if one was initialised."""
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


class DDPContext:
    """Context manager wrapping the DDP lifecycle.

    Usage:
        with DDPContext() as (local_rank, rank, world_size):
            ...

    Guarantees ``cleanup_distributed()`` runs even if the body raises — this
    prevents zombie NCCL process groups from holding VRAM on crash.
    """

    def __enter__(self) -> tuple[int, int, int]:
        self.local_rank, self.rank, self.world_size = setup_distributed()
        return self.local_rank, self.rank, self.world_size

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        cleanup_distributed()
        return None
