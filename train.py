# -*- coding: utf-8 -*-
"""Training-only entry point for CL experiments.

Trains each task sequentially and writes a per-task checkpoint (``.pth``) plus
its class-context JSON (``classes_{task_id}.json``). Does not import any OOD
dataloader, evaluation manager, or W&B evaluation metric — all OOD/evaluation
work lives in ``evaluate.py``.

Usage:
    python train.py --config configs/cl/<experiment>.yaml [overrides]
    torchrun --nproc_per_node=<N> train.py --config configs/cl/<experiment>.yaml

``exp_name`` must still contain ``_OOD_`` so that ``evaluate.py`` can reuse
these checkpoints from the same config file.
"""
import gc
import os

import torch
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss

from cl_extensions.amp import enable_amp
from cl_extensions.ddp_plugin import DDPSamplerPlugin
from core.setup import build_optimizer, get_benchmark, init_model, setup_config
from managers.training_manager import TrainingManager
from utils.dist_utils import (
    DDPContext,
    is_dist_available_and_initialized,
    is_main_process,
)
from utils.factory import get_strategy_class
from utils.helpers import ExperimentGuard, set_seed


def _free_task_memory():
    """Release GPU and CPU memory between tasks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    # Distributed setup must happen before anything touches CUDA.
    # torchrun populates LOCAL_RANK / RANK / WORLD_SIZE before the script
    # starts; single-GPU launches take the no-op path inside setup_distributed.
    with DDPContext() as (local_rank, rank, world_size):
        _run(local_rank, rank, world_size, world_size > 1)


def _run(local_rank, rank, world_size, distributed):
    # 1. Configuration & Setup
    if is_main_process():
        print("--- 1. Setting up Training-Only Experiment ---")
    config = setup_config()
    set_seed(config.scenario.seed)

    # Auto-tune cuDNN conv algorithms for fixed input sizes (e.g. 224×224).
    torch.backends.cudnn.benchmark = True

    # 2. Data & Benchmark (no OOD loaders)
    if is_main_process():
        print("--- 2. Building Benchmark ---")
    benchmark, _ = get_benchmark(config)

    # 3. Model & Optimization
    if is_main_process():
        print("--- 3. Initializing Model ---")
    model, fc_layer = init_model(config, benchmark)
    # Use the per-process local_rank so each DDP worker places the model on
    # its own GPU. The module-level `device` is resolved at import time
    # (before setup_distributed), so it may point to the wrong ordinal.
    dev = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(dev)

    optimizer = build_optimizer(config, model)
    criterion = CrossEntropyLoss()

    # 4. Strategy — lightweight eval plugin (console-only on rank 0)
    if is_main_process():
        print("--- 4. Building Strategy ---")
    loggers = [InteractiveLogger()] if is_main_process() else []
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loggers=loggers,
    )

    strategy, strategy_type = get_strategy_class(
        config, model, optimizer, criterion, eval_plugin, fc_layer=fc_layer
    )
    if is_main_process():
        print(f"Strategy: {config.strategy.name} (type={strategy_type})")

    # 4a. AMP (mixed precision) via AMPMixin subclass (no monkey-patching)
    use_amp = str(getattr(config, "use_amp", False)).lower() == "true"
    if use_amp and torch.cuda.is_available():
        enable_amp(strategy)
        if is_main_process():
            print("Mixed precision (AMP) training enabled")

    # 4b. Multi-GPU: wrap in DDP and inject sampler plugin
    if distributed:
        if is_main_process():
            print(f"Using {world_size} GPUs with DistributedDataParallel!")
        strategy.model = torch.nn.parallel.DistributedDataParallel(
            strategy.model,
            device_ids=[local_rank],
            output_device=local_rank,
            # find_unused_parameters handles CL methods that selectively freeze
            # parts of the network (e.g. PackNet, EWC with frozen layers).
            find_unused_parameters=True,
            bucket_cap_mb=50,
        )
        ddp_num_workers = int(getattr(config, "num_workers", 4))
        strategy.plugins.append(
            DDPSamplerPlugin(
                rank=rank, world_size=world_size, num_workers=ddp_num_workers
            )
        )
        # Mark all plugins as DDP-compatible to suppress Avalanche's warning.
        # Our custom plugins declare this; Avalanche built-ins (LRScheduler,
        # BiC, …) are safe under DDP but don't declare it.
        for plugin in strategy.plugins:
            if not getattr(plugin, "supports_distributed", False):
                plugin.supports_distributed = True

    # 5. Training Manager (handles checkpoint save/load + context JSON)
    train_manager = TrainingManager(
        config,
        wandb_logger=None,
        strategy=strategy,
        benchmark=benchmark,
        device=dev,
        is_main_process=is_main_process(),
    )

    # 6. Training Loop
    train_stream = benchmark.train_stream
    n_tasks = len(train_stream)
    if is_main_process():
        print(f"\n--- Starting Training Loop ({n_tasks} tasks) ---")

    for task_id, train_task in enumerate(train_stream):
        if is_main_process():
            print(f"\n{'='*30}\n===> Task {task_id + 1}/{n_tasks}\n{'='*30}")
            print(
                "Classes in this experience:",
                train_task.classes_in_this_experience,
            )

        with ExperimentGuard(f"Training Task {task_id}", wandb_logger=None):
            # TrainingManager.train_task writes ckpt_{task_id}.pth and the
            # companion classes_{task_id}.json via _save_context_file.
            train_manager.train_task(train_task, task_id)

        # Barrier before GC: prevents a fast worker from calling
        # torch.cuda.empty_cache() while a slow worker is still inside an
        # active NCCL collective (e.g. allreduce on the last backward pass).
        if is_dist_available_and_initialized():
            torch.distributed.barrier()
        _free_task_memory()
        if torch.cuda.is_available() and is_main_process():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(
                f"[Memory] After task {task_id}: "
                f"allocated={allocated:.2f}GB  reserved={reserved:.2f}GB"
            )

    if is_main_process():
        print("\n--- Training complete. All checkpoints saved. ---")
        ckpt_dir = config.scenario.ckpt_dir
        print(f"Checkpoints are in: {os.path.abspath(ckpt_dir)}")


if __name__ == "__main__":
    main()
