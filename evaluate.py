# -*- coding: utf-8 -*-
"""Evaluation-only entry point for CL-OOD experiments.

Loads pretrained per-task checkpoints (written by train.py) and runs the full
OOD evaluation pipeline — accuracy, forgetting, OOD AUROC/FPR95/AUPR, and
representational analysis — without training.

Usage:
    python evaluate.py --config configs/cl/<experiment>.yaml [overrides]

Inputs:
    - {config.scenario.ckpt_dir}/{base_name}_{task_id}.pth  (model weights)
    - {config.scenario.ckpt_dir}/classes_{task_id}.json     (class context)

Both artifacts are written by train.py. Missing files raise FileNotFoundError
with an actionable message.

Invariants:
    - No TrainingManager, DDPSamplerPlugin, LRSchedulerPlugin, or GradScaler.
    - No optimizer state loaded into GPU memory.
    - Metrics must match the reference cl_ood.py run to within 1e-5.
"""
import json
import os

import torch
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models.dynamic_modules import avalanche_model_adaptation
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss

from core.setup import get_benchmark, init_model, setup_config
from managers.evaluation_manager import EvaluationManager
from managers.ood_manager import get_ood_manager
from managers.recorder_manager import RecorderManager
from openood.datasets import get_ood_dataloader
from utils import device
from utils.factory import get_strategy_class
from utils.helpers import (
    AverageIncrementalAccuracy,
    CustomWandbLogger,
    ExperimentGuard,
    StrategyStateHelper,
    set_seed,
)


def _ckpt_path(config, task_id):
    """Derive the per-task checkpoint path using the same naming convention
    as TrainingManager._get_ckpt_path."""
    assert (
        "_OOD_" in config.exp_name
    ), "Experiment name must contain '_OOD_' to separate base name."
    base_name = config.exp_name.split("_OOD_")[0]
    return os.path.join(
        config.scenario.ckpt_dir, f"{base_name}_{task_id}.pth"
    )


def _context_path(config, task_id):
    return os.path.join(
        config.scenario.ckpt_dir, f"classes_{task_id}.json"
    )


def _load_task_context(config, task_id):
    """Load the classes_{task_id}.json written by TrainingManager.

    Raises FileNotFoundError with the expected path if the context file is
    missing — this is the usual symptom of evaluating checkpoints produced
    by a pre-refactor training run.
    """
    path = _context_path(config, task_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing per-task context file: {path}. Run train.py on the "
            f"same config to produce classes_{task_id}.json, or copy the "
            f"file from an older checkpoint directory."
        )
    with open(path) as f:
        return json.load(f)


def main():
    # 1. Configuration & Setup
    print("--- 1. Setting up Evaluation-Only Experiment ---")
    config = setup_config()
    set_seed(config.scenario.seed)

    # 2. Logging
    project_name = os.getenv("WANDB_PROJECT", "cl_ood")
    wandb_logger = CustomWandbLogger(
        project_name=project_name, run_name=config.exp_name, config=config
    )
    interactive_logger = InteractiveLogger()

    # 3. Data & Benchmark + OOD loaders
    benchmark, eval_transform = get_benchmark(config)
    config.ood_dataset.shuffle = False
    ood_loader_dict = get_ood_dataloader(config)

    # 4. Model (no optimizer — evaluation runs in inference mode only)
    model, fc_layer = init_model(config, benchmark)
    model = model.to(device)

    # 5. Evaluation plugin. AverageIncrementalAccuracy is included so that
    # final/AvgAccuracy is populated — its hooks (before_eval / after_eval_iteration
    # / after_eval) all fire when strategy.eval(test_stream[:k+1]) is called per
    # task, which is exactly what the eval loop below does.
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        AverageIncrementalAccuracy(),
        loggers=[interactive_logger, wandb_logger],
    )

    # Strategy in eval-only mode. Avalanche requires an optimizer object even
    # for eval calls (it's touched during strategy construction), but no
    # optimizer state is loaded from checkpoints — build_optimizer is NOT
    # used here. We pass a dummy SGD over model.parameters() with lr=0 so
    # the strategy constructor is satisfied without any gradient machinery.
    dummy_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    criterion = CrossEntropyLoss()
    strategy, strategy_type = get_strategy_class(
        config, model, dummy_optimizer, criterion, eval_plugin, fc_layer=fc_layer
    )
    wandb_logger.wandb.config.update(
        {"Strategy Type": strategy_type, "Mode": "eval-only"},
        allow_val_change=True,
    )

    # 6. Managers

    # 6a. OOD Postprocessor Manager
    ood_manager = get_ood_manager(
        config, strategy, benchmark, ood_loader_dict, wandb_logger
    )

    # 6b. Recorder Manager
    rec_manager = RecorderManager(config, benchmark, wandb_logger)

    # 6c. Evaluation Manager
    eval_manager = EvaluationManager(
        config,
        strategy,
        benchmark,
        ood_manager,
        rec_manager,
        ood_loader_dict,
        eval_transform,
        wandb_logger=wandb_logger,
    )

    # 7. Resume evaluation state (if enabled)
    eval_start_task_id = rec_manager.maybe_resume_eval_state()

    # 8. Checkpoint iteration loop
    train_stream = benchmark.train_stream
    n_tasks = len(train_stream)
    print(f"\n--- Evaluating {n_tasks} tasks from {config.scenario.ckpt_dir} ---")

    for task_id, train_task in enumerate(train_stream):
        print(f"\n{'='*30}\n===> Evaluating Task {task_id + 1}/{n_tasks}\n{'='*30}")

        # Adapt the classifier head for this experience BEFORE the resume
        # skip. The head grows incrementally (task 0 adds 10 units, task 1
        # adds 10 more, ...). If we skip the adaptation for resumed tasks,
        # the head stays sized for the latest seen experience only and
        # safe_load_state_dict silently drops the wider classifier weights
        # in the saved checkpoint — pushing accuracy to chance.
        avalanche_model_adaptation(strategy.model, train_task)

        # Skip tasks already completed in a previous resumable run.
        if task_id < eval_start_task_id:
            print(f"Skipping evaluation for task {task_id} (resumed from state)")
            continue

        # Load per-task context (classes_in_this_experience) from disk so the
        # strategy's model head is grown consistently with training.
        ctx = _load_task_context(config, task_id)
        print(f"Classes in this experience (from context): {ctx['classes_in_this_experience']}")

        # Load the checkpoint. StrategyStateHelper.load restores model state,
        # strategy_attrs, plugin state, and iCarl/DER buffers — but NOT any
        # optimizer state (we never saved any).
        ckpt_path = _ckpt_path(config, task_id)
        if not StrategyStateHelper.load(strategy, ckpt_path, device):
            raise FileNotFoundError(
                f"Missing checkpoint: {ckpt_path}. Run train.py on the same "
                f"config to produce checkpoints, or verify scenario.ckpt_dir."
            )

        with ExperimentGuard(f"Evaluation Task {task_id}", wandb_logger):
            eval_manager.evaluate_task(task_id, n_tasks)
            rec_manager.save_eval_state(task_id)

    # 9. Final Analysis
    with ExperimentGuard("Generating Final Report", wandb_logger):
        rec_manager.generate_final_report(benchmark)

    wandb_logger.wandb.finish()


if __name__ == "__main__":
    main()
