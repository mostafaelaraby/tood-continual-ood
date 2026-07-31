import json
import os

import torch
import torch.distributed as dist
from avalanche.models.dynamic_modules import avalanche_model_adaptation

from utils.dist_utils import is_dist_available_and_initialized, unwrap_model
from utils.helpers import (
    FileLock,
    StrategyStateHelper,
    log_wandb_metrics,
    safe_load_state_dict,
    set_task_id,
)


class TrainingManager:
    """
    Manages the training process for Continual Learning tasks.

    Responsibilities:
    - Handling checkpoint loading and saving.
    - Adapting the model structure for new tasks (Avalanche adaptation).
    - loading initial weights for the first task if specified.
    - Executing the training strategy.
    """

    def __init__(
        self, config, wandb_logger, strategy, benchmark, device, is_main_process=True
    ):
        self.config = config
        self.wandb_logger = wandb_logger
        self.strategy = strategy
        self.benchmark = benchmark
        self.device = device
        self.is_main_process = is_main_process

        # Determine if the strategy is Joint training (offline) or Continual
        # This is often needed to skip CL-specific adaptations
        self.is_joint = self.config.strategy.name.lower() == "joint"

    def print_model_summary(self, task_id):
        """Prints a summary of the model architecture."""
        trainable_params = sum(
            p.numel()
            for p in self.strategy.model.parameters()
            if p.requires_grad
        )
        trainable_params_m = trainable_params / 1e6
        if task_id is None:
            log_wandb_metrics(
                self.wandb_logger,
                {"trainable_params": trainable_params_m},
                task_id,
            )
        else:
            log_wandb_metrics(
                self.wandb_logger,
                {f"trainable_params/Task_{task_id}": trainable_params_m},
                task_id,
            )
        print(f"Trainable Parameters: {trainable_params_m:.2f}M", flush=True)
        print("-" * 70, flush=True)

    def train_task(self, train_task, task_id):
        """
        Trains the model on a specific task or loads a checkpoint if available.

        Args:
            train_task: The Avalanche experience/task object containing the dataset.
            task_id (int): The index of the current task.
        """

        ckpt_path = self._get_ckpt_path(task_id)

        # 1. Prepare Model State
        self.strategy.model.train()
        self.print_model_summary(task_id)
        # 2. Handle CL-Specific Model Adaptation
        # If we are not doing Joint training, we need to adapt the model (e.g. grow head)
        if not self.is_joint:
            # Handle Task ID for Multi-Head settings
            if self.config.scenario.return_task_id:
                set_task_id(self.strategy.model, task_id)

            # Dynamically update the model (e.g. classifier) for the new classes in this task
            if os.path.exists(ckpt_path):
                avalanche_model_adaptation(
                    unwrap_model(self.strategy.model), train_task
                )

        # 3. Checkpoint Management
        # Try to load an existing checkpoint for this task to skip retraining
        if StrategyStateHelper.load(self.strategy, ckpt_path, self.device):
            if self.is_main_process:
                self._ensure_context_file(task_id, train_task)
            print(
                f"Checkpoint loaded from {ckpt_path}. Skipping training for task {task_id}."
            )
            return

        # 4. Initialization (First Task Only)
        # If this is the first task and we have a pre-defined init weight path, load it.
        if task_id == 0:
            self._load_init_weights()

        # 5. Execute Training
        print(f"Training on experience {task_id}...")
        set_task_id(self.strategy.model, None)
        num_workers = int(getattr(self.config, "num_workers", 4))
        self.strategy.train(
            train_task,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        # 6. Save Checkpoint — only rank-0 writes to disk.
        if self.is_main_process:
            with FileLock(ckpt_path):
                StrategyStateHelper.save(self.strategy, ckpt_path)
            self._save_context_file(task_id, train_task)

        # 7. DDP replay-buffer synchronisation.
        #
        # Problem: each DDP worker trains on a *different* data shard
        # (DistributedSampler), so storage_policy / iCaRL x_memory diverge
        # across processes after each experience. Non-rank-0 workers end up
        # replaying different exemplars than rank-0 in subsequent experiences,
        # which also breaks deterministic checkpoint resume (workers would load
        # rank-0's buffer but had a different buffer during the original run).
        #
        # Fix:
        #   (a) barrier ensures rank-0 has finished writing the buffer file
        #       before any worker tries to read it.
        #   (b) non-rank-0 workers reload the buffer file that rank-0 just
        #       wrote, making all workers' buffers identical going forward.
        if is_dist_available_and_initialized():
            dist.barrier()
            if not self.is_main_process:
                StrategyStateHelper._load_buffer_file(self.strategy, ckpt_path)

    def _get_ckpt_path(self, task_id):
        """Generates the file path for the checkpoint."""
        ckpt_dir = self.config.scenario.ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        # Enforce naming convention to separate OOD method names from base model names
        # This ensures OOD postprocessors (which don't affect training) reuse the same base checkpoints.
        assert (
            "_OOD_" in self.config.exp_name
        ), "Experiment name must contain '_OOD_' to separate base name."

        base_name = self.config.exp_name.split("_OOD_")[0]

        if task_id is not None:
            ckpt_file_name = f"{base_name}_{task_id}.pth"
        else:
            ckpt_file_name = f"{base_name}.pth"

        return os.path.join(ckpt_dir, ckpt_file_name)

    def _save_context_file(self, task_id, train_task):
        """Write the per-task class-context JSON alongside the checkpoint.

        evaluate.py relies on this file to restore each task's
        ``classes_in_this_experience`` without loading the full Avalanche
        training state.
        """
        ckpt_dir = self.config.scenario.ckpt_dir
        path = os.path.join(ckpt_dir, f"classes_{task_id}.json")
        payload = {
            "task_id": task_id,
            "classes_in_this_experience": list(
                train_task.classes_in_this_experience
            ),
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    def _ensure_context_file(self, task_id, train_task):
        """Backfill classes_{task_id}.json for older checkpoint directories.

        This keeps ``train.py`` idempotent for checkpoint directories created
        before context files were introduced: rerunning training recreates the
        JSON metadata required by ``evaluate.py`` without retraining.
        """
        path = os.path.join(
            self.config.scenario.ckpt_dir, f"classes_{task_id}.json"
        )
        if not os.path.exists(path):
            print(
                f"Backfilling missing context file for task {task_id}: {path}"
            )
            self._save_context_file(task_id, train_task)

    def _load_init_weights(self):
        """Loads specific initialization weights if provided in config."""
        init_path = str(
            getattr(self.config.scenario, "init_weights_path", "") or ""
        ).strip()
        init_mode = getattr(self.config.scenario, "init_mode", None)
        if init_mode is None:
            # Backward-compatible fallback for older configs/CLI overrides.
            init_mode = (
                "random"
                if init_path.lower() in {"", "none", "null", "random"}
                else "checkpoint"
            )
        init_mode = str(init_mode).strip().lower()

        if init_mode == "random":
            print(
                "Initialization mode is 'random'. "
                "Using seeded random initialization."
            )
            return

        if init_mode != "checkpoint":
            raise ValueError(
                "scenario.init_mode must be either 'checkpoint' or 'random', "
                f"got {init_mode!r}."
            )

        if not init_path:
            raise ValueError(
                "scenario.init_mode='checkpoint' requires "
                "scenario.init_weights_path to be set."
            )

        if os.path.exists(init_path):
            print(
                f"Loading initial weights from {init_path} for first task training."
            )
            checkpoint = torch.load(init_path, map_location=self.device)
            safe_load_state_dict(
                self.strategy.model, checkpoint["model_state_dict"]
            )
        else:
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {init_path}. "
                "Either provide a valid checkpoint path or set "
                "scenario.init_mode=random."
            )
