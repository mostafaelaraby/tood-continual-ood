import copy
from collections import defaultdict

import torch
from utils.dist_utils import unwrap_model
import torch.nn.functional as F
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.core import SupervisedPlugin
from avalanche.models import MultiTaskModule, avalanche_forward
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.utils import freeze_everything


class WeightAlignmentPlugin(SupervisedPlugin):
    """
    Self-contained Weight Alignment Plugin for Avalanche.

    Implements 'Maintaining Discrimination and Fairness in Class Incremental
    Learning' (CVPR 2020), following the OpenCIL reference implementation:
    https://github.com/mala-lab/OpenCIL/blob/main/opencil/trainers/incremental_wa_pycil.py

    Key design vs the previous version:
    - Memory is managed internally via ClassBalancedBuffer (no separate Replay
      strategy term needed — use Naive as the base strategy).
    - In after_train_dataset_adaptation the exemplar buffer is concatenated
      directly into the current-task dataset so memory and new data share a
      single dataloader.
    - CE loss is computed over ALL samples in the mini-batch (old exemplars +
      new data) against ALL seen class logits, matching the reference.
    - KD loss is computed on old-class logits using the frozen old model.
    - Combined loss: (1 - λ) * CE_all + λ * KD,  λ = n_old / n_total.
    - Weight Alignment rescales new-class weights after training.
    """

    def __init__(self, mem_size: int, temp: float = 2.0):
        """
        :param mem_size: Total number of exemplars to keep across all seen classes.
        :param temp: Temperature for Knowledge Distillation (default: 2.0).
        """
        super().__init__()
        self.temp = temp
        self.storage_policy = ClassBalancedBuffer(
            max_size=mem_size, adaptive_size=True
        )
        self.old_model = None
        self.prev_classes_by_task = defaultdict(set)

    # ------------------------------------------------------------------ #
    # Dataset preparation                                                   #
    # ------------------------------------------------------------------ #

    def after_train_dataset_adaptation(self, strategy, **kwargs):
        """Concatenate the exemplar buffer into the current-task dataset.

        This mirrors OpenCIL's unified dataloader that combines new-task data
        and rehearsal memory in a single pass — no extra replay forward/backward
        is added.
        """
        if strategy.clock.train_exp_counter == 0:
            return  # First task: no memory yet.
        buffer = self.storage_policy.buffer
        if len(buffer) == 0:
            return
        strategy.adapted_dataset = concat_datasets(
            (strategy.adapted_dataset, buffer)
        )

    # ------------------------------------------------------------------ #
    # Loss computation                                                      #
    # ------------------------------------------------------------------ #

    def before_backward(self, strategy, **kwargs):
        """Compute the combined CE + KD loss.

        Follows the OpenCIL WA formulation:
            loss = (1 - λ) * CE(all_samples) + λ * KD(old_logits)
            λ = n_old_classes / n_total_classes

        CE is computed on ALL samples in the mini-batch (memory + new data)
        over ALL seen class logits.  KD distills the frozen old model's
        predictions on old-class output units.
        """
        if strategy.experience.current_experience == 0 or self.old_model is None:
            return  # Standard CE for the first task (default strategy.loss).

        mb_output = strategy.mb_output
        n_new = len(strategy.experience.classes_in_this_experience)
        if isinstance(mb_output, dict):
            n_total = sum(v.shape[1] for v in mb_output.values())
        else:
            n_total = mb_output.shape[1]
        n_known = n_total - n_new
        kd_lambda = n_known / n_total

        # -- CE over ALL samples (memory exemplars + new-task data) --------
        loss_ce = F.cross_entropy(mb_output, strategy.mb_y)

        # -- KD on old-class logits ----------------------------------------
        if isinstance(self.old_model, MultiTaskModule):
            with torch.no_grad():
                y_prev = avalanche_forward(self.old_model, strategy.mb_x, None)
            y_curr = avalanche_forward(strategy.model, strategy.mb_x, None)
        else:
            with torch.no_grad():
                y_prev = {0: self.old_model(strategy.mb_x)}
            y_curr = {0: mb_output}

        loss_kd = 0.0
        for task_id, yp in y_prev.items():
            if task_id in self.prev_classes_by_task:
                yc = y_curr[task_id]
                au = list(self.prev_classes_by_task[task_id])
                loss_kd += self._kd_loss(yc[:, au], yp[:, au], self.temp)

        strategy.loss = (1.0 - kd_lambda) * loss_ce + kd_lambda * loss_kd

    # ------------------------------------------------------------------ #
    # Post-task hooks                                                        #
    # ------------------------------------------------------------------ #

    def after_training_exp(self, strategy, **kwargs):
        """
        1. Apply Weight Alignment to new-class weights (experience > 0).
        2. Save a frozen copy of the model for the next round of KD.
        3. Update the exemplar memory with samples from the current experience.
        """
        if strategy.experience.current_experience > 0:
            self._weight_align(strategy)
        self._save_old_model(strategy, strategy.experience)
        self.storage_policy.update(strategy, **kwargs)

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    def _save_old_model(self, agent, exp):
        """Freeze and store a copy of the current model for future KD."""
        print(
            f"WeightAlignmentPlugin: Saving model for KD at exp "
            f"{exp.current_experience}."
        )
        self.old_model = copy.deepcopy(agent.model)
        self.old_model.eval()
        freeze_everything(self.old_model)
        task_ids = [int(x) for x in exp.dataset.targets_task_labels.uniques]
        for task_id in task_ids:
            task_data = exp.dataset.task_set[task_id]
            pc = set(task_data.targets.uniques)
            self.prev_classes_by_task[task_id] = (
                self.prev_classes_by_task[task_id].union(pc)
            )

    def _kd_loss(self, pred, soft, T):
        """Standard Knowledge Distillation loss."""
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1.0 * torch.mul(soft, pred).sum() / pred.shape[0]

    def _weight_align(self, strategy):
        """Rescale new-class classifier weights so their mean L2 norm matches
        the mean norm of the old-class weights."""
        print("WeightAlignmentPlugin: Applying Weight Alignment.")
        classifier = unwrap_model(strategy.model).get_fc_layer()
        if classifier is None:
            print(
                "WeightAlignmentPlugin Warning: Linear classifier not found. "
                "Skipping alignment."
            )
            return

        new_classes = set(strategy.experience.classes_in_this_experience)
        all_prev_classes = {
            c for classes in self.prev_classes_by_task.values() for c in classes
        }
        old_classes = list(all_prev_classes)
        new_classes_list = list(new_classes)

        if not old_classes:
            print("WA: No old classes found. Skipping alignment.")
            return

        if isinstance(classifier, list):
            weights = torch.cat([cl.weight for cl in classifier], dim=0)
        else:
            weights = classifier.weight.data

        W_old = weights[old_classes]
        W_new = weights[new_classes_list]

        mean_norm_old = torch.norm(W_old, p=2, dim=1).mean()
        mean_norm_new = torch.norm(W_new, p=2, dim=1).mean()

        if mean_norm_new.item() != 0:
            gamma = mean_norm_old / mean_norm_new
            print(
                f"WA: Task {strategy.experience.current_experience} | "
                f"Gamma: {gamma:.4f} | "
                f"Old: {len(old_classes)} classes | New: {len(new_classes_list)} classes"
            )
            if isinstance(classifier, list):
                classifier[-1].weight.data = W_new * gamma
            else:
                weights[new_classes_list] = W_new * gamma
