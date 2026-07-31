"""FOSTER: Feature Boosting and Compression for Class-Incremental Learning.

Wang, Fu-Yun, et al. "FOSTER: Feature Boosting and Compression for
Class-Incremental Learning." ECCV 2022.
https://arxiv.org/abs/2204.04662

Reference PyCIL implementation:
https://github.com/LAMDA-CL/PyCIL/blob/master/models/foster.py

This Avalanche plugin implements the three-phase FOSTER pipeline:

1. **Initial training** (task 0): Normal Avalanche training with CE loss.
2. **Feature boosting** (tasks ≥ 1): The per-batch CE loss is replaced with a
   class-balanced weighted CE (effective-number weighting, beta1) plus a
   knowledge-distillation (KD) loss from the frozen old model. The exemplar
   buffer is merged into the adapted_dataset so old and new data train together.
3. **Feature compression** (tasks ≥ 1, post-training): A student network
   initialised from the previous task's frozen model (with its output head
   expanded to cover all current classes) is trained with Balanced KD (BKD)
   from the teacher (current strategy model). After convergence the student's
   weights replace the strategy model. Optional weight alignment is applied to
   both teacher and student.
"""

import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.utils import freeze_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dist_utils import unwrap_model
from utils.helpers import preprocess_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collate_xy(batch):
    """Collate that keeps only (x, y) with correct tensor types."""
    xs = [
        item[0] if isinstance(item[0], torch.Tensor) else torch.tensor(item[0])
        for item in batch
    ]
    ys = [
        item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1])
        for item in batch
    ]
    return torch.stack(xs), torch.stack(ys)


def _compute_class_weights(targets, num_classes: int, beta: float) -> torch.Tensor:
    """Per-class weights via effective number of samples.

    w_c = (1 - beta) / (1 - beta^{n_c})

    Weights are normalised so their mean equals 1 (i.e. sum = num_classes).
    """
    counts = torch.zeros(num_classes)
    for t in targets:
        idx = int(t)
        if 0 <= idx < num_classes:
            counts[idx] += 1.0

    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        n = counts[c].item()
        if n > 0:
            weights[c] = (1.0 - beta) / (1.0 - beta**n)

    total = weights.sum()
    if total > 0:
        weights = weights / total * num_classes  # mean ≈ 1
    return weights


def _get_targets(dataset) -> list:
    """Collect integer class labels from an Avalanche or plain torch Dataset."""
    if hasattr(dataset, "targets"):
        t = dataset.targets
        if hasattr(t, "tolist"):
            return t.tolist()
        return [int(v) for v in t]
    # Fallback: iterate (slow, only on small datasets)
    targets = []
    for item in dataset:
        y = item[1]
        targets.append(y.item() if isinstance(y, torch.Tensor) else int(y))
    return targets


def _forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    """Run model forward and return a single flat logit tensor."""
    out = model(x)
    if isinstance(out, dict):
        return torch.cat([out[k] for k in sorted(out.keys())], dim=1)
    return out


# ---------------------------------------------------------------------------
# Main Plugin
# ---------------------------------------------------------------------------


class FOSTERPlugin(SupervisedPlugin):
    """Feature Boosting and Compression for Class-Incremental Learning.

    Parameters
    ----------
    mem_size : int
        Total exemplars kept across all seen classes (class-balanced).
    beta1 : float
        Effective-number β for class-weight computation during feature boosting
        (CE loss weighting).
    beta2 : float
        Effective-number β for BKD class weights during feature compression.
    lambda_okd : float
        Weight on the KD loss during feature boosting (λ in the paper).
    T : float
        Temperature for knowledge distillation (default: 2.0).
    compression_epochs : int
        Training epochs for the feature-compression phase.
    compression_lr : float
        Initial learning rate for the compression SGD optimiser.
    weight_decay : float
        L2 weight decay for the compression optimiser.
    compress_batch_size : int
        Batch size used during the compression phase.
    use_weight_align : bool
        Whether to apply weight alignment (WA) to teacher and student after
        each task.
    device : torch.device | None
        Device override; defaults to ``strategy.device``.
    """

    def __init__(
        self,
        mem_size: int = 2000,
        beta1: float = 0.5,
        beta2: float = 0.5,
        lambda_okd: float = 1.0,
        T: float = 2.0,
        compression_epochs: int = 170,
        compression_lr: float = 0.1,
        weight_decay: float = 2e-4,
        compress_batch_size: int = 128,
        use_weight_align: bool = False,
        device=None,
    ):
        super().__init__()
        self.mem_size = mem_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_okd = lambda_okd
        self.T = T
        self.compression_epochs = compression_epochs
        self.compression_lr = compression_lr
        self.weight_decay = weight_decay
        self.compress_batch_size = compress_batch_size
        self.use_weight_align = use_weight_align
        self.device = device

        self.storage_policy = ClassBalancedBuffer(
            max_size=mem_size, adaptive_size=True
        )

        self._old_model = None          # frozen teacher from previous task
        self._class_weights = None      # weighted-CE tensor for boosting phase
        self._prev_classes_by_task: dict = defaultdict(set)  # for WA tracking

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def after_train_dataset_adaptation(self, strategy, **kwargs):
        """Merge exemplar buffer into the current-task training dataset.

        Mirrors WeightAlignmentPlugin: old exemplars and new-task data share
        a single dataloader so the KD and weighted-CE losses cover both.
        """
        if strategy.clock.train_exp_counter == 0:
            return  # No memory on the first task.
        buf = self.storage_policy.buffer
        if len(buf) == 0:
            return
        strategy.adapted_dataset = concat_datasets(
            (strategy.adapted_dataset, buf)
        )

    # ------------------------------------------------------------------
    # Before-task setup
    # ------------------------------------------------------------------

    def before_training_exp(self, strategy, **kwargs):
        """Freeze old model as teacher; compute class-balance weights."""
        task_id = strategy.experience.current_experience
        if task_id == 0:
            return

        device = self.device or strategy.device

        # --- Save frozen teacher (old model before this task's training) ---
        self._old_model = copy.deepcopy(strategy.model)
        self._old_model.eval()
        freeze_everything(self._old_model)
        self._old_model.to(device)
        print(f"[FOSTER] Saved teacher model for task {task_id}.")

        # --- Compute per-class weights for weighted CE ---
        # adapted_dataset already includes the buffer (merged in
        # after_train_dataset_adaptation), so all_targets covers every class.
        all_targets = _get_targets(strategy.adapted_dataset)
        new_cls = set(strategy.experience.classes_in_this_experience)
        all_cls = set(all_targets)
        num_classes = len(all_cls)
        print(
            f"[FOSTER] n_old={len(all_cls - new_cls)}, "
            f"n_new={len(new_cls)}, n_total={num_classes}."
        )
        self._class_weights = _compute_class_weights(
            all_targets, num_classes, self.beta1
        ).to(device)

    # ------------------------------------------------------------------
    # Feature boosting: augment loss during the main Avalanche training loop
    # ------------------------------------------------------------------

    def before_backward(self, strategy, **kwargs):
        """Replace/augment the mini-batch loss with weighted CE + KD.

        Active only for tasks ≥ 1 when an old (teacher) model is available.
        Loss = Loss_clf (class-balanced CE) + λ_okd * Loss_kd (logit KD on old classes).
        """
        if strategy.experience.current_experience == 0 or self._old_model is None:
            return

        device = self.device or strategy.device
        x = strategy.mb_x
        y = strategy.mb_y
        mb_out = strategy.mb_output

        cur_logits = (
            torch.cat([mb_out[k] for k in sorted(mb_out.keys())], dim=1)
            if isinstance(mb_out, dict)
            else mb_out
        )

        # --- Class-balanced CE ---
        if (
            self._class_weights is not None
            and self._class_weights.shape[0] == cur_logits.shape[1]
        ):
            loss_clf = F.cross_entropy(cur_logits, y, weight=self._class_weights)
        else:
            loss_clf = F.cross_entropy(cur_logits, y)

        # --- KD loss on old-class logit columns ---
        loss_kd = torch.tensor(0.0, device=device)
        try:
            with torch.no_grad():
                old_logits = _forward_logits(self._old_model, x)
            n_old = old_logits.shape[1]
            # KL divergence over the old class distribution
            loss_kd = (
                F.kl_div(
                    F.log_softmax(cur_logits[:, :n_old] / self.T, dim=1),
                    F.softmax(old_logits / self.T, dim=1),
                    reduction="batchmean",
                )
                * (self.T**2)
            )
        except Exception as exc:
            print(f"[FOSTER] KD loss skipped in before_backward: {exc}")

        strategy.loss = loss_clf + self.lambda_okd * loss_kd

    # ------------------------------------------------------------------
    # Post-task: weight alignment + feature compression + memory update
    # ------------------------------------------------------------------

    def after_training_exp(self, strategy, **kwargs):
        """Run WA → compression → WA → update memory."""
        task_id = strategy.experience.current_experience
        device = self.device or strategy.device

        if task_id == 0:
            # Just seed the memory; no compression on the first task.
            self.storage_policy.update(strategy, **kwargs)
            return

        # 1. Weight-align the teacher (current strategy model)
        if self.use_weight_align:
            self._weight_align(strategy.model, strategy.experience)
            print("[FOSTER] Applied WA to teacher.")

        # 2. Feature compression via BKD
        self._feature_compression(strategy, device)

        # 3. Weight-align the student (now loaded into strategy.model)
        if self.use_weight_align:
            self._weight_align(strategy.model, strategy.experience)
            print("[FOSTER] Applied WA to student.")

        # 4. Update exemplar memory
        self.storage_policy.update(strategy, **kwargs)

        # 5. Record classes seen so far (used by WA in future tasks)
        exp = strategy.experience
        try:
            task_labels = list(set(exp.dataset.targets_task_labels))
        except Exception:
            task_labels = [exp.task_label]
        for tid in task_labels:
            try:
                td = exp.dataset.task_set[tid]
                self._prev_classes_by_task[tid] = self._prev_classes_by_task[
                    tid
                ].union(set(td.targets.uniques))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Feature compression (Balanced KD)
    # ------------------------------------------------------------------

    def _feature_compression(self, strategy, device):
        """Train a student network with Balanced KD from the teacher model.

        The student is initialised from ``self._old_model`` (the frozen
        checkpoint saved before this task's training began) with its output
        head expanded to the current total number of classes. BKD trains the
        student to match the teacher while correcting for class imbalance via
        per-sample effective-number weights. When done, student weights are
        loaded back into ``strategy.model``.
        """
        if self._old_model is None:
            print("[FOSTER] No old model; skipping feature compression.")
            return

        teacher = unwrap_model(strategy.model)
        teacher.eval()

        # --- Build student: old model backbone + expanded FC ---
        student = copy.deepcopy(self._old_model)
        for p in student.parameters():
            p.requires_grad_(True)

        if hasattr(teacher, "get_fc_layer") and hasattr(student, "get_fc_layer"):
            t_fc = teacher.get_fc_layer()
            s_fc = student.get_fc_layer()
            if t_fc is not None and s_fc is not None:
                n_old_out = s_fc.out_features
                n_total_out = t_fc.out_features
                if n_total_out > n_old_out:
                    has_bias = s_fc.bias is not None
                    new_fc = nn.Linear(
                        s_fc.in_features, n_total_out, bias=has_bias
                    ).to(device)
                    with torch.no_grad():
                        # Copy old-class weights from student (old model)
                        new_fc.weight[:n_old_out].copy_(s_fc.weight)
                        if has_bias:
                            new_fc.bias[:n_old_out].copy_(s_fc.bias)
                        # Initialise new-class neurons from teacher
                        new_fc.weight[n_old_out:].copy_(t_fc.weight[n_old_out:])
                        if has_bias and t_fc.bias is not None:
                            new_fc.bias[n_old_out:].copy_(t_fc.bias[n_old_out:])
                    self._set_fc(student, new_fc)

        student.to(device)
        student.train()

        # --- Data: current task data merged with buffer ---
        dataset = strategy.adapted_dataset
        all_targets = _get_targets(dataset)
        t_fc = teacher.get_fc_layer() if hasattr(teacher, "get_fc_layer") else None
        num_classes = (
            t_fc.out_features
            if t_fc is not None
            else len(set(all_targets))
        )
        bkd_weights = _compute_class_weights(
            all_targets, num_classes, self.beta2
        ).to(device)

        loader = DataLoader(
            dataset,
            batch_size=self.compress_batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
            collate_fn=_collate_xy,
        )

        # --- Optimiser with multi-step LR decay (mirrors PyCIL defaults) ---
        optimizer = optim.SGD(
            student.parameters(),
            lr=self.compression_lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        milestones = [
            int(self.compression_epochs * 0.55),
            int(self.compression_epochs * 0.70),
            int(self.compression_epochs * 0.85),
        ]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )

        log_every = max(1, self.compression_epochs // 10)
        print(
            f"[FOSTER] Feature compression: {self.compression_epochs} epochs, "
            f"dataset size={len(dataset)}, num_classes={num_classes}."
        )

        for epoch in tqdm(
            range(self.compression_epochs), desc="[FOSTER] Compression"
        ):
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                x, y = preprocess_batch(batch, device)

                # Teacher soft targets (frozen)
                with torch.no_grad():
                    t_logits = _forward_logits(teacher, x)

                # Student predictions
                s_logits = _forward_logits(student, x)

                # --- Balanced KD: per-sample weight from class frequency ---
                sample_w = bkd_weights[y]  # [B]
                kd_per_sample = F.kl_div(
                    F.log_softmax(s_logits / self.T, dim=1),
                    F.softmax(t_logits / self.T, dim=1),
                    reduction="none",
                ).sum(dim=1)  # [B]
                loss_bkd = (sample_w * kd_per_sample).mean() * (self.T**2)

                # --- Class-balanced CE ---
                loss_clf = F.cross_entropy(s_logits, y, weight=bkd_weights)

                loss = loss_clf + loss_bkd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            if n_batches > 0 and (epoch + 1) % log_every == 0:
                tqdm.write(
                    f"  [FOSTER Compression] Epoch {epoch + 1}: "
                    f"avg_loss={epoch_loss / n_batches:.4f}"
                )

        # --- Load student weights into strategy model ---
        try:
            strategy.model.load_state_dict(student.state_dict())
        except RuntimeError as exc:
            print(
                f"[FOSTER] load_state_dict mismatch ({exc}). "
                "Copying matched parameters manually."
            )
            s_sd = student.state_dict()
            m_sd = strategy.model.state_dict()
            for k in m_sd:
                if k in s_sd and m_sd[k].shape == s_sd[k].shape:
                    m_sd[k].copy_(s_sd[k])
            strategy.model.load_state_dict(m_sd)

        print("[FOSTER] Compression complete. Model updated with student.")

    # ------------------------------------------------------------------
    # Weight Alignment
    # ------------------------------------------------------------------

    def _weight_align(self, model, experience):
        """Rescale new-class FC weights so their mean L2 norm equals that of
        old-class weights (Zhao et al., CVPR 2020)."""
        model = unwrap_model(model)
        if not hasattr(model, "get_fc_layer"):
            return
        fc = model.get_fc_layer()
        if fc is None:
            return

        new_classes = list(experience.classes_in_this_experience)
        all_prev = {c for cs in self._prev_classes_by_task.values() for c in cs}
        old_classes = list(all_prev)

        if not old_classes or not new_classes:
            return

        W = fc.weight.data
        norm_old = W[old_classes].norm(dim=1).mean()
        norm_new = W[new_classes].norm(dim=1).mean()

        if norm_new.item() > 0:
            gamma = norm_old / norm_new
            print(
                f"[FOSTER WA] gamma={gamma:.4f} | "
                f"old={len(old_classes)} cls | new={len(new_classes)} cls"
            )
            fc.weight.data[new_classes] = W[new_classes] * gamma

    # ------------------------------------------------------------------
    # Utility: replace the output FC layer inside a WrapModel
    # ------------------------------------------------------------------

    def _set_fc(self, model, new_fc: nn.Linear):
        """Replace the final classifier in a WrapModel (or bare model)."""
        # WrapModel keeps both model.classifier and model.model.<fc_attr>
        if hasattr(model, "classifier"):
            model.classifier = new_fc
        if hasattr(model, "model"):
            inner = model.model
            for attr in ("fc", "classifier", "output", "heads"):
                if hasattr(inner, attr):
                    setattr(inner, attr, new_fc)
                    break
