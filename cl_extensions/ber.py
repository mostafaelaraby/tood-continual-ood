import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from avalanche.core import SupervisedPlugin
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dist_utils import unwrap_model
from utils.helpers import preprocess_batch

"""Miao W, Pang G, Nguyen TT, Fang R, Zheng J, Bai X. Opencil:
Benchmarking out-of-distribution detection in class incremental learning.
Pattern Recognition. 2025 Jul 20:112163.

Implements both NTER (New Task Energy Regularization) and OTER (Old Task
Energy Regularization).  NTER synthesises pseudo-OOD samples via intra-batch
mixup of new-task features and regularises their energy away from ID.  OTER
mixes a small fraction of new-task features into old-class exemplars and
pushes their energy towards the ID margin, countering the confidence drop
caused by catastrophic forgetting.
"""


def _ber_collate(batch):
    """Collate that keeps only (x, y), ensuring y is a tensor."""
    # item[0] is usually the image (Tensor), item[1] is the label (int or Tensor)
    xs = [
        item[0] if isinstance(item[0], torch.Tensor) else torch.tensor(item[0])
        for item in batch
    ]
    ys = [
        (
            torch.tensor(item[1])
            if not isinstance(item[1], torch.Tensor)
            else item[1]
        )
        for item in batch
    ]

    return torch.stack(xs), torch.stack(ys)


class BeRPlugin(SupervisedPlugin):
    """
    Bi-directional Energy Regularization (BeR) Plugin.

    Default hyper-parameters follow the OpenCIL paper (Sec 4.1):
        m_in=-27, m_out=-5, alpha=0.1, beta=1.0, val_beta=0.002,
        lr=0.1, weight_decay=5e-4, num_epochs=10.
    """

    def __init__(
        self,
        m_in=-27.0,
        m_out=-5.0,
        beta=1.0,
        val_beta=0.002,
        alpha=0.1,
        lr=0.1,
        weight_decay=5e-4,
        num_epochs=10,
        batch_size=None,
        reset_after_eval=False,
        device=None,
    ):
        super().__init__()
        self.m_in = m_in
        self.m_out = m_out
        self.beta = beta
        self.val_beta = val_beta  # lambda in paper (OTER mixup ratio)
        self.alpha = alpha  # weight for energy regularisation terms
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.reset_after_eval = reset_after_eval
        self.device = device

        self.original_head_state = None
        self._ber_head_state = None
        self._last_task_label = (
            None  # stored in after_training_exp for before_eval
        )

    # ------------------------------------------------------------------
    # Memory buffer helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_memory_buffer(strategy):
        """Return the replay-memory dataset (old-class exemplars).

        Searches strategy-level and plugin-level storage policies, then
        falls back to iCaRL-style x_memory / y_memory.
        """
        # 1. Strategy-level storage_policy (DER, custom strategies)
        if hasattr(strategy, "storage_policy"):
            sp = strategy.storage_policy
            if hasattr(sp, "buffer") and len(sp.buffer) > 0:
                return sp.buffer

        # 2. Plugin-level storage_policy (BiC, etc.)
        for plugin in strategy.plugins:
            if hasattr(plugin, "storage_policy"):
                sp = plugin.storage_policy
                if hasattr(sp, "buffer") and len(sp.buffer) > 0:
                    return sp.buffer

        # 3. iCaRL-style x_memory / y_memory
        for plugin in strategy.plugins:
            if hasattr(plugin, "x_memory") and len(plugin.x_memory) > 0:
                from torch.utils.data import TensorDataset

                x = torch.cat([xi.cpu() for xi in plugin.x_memory])
                y = torch.tensor(
                    list(itertools.chain.from_iterable(plugin.y_memory))
                )
                return TensorDataset(x, y)

        return None

    # ------------------------------------------------------------------
    # Avalanche callbacks
    # ------------------------------------------------------------------
    def after_training_exp(self, strategy, **kwargs):
        task_id = strategy.experience.current_experience
        print(
            f"\n[BeR Plugin] Starting Post-Task Finetuning for Task {task_id}..."
        )

        # --- 1. Dynamic Config ---
        device = self.device if self.device else strategy.device
        batch_size = (
            self.batch_size if self.batch_size else strategy.train_mb_size
        )

        # --- 2. Data Loading ---
        # Use adapted_dataset: it already contains new-task data + memory with
        # proper transforms (iCaRL applies buffer_transform, BiC adds buffer via
        # storage_policy during after_train_dataset_adaptation).  This ensures
        # the CE loss sees ALL seen classes and both new/old data have consistent
        # augmentation, matching the reference OpenCIL implementation.
        memory_buffer = self._get_memory_buffer(strategy)
        val_loader = None

        train_loader = DataLoader(
            strategy.adapted_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
            collate_fn=_ber_collate,
        )

        if memory_buffer is not None and task_id > 0:
            # Separate old-exemplar loader for OTER
            val_loader = DataLoader(
                memory_buffer,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=False,
                collate_fn=_ber_collate,
            )
            print(
                f"[BeR Plugin] Train dataset size: {len(strategy.adapted_dataset)}, "
                f"OTER memory size: {len(memory_buffer)}."
            )
        else:
            print(
                f"[BeR Plugin] Train dataset size: {len(strategy.adapted_dataset)}, "
                f"NTER only (no memory or task 0)."
            )

        # --- 3. Model Setup ---
        model = unwrap_model(strategy.model)
        model.eval()  # Freeze BN stats

        if hasattr(model, "task_id"):
            model.task_id = strategy.experience.task_label

        if not hasattr(model, "get_fc_layer"):
            print(
                "[BeR Plugin] Warning: Model missing `get_fc_layer`. Skipping."
            )
            return

        target_fc = model.get_fc_layer()

        # Clone Head
        aux_head = copy.deepcopy(target_fc).to(device)
        aux_head.train()

        # --- 4. Optimizer ---
        optimizer = optim.SGD(
            aux_head.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )

        # --- 5. Training Loop ---
        self._train_loop(
            model,
            aux_head,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
        )

        # --- 6. Store BER head (defer application to before_eval) ---
        # We do NOT modify the model here.  Applying the BER head in
        # after_training_exp would corrupt the old_model that ICaRLLossPlugin
        # deep-copies for knowledge distillation.  Instead we store the trained
        # state and apply it just before evaluation begins (before_eval).
        self._ber_head_state = copy.deepcopy(aux_head.state_dict())
        self._last_task_label = strategy.experience.task_label
        print(
            "[BeR Plugin] Finetuning complete. Stored BER head for evaluation."
        )

    def before_eval(self, strategy, **kwargs):
        """Apply the BER-finetuned head just before evaluation begins.

        This ensures:
        - ICaRLLossPlugin.after_training_exp deep-copies the ORIGINAL head.
        - Checkpoints save the model with the ORIGINAL head.
        - The BER head is only active during evaluation (CL + OOD).

        NOTE: strategy.experience is None at this point because Avalanche fires
        before_eval BEFORE iterating over eval experiences (base.py:209-210).
        We use self._last_task_label (saved in after_training_exp) instead.
        """
        if self._ber_head_state is not None:
            model = unwrap_model(strategy.model)
            if hasattr(model, "task_id") and self._last_task_label is not None:
                model.task_id = self._last_task_label

            if hasattr(model, "get_fc_layer"):
                target_fc = model.get_fc_layer()
                if self.reset_after_eval:
                    self.original_head_state = copy.deepcopy(
                        target_fc.state_dict()
                    )
                target_fc.load_state_dict(self._ber_head_state)
                self._ber_head_state = None
                print("[BeR Plugin] Applied BER head for evaluation.")

    def reset_ber_head(self, strategy):
        if self.original_head_state is None:
            return
        model = unwrap_model(strategy.model)
        if hasattr(model, "task_id") and strategy.experience is not None:
            model.task_id = strategy.experience.task_label

        if hasattr(model, "get_fc_layer"):
            target_fc = model.get_fc_layer()
            current_state = target_fc.state_dict()
            old_state = self.original_head_state
            old_out_features = old_state["weight"].shape[0]
            # Partial restore: copy old weights into the (possibly expanded)
            # head, keeping new-class neurons from classifier expansion.
            current_state["weight"][:old_out_features] = old_state["weight"]
            if "bias" in old_state and "bias" in current_state:
                current_state["bias"][:old_out_features] = old_state["bias"]
            target_fc.load_state_dict(current_state)
            self.original_head_state = None
            print("[BeR Plugin] Reverted classifier to pre-finetuned state.")

    def before_training_exp(self, strategy, **kwargs):
        """Reset the classifier head if it was modified by BER in the previous task.

        This runs at the START of each new task's training, ensuring the
        BER-finetuned head persists through ALL evaluation phases (CL metrics +
        OOD detection) after the previous task, but is reverted before new task
        training begins.
        """
        self.reset_ber_head(strategy)

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        model,
        aux_head,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
    ):
        if val_loader is not None:
            val_iterator = iter(val_loader)

        for epoch in tqdm(
            range(self.num_epochs), desc="[BeR Plugin] Training Epochs"
        ):
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                x, y = preprocess_batch(batch, device)

                # ---- A. NTER: Image-level mixup for pseudo-OOD ----
                lam = np.random.beta(self.beta, self.beta)
                perm_idx = torch.randperm(x.size(0), device=device)
                dummy_x = lam * x + (1 - lam) * x[perm_idx]

                # ---- B. Forward through Frozen Backbone ----
                with torch.no_grad():
                    feats = model.get_features(x)
                    dummy_feats = model.get_features(dummy_x)

                # ---- C. Forward through aux head ----
                clean_logits = aux_head(feats)
                dummy_logits = aux_head(dummy_feats)

                # ---- D. Classification loss (new task) ----
                loss_cls = F.cross_entropy(clean_logits, y)

                # ---- E. NTER energy losses ----
                Ec_train_in = -torch.logsumexp(clean_logits, dim=1)
                Ec_train_out = -torch.logsumexp(dummy_logits, dim=1)

                loss_e_train_in = torch.pow(
                    F.relu(Ec_train_in - self.m_in), 2
                ).mean()
                loss_e_train_out = torch.pow(
                    F.relu(self.m_out - Ec_train_out), 2
                ).mean()

                # ---- F. OTER: old-task energy regularisation ----
                loss_cls_val = torch.tensor(0.0, device=device)
                loss_e_val_in = torch.tensor(0.0, device=device)

                if val_loader is not None:
                    try:
                        val_batch = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(val_loader)
                        val_batch = next(val_iterator)

                    val_x, val_y = preprocess_batch(val_batch, device)

                    # Mix at IMAGE level: tiny new-task + mostly old-class
                    mixed_size = min(val_x.size(0), x.size(0))
                    mixed_val_x = (
                        self.val_beta * x[:mixed_size]
                        + (1 - self.val_beta) * val_x[:mixed_size]
                    )

                    with torch.no_grad():
                        mixed_val_feats = model.get_features(mixed_val_x)

                    mixed_val_logits = aux_head(mixed_val_feats)
                    val_targets = val_y[:mixed_size]

                    # CE on mixed old-class data
                    loss_cls_val = F.cross_entropy(
                        mixed_val_logits, val_targets
                    )

                    # Energy in-distribution for mixed old-class data
                    Ec_val_in = -torch.logsumexp(mixed_val_logits, dim=1)
                    loss_e_val_in = torch.pow(
                        F.relu(Ec_val_in - self.m_in), 2
                    ).mean()

                # ---- G. Total loss (Matches OpenCIL 0.1 weighting) ----
                loss = (
                    loss_cls
                    + loss_cls_val
                    + self.alpha
                    * (loss_e_train_in + loss_e_train_out + loss_e_val_in)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            if n_batches > 0:
                avg = epoch_loss / n_batches
                tqdm.write(f"  [BeR] Epoch {epoch}: avg_loss={avg:.4f}")
