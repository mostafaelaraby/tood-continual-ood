import copy
import itertools

import numpy as np
import torch
import torch.distributed as dist
from avalanche.benchmarks.utils import (
    _make_taskaware_tensor_classification_dataset,
)
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.core import SupervisedPlugin
from avalanche.models import NCMClassifier as NCMClassifierAVL
from avalanche.training.losses import ICaRLLossPlugin as ICaRLLossPluginAvL
from avalanche.training.supervised.icarl import _ICaRLPlugin as _ICaRLPluginAVL
from avalanche.training.templates.common_templates import SupervisedTemplate
from avalanche.training.utils import freeze_everything

from utils.dist_utils import (
    get_world_size,
    is_dist_available_and_initialized,
    is_main_process,
    unwrap_model,
)


class ClipGradients(SupervisedPlugin):
    supports_distributed = True

    def __init__(self, max_norm: float = 1.0):
        super().__init__()
        self.max_norm = max_norm

    def after_backward(self, strategy, **kwargs):
        torch.nn.utils.clip_grad_norm_(
            strategy.model.parameters(), self.max_norm
        )


def _unwrap_model(model):
    """Strip DataParallel or DistributedDataParallel wrapper if present."""
    return unwrap_model(model)


class _ICaRLPlugin(_ICaRLPluginAVL):
    def before_forward(self, strategy: "SupervisedTemplate", **kwargs):
        if self.input_size is None:
            with torch.no_grad():
                raw_model = _unwrap_model(strategy.model)
                self.input_size = strategy.mb_x.shape[1:]
                self.output_size = strategy.model(
                    strategy.mb_x, task_labels=strategy.mb_task_id
                ).shape[1]
                self.embedding_size = raw_model.feature_extractor(
                    strategy.mb_x
                ).shape[1]

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """Build exemplar set then immediately move raw images to CPU.

        Avalanche's construct_exemplar_set stores x_memory on strategy.device
        (GPU). With 100 tasks and 20 000 ImageNet exemplars this fills VRAM.
        Moving to CPU right after construction keeps the GPU free while still
        allowing compute_class_means (which called to(device) per-batch) to
        work correctly.

        In DDP each rank only holds a shard of the dataset, so
        construct_exemplar_set can see zero samples for some classes on
        non-rank-0 processes, causing torch.cat to fail with an empty list.
        Fix: only rank-0 constructs exemplars, then broadcast to all ranks
        before every rank computes class means (needed for NCM inference).
        """
        is_ddp = is_dist_available_and_initialized() and get_world_size() > 1

        if not is_ddp:
            super().after_training_exp(strategy, **kwargs)
            self.x_memory = [x.cpu() for x in self.x_memory]
            return

        # --- DDP path ---
        if is_main_process():
            self.construct_exemplar_set(strategy)
            self.x_memory = [x.cpu() for x in self.x_memory]

        # Broadcast x_memory / y_memory from rank 0 to all other ranks.
        objects = [
            getattr(self, "x_memory", []),
            getattr(self, "y_memory", []),
        ]
        dist.broadcast_object_list(objects, src=0)
        self.x_memory, self.y_memory = objects[0], objects[1]

        # All ranks must compute class means (used by NCM classifier).
        self.compute_class_means(strategy)

    def after_train_dataset_adaptation(
        self, strategy: "SupervisedTemplate", **kwargs
    ):
        if strategy.clock.train_exp_counter == 0:
            return
        if hasattr(self, "x_memory") and len(self.x_memory) > 0:
            memory = _make_taskaware_tensor_classification_dataset(
                torch.cat([x.cpu() for x in self.x_memory]),
                torch.tensor(
                    list(itertools.chain.from_iterable(self.y_memory))
                ),
                transform=self.buffer_transform,
                target_transform=None,
            )

            strategy.adapted_dataset = concat_datasets(
                (strategy.adapted_dataset, memory)
            )
        else:
            print("Warning: x_memory was empty, skipping concatenation.")


class ICaRLLossPlugin(ICaRLLossPluginAvL):
    def before_forward(self, strategy, **kwargs):
        if self.old_model is not None:
            self.old_model.eval()
            with torch.no_grad():
                raw_old_model = _unwrap_model(self.old_model)
                features = raw_old_model.feature_extractor(strategy.mb_x)
                # Unconditionally get all logits, do not restrict to just the last task_id
                self.old_logits = raw_old_model.train_classifier(features)

    def after_training_exp(self, strategy, **kwargs):
        # Deepcopy the *unwrapped* module, not the DDP wrapper.
        # deepcopy(DDP_module) copies a broken process-group reference and
        # keeps unnecessary DDP state. We only need the raw nn.Module for
        # distillation inference, so unwrap first then copy.
        old_model = copy.deepcopy(unwrap_model(strategy.model))
        self.old_model = old_model.to(strategy.device)
        self.old_model.eval()
        freeze_everything(self.old_model)
        self.old_classes += np.unique(
            strategy.experience.dataset.targets
        ).tolist()


class NCMClassifier(NCMClassifierAVL):
    def __init__(self, normalize=True):
        super().__init__(normalize)
        self.task_id = None
        self.n_classes_per_task = None
        self.return_task_id = False

    def replace_class_means_dict(self, class_means_dict):
        """
        Replace existing dictionary of means with a given dictionary.
        """
        assert isinstance(class_means_dict, dict), (
            "class_means_dict must be a dictionary mapping class_id "
            "to mean vector"
        )
        if (
            self.task_id is not None
            and self.task_id > 0
            and self.return_task_id
        ):
            n_classes = (
                self.n_classes_per_task * self.task_id
                if self.task_id is not None
                else 0
            )
            tmp_dict = copy.deepcopy(class_means_dict)
            for k, v in tmp_dict.items():
                class_means_dict[int(k + n_classes)] = v
                class_means_dict[k] = self.class_means_dict.get(
                    k, torch.zeros_like(v)
                )
        super().replace_class_means_dict(class_means_dict)
