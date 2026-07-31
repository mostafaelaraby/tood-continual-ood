import gc
import io
import os
import platform
import random
import re
import sys
import traceback
from collections import defaultdict
from contextlib import ContextDecorator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import (
    AlternativeValues,
    MetricValue,
    TensorImage,
)
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.logging import WandBLogger
from avalanche.models import MobilenetV1
from matplotlib.figure import Figure
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate

from openood.networks.resnet50 import ResNet50
from openood.networks.vit_b_16 import ViT_B_16
from utils.dist_utils import unwrap_model

# Platform-specific file locking
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    import msvcrt
else:
    import fcntl


class CustomWandbLogger(WandBLogger):

    def log_single_metric(self, name, value, x_plot):
        self.step = x_plot

        if name.startswith("WeightCheckpoint"):
            if self.log_artifacts:
                self._log_checkpoint(name, value, x_plot)
            return

        if isinstance(value, AlternativeValues):
            value = value.best_supported_value(
                Image,
                Tensor,
                TensorImage,
                Figure,
                float,
                int,
                self.wandb.plot.custom_chart.CustomChart,
            )

        if not isinstance(
            value,
            (
                Image,
                TensorImage,
                Tensor,
                Figure,
                float,
                int,
                self.wandb.plot.custom_chart.CustomChart,
            ),
        ):
            # Unsupported type
            return

        if isinstance(value, Image):
            self.wandb.log({name: self.wandb.Image(value)}, step=self.step)

        elif isinstance(value, Tensor):
            value = np.histogram(value.view(-1).numpy())
            self.wandb.log(
                {name: self.wandb.Histogram(np_histogram=value)}, step=self.step
            )

        elif isinstance(value, Figure):
            # Wrap in wandb.Image to force static PNG rendering.
            # Passing a Figure directly lets wandb attempt a Plotly conversion
            # which can produce an invalid bargap value (-ε) for histograms.
            self.wandb.log({name: self.wandb.Image(value)}, step=self.step)

        elif isinstance(
            value,
            (float, int, self.wandb.plot.custom_chart.CustomChart),
        ):
            self.wandb.log({name: value}, step=self.step)

        elif isinstance(value, TensorImage):
            self.wandb.log(
                {name: self.wandb.Image(np.array(value))}, step=self.step
            )


class RemapLabelDataset(torch.utils.data.Dataset):
    """
    Wrapper to remap labels of a dataset without modifying the original immutable dataset.
    Exposes .targets so BalancedBatchSampler can function correctly.
    """

    def __init__(self, dataset, new_targets):
        self.dataset = dataset
        self.targets = new_targets

    def __getitem__(self, index):
        # Retrieve original item (AvalancheDataset usually returns x, y, t)
        item = self.dataset[index]
        return [item[0], self.targets[index]] + item[2:]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # Forward any other attribute access (like transform groups) to the original dataset
        return getattr(self.dataset, name)


class FileLock:
    """
    A context manager for exclusive file locking.
    Ensures the lock is held for the duration of the block and released upon exit.
    """

    def __init__(self, fname):
        self.fname = fname + ".lock"
        self.file = None

    def __enter__(self):
        self.file = open(self.fname, "w")
        if IS_WINDOWS:
            # Lock the file from the beginning (0) to the end (-1 implied by usage, though size is technically 1 here for the byte)
            # Using msvcrt.LK_LOCK (blocking)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            # Exclusive blocking lock
            fcntl.flock(self.file, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            try:
                if IS_WINDOWS:
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self.file, fcntl.LOCK_UN)
            finally:
                self.file.close()
                self.file = None


def set_task_id(model, task_id):
    """Set the task id for the model if it has a task_id attribute."""
    model = unwrap_model(model)
    if hasattr(model, "task_id"):
        model.task_id = task_id
    if hasattr(model, "model") and hasattr(model.model, "task_id"):
        model.model.task_id = task_id


def get_class_from_buffer(buffer, class_id):
    """
    Returns an Dataset containing only samples of `class_id`
    from the ClassBalancedBuffer.
    """
    # 1. Access the internal dataset
    dataset = buffer.buffer

    # 2. Robustly get targets (handle Tensor vs List)
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()

    # 3. Find indices where target matches class_id
    # numpy is often faster for this than list comprehensions for large buffers
    indices = [i for i, label in enumerate(targets) if label == class_id]

    if len(indices) == 0:
        print(f"Warning: No samples found for Class {class_id}")
        return None

    # 4. Create a Subset using these indices
    class_subset = AvalancheDataset(dataset, indices=indices)

    return class_subset


def log_wandb_metrics(wandb_logger, metric_dict, step=None):
    """Log metrics to WandB."""
    if wandb_logger is None:
        return
    if step is None:
        step = wandb_logger.wandb.run.step + 1
    for key, value in metric_dict.items():
        wandb_logger.log_single_metric(key, value, step)


def forward_rep(model, x, enforce_eval=False):
    """
    Extract feature representations from the model.

    Args:
        model: The model to extract features from
        x: Input tensor
        enforce_eval: If True, ensures model is in eval mode during extraction
                     and restores original mode afterward. Default is False for
                     backward compatibility.

    Returns:
        Feature representation tensor (batch_size, feature_dim)

    Note:
        When enforce_eval=False (legacy), assumes caller has set appropriate mode.
        When enforce_eval=True, explicitly manages model train/eval mode to ensure
        deterministic feature extraction (no dropout stochasticity).
    """
    original_mode = model.training if enforce_eval else None

    try:
        if enforce_eval and model.training:
            model.eval()
        if hasattr(model, "feature_extractor"):
            return model.feature_extractor(x)
        if hasattr(model, "forward_rep"):
            return model.forward_rep(x)
        # ReAct-compatible models expose their representation through
        # ``get_features`` rather than ``feature_extractor`` or ``forward_rep``.
        if hasattr(model, "get_features"):
            return model.get_features(x)
        if isinstance(model, MobilenetV1):
            return model.end_features(model.lat_features(x)).reshape(
                x.shape[0], -1
            )
        if isinstance(model, ViT_B_16):
            return model.encoder(model.get_backbone(x))[:, 0].reshape(
                x.shape[0], -1
            )
        if isinstance(model, ResNet50):
            feature1 = model.relu(model.bn1(model.conv1(x)))
            feature1 = model.maxpool(feature1)
            feature2 = model.layer1(feature1)
            feature3 = model.layer2(feature2)
            feature4 = model.layer3(feature3)
            feature5 = model.layer4(feature4)
            feature5 = model.avgpool(feature5)
            return feature5.view(x.size(0), -1)
        try:
            return model(x, return_feature=True)[1].reshape(x.shape[0], -1)
        except Exception as e:
            raise ValueError(
                "No get_features_method found in the model {}".format(e)
            )
    finally:
        # Restore original mode if enforce_eval was True
        if enforce_eval and original_mode is not None:
            if original_mode:
                model.train()
            else:
                model.eval()


def get_fc_w_b(fc, num_classes=None):
    """Get the weights and biases from the fully connected layer of the model."""
    if isinstance(fc, list):
        weights_list = []
        biases_list = []
        for fc_layer in fc:
            weights_list.append(fc_layer.weight.cpu().detach().numpy())
            bias = (
                fc_layer.bias
                if hasattr(fc_layer, "bias")
                else torch.zeros(fc_layer.weight.size(0))
            )
            biases_list.append(bias.cpu().detach().numpy())
        weights = np.concatenate(weights_list, axis=0)
        biases = np.stack(biases_list).reshape(-1)
        return weights, biases
    bias = fc.bias if hasattr(fc, "bias") else torch.zeros(fc.weight.size(0))
    if num_classes is None:
        return (
            fc.weight.cpu().detach().numpy(),
            bias.cpu().detach().numpy(),
        )
    return (
        fc.weight.cpu().detach().numpy()[:num_classes],
        bias.cpu().detach().numpy()[:num_classes],
    )


def get_fc_layer(model, task_id=None):
    model = unwrap_model(model)
    fc_layer = None
    if hasattr(model, "weight"):
        fc_layer = model
    if hasattr(model, "classifiers"):
        fc_layer = model.classifiers
    if hasattr(model, "classifier"):
        fc_layer = model.classifier
    if hasattr(model, "fc") and hasattr(model.fc, "weight"):
        fc_layer = model.fc
    if hasattr(model, "fc") and hasattr(model.fc, "classifier"):
        fc_layer = model.fc.classifier
    if hasattr(model, "eval_classifier") and not (model.training):
        fc_layer = model.eval_classifier
    if hasattr(model, "train_classifier") and (model.training):
        fc_layer = model.train_classifier

    if isinstance(fc_layer, nn.ModuleDict) and fc_layer is not None:
        fc_layer = (
            fc_layer[str(int(task_id))].classifier
            if task_id is not None
            else [fc_layer[i].classifier for i in fc_layer.keys()]
        )
        return fc_layer
    if fc_layer is not None:
        return fc_layer
    raise ValueError("No fc layer found in the model {}".format(dir(model.fc)))


def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    return {"data": data, "label": labels}


def collate_fn_ood(batch):
    if isinstance(batch[0], dict):
        data = torch.stack([item["data"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
    else:
        data = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
    return {"data": data, "label": labels}


def preprocess_batch(batch, device=None):
    """Preprocess input batch and put it to CUda device

    Args:
        batch (dict): dictionary or tuple of data and labels

    Returns:
        tupe: a tuple of data and labels
    """
    if type(batch) is dict:
        batch = batch["data"], batch["label"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = batch[0].to(device).float(), batch[1].to(device).long()
    return batch


def replace_transforms(dataset, new_transforms):
    if new_transforms is None:
        return dataset
    dataset.replace_current_transform_group(new_transforms)
    return dataset


def _seed_worker(worker_id):
    """Ensure each DataLoader worker has a deterministic seed derived from
    the base torch seed so that multi-worker loading is reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=None,
    batch_sampler=None,
    collate_fn=collate_fn,
):
    g = torch.Generator()
    g.manual_seed(torch.initial_seed() % 2**32)

    if batch_sampler is not None:
        return DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            pin_memory=True,
            worker_init_fn=_seed_worker,
            generator=g,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler,
        worker_init_fn=_seed_worker,
        generator=g,
    )


class BalancedBatchSampler(Sampler):
    """
    A custom PyTorch Sampler to create batches with an equal number
    of samples from each class. Can be deterministic if shuffle=False.
    """

    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)

        if self.batch_size % self.num_classes != 0:
            raise ValueError(
                f"Batch size {self.batch_size} must be divisible by the "
                f"number of classes {self.num_classes} for balanced sampling."
            )

        self.samples_per_class = self.batch_size // self.num_classes

        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)

        self.min_class_size = min(len(v) for v in self.class_indices.values())
        if self.min_class_size < self.samples_per_class:
            raise ValueError(
                f"The smallest class has only {self.min_class_size} samples, but "
                f"{self.samples_per_class} are required per batch."
            )

    def __iter__(self):
        """Yields a batch of indices."""
        if self.shuffle:
            proc_class_indices = {
                cls: np.random.permutation(indices)
                for cls, indices in self.class_indices.items()
            }
        else:
            proc_class_indices = {
                cls: np.sort(indices)
                for cls, indices in self.class_indices.items()
            }

        num_batches = self.min_class_size // self.samples_per_class

        for i in range(num_batches):
            batch_indices = []
            for cls in self.classes:
                start_idx = i * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                batch_indices.extend(
                    proc_class_indices[cls][start_idx:end_idx].tolist()
                )

            if self.shuffle:
                np.random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        """Returns the total number of batches."""
        return self.min_class_size // self.samples_per_class


def get_task_datasets(
    train_stream,
    test_stream,
    task_id,
    include_prev_tasks=False,
    transforms=None,
):
    if task_id > 0 and include_prev_tasks:
        train_dataset = ConcatDataset(
            [
                replace_transforms(t.dataset, transforms)
                for t in train_stream[: task_id + 1]
            ]
        )
        test_dataset = ConcatDataset(
            [
                replace_transforms(t.dataset, transforms)
                for t in test_stream[: task_id + 1]
            ]
        )
    else:
        train_dataset = replace_transforms(
            train_stream[task_id].dataset, transforms
        )
        test_dataset = replace_transforms(
            test_stream[task_id].dataset, transforms
        )
    return train_dataset, test_dataset


def get_task_dataloaders(
    config,
    train_stream,
    test_stream,
    task_id,
    include_prev_tasks=False,
    transforms=None,
    shuffle_train=True,
    shuffle_test=False,
    balanced_sampling=False,
    num_workers=4,
    return_task_id=False,
    test_only=False,
):
    """
    Creates dataloaders for a given task from an Avalanche stream.
    Supports balanced sampling for both training and test sets.
    Ensures deterministic order if shuffling is disabled.

    Args:
        test_only: If True, skip creating the train dataloader and return
            (None, test_loader). Useful during evaluation-only runs to
            avoid the overhead of constructing train DataLoaders.
    """

    # --- Dataset Preparation ---
    if return_task_id and task_id > 0:
        # we need to remap classes of each task to the correct global class ids
        # create wrappers without modifying the underlying datasets
        if not include_prev_tasks:
            train_dataset, test_dataset = get_task_datasets(
                train_stream,
                test_stream,
                task_id,
                include_prev_tasks=False,
                transforms=transforms,
            )
            prev_task_class_count = sum(
                len(exp.classes_in_this_experience)
                for exp in train_stream[:task_id]
            )
            new_train_targets = [
                t + prev_task_class_count for t in train_dataset.targets
            ]
            new_test_targets = [
                t + prev_task_class_count for t in test_dataset.targets
            ]
            train_dataset = RemapLabelDataset(
                train_dataset, new_train_targets
            )
            test_dataset = RemapLabelDataset(
                test_dataset, new_test_targets
            )
        else:
            all_train_datasets = [train_stream[0].dataset]
            all_test_datasets = [test_stream[0].dataset]
            prev_task_class_count = 0
            for current_task_id in range(1, task_id + 1):
                train_dataset, test_dataset = get_task_datasets(
                    train_stream,
                    test_stream,
                    current_task_id,
                    include_prev_tasks=False,
                    transforms=transforms,
                )
                prev_task_class_count += len(
                    train_stream[
                        current_task_id - 1
                    ].classes_in_this_experience
                )
                new_train_targets = [
                    t + prev_task_class_count for t in train_dataset.targets
                ]
                new_test_targets = [
                    t + prev_task_class_count for t in test_dataset.targets
                ]
                all_train_datasets.append(
                    RemapLabelDataset(train_dataset, new_train_targets)
                )
                all_test_datasets.append(
                    RemapLabelDataset(test_dataset, new_test_targets)
                )
            train_dataset = ConcatDataset(all_train_datasets)
            test_dataset = ConcatDataset(all_test_datasets)
            del all_train_datasets, all_test_datasets
    else:
        train_dataset, test_dataset = get_task_datasets(
            train_stream,
            test_stream,
            task_id,
            include_prev_tasks=include_prev_tasks,
            transforms=transforms,
        )
    # --- Create Dataloaders ---
    id_train, id_test = None, None

    # Helper function to create a dataloader
    def _create_loader(dataset, batch_size, shuffle_flag):
        batch_sampler = None
        if balanced_sampling:
            labels = (
                np.concatenate([d.targets for d in dataset.datasets])
                if isinstance(dataset, ConcatDataset)
                else dataset.targets
            )
            num_classes = len(np.unique(labels))
            if batch_size % num_classes != 0:
                # Round down to the nearest multiple of num_classes
                new_batch_size = (batch_size // num_classes) * num_classes
                if new_batch_size == 0:
                    raise ValueError(
                        f"Batch size {batch_size} is too small for {num_classes} classes."
                    )

                print(
                    f"⚠️ Warning: Original batch size {batch_size} is not divisible by the number of classes ({num_classes}). "
                    f"Adjusting batch size to {new_batch_size}."
                )
                batch_size = new_batch_size
            batch_sampler = BalancedBatchSampler(
                labels=labels, batch_size=batch_size, shuffle=shuffle_flag
            )
        return get_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle_flag,
            batch_sampler=batch_sampler,
        )

    # --- Training DataLoader ---
    if test_only:
        id_train = None
    else:
        print(
            f"Creating {'balanced' if balanced_sampling else 'standard'} training dataloader..."
        )
        id_train = _create_loader(
            dataset=train_dataset,
            batch_size=config.strategy.train_mb_size,
            shuffle_flag=shuffle_train,
        )

    # --- Test DataLoader ---
    print(
        f"Creating {'balanced' if balanced_sampling else 'standard'} test dataloader..."
    )
    id_test = _create_loader(
        dataset=test_dataset,
        batch_size=config.strategy.eval_mb_size,
        shuffle_flag=shuffle_test,
    )

    return id_train, id_test


def make_replay_collate(target_dim: int, pad_value: float = 0.0):
    """
    Collate function for replay batches with variable-width logits.
    Pads (or slices) per-sample logits to `target_dim` so torch.stack works.
    Expects each dataset item as: (x, y, tid, logits)
    where `logits` is a 1D tensor of length C_item (e.g., 2/4/6/8/10).
    """

    def _collate(batch):
        xs, ys, tids, logits_list = zip(*batch)

        # Collate x, y, tid with the default logic
        batch_x = default_collate(xs)
        batch_y = default_collate(ys)
        batch_tid = default_collate(tids)

        fixed_logits = []
        for lg in logits_list:
            # Ensure 1D: shape [C]
            if lg.dim() == 2 and lg.size(0) == 1:
                lg = lg.squeeze(0)
            assert lg.dim() == 1, f"Expected 1D logits per item, got {lg.shape}"

            c = lg.shape[-1]
            if c < target_dim:
                pad = lg.new_full((target_dim - c,), pad_value)
                lg = torch.cat([lg, pad], dim=-1)
            elif c > target_dim:
                lg = lg[:target_dim]
            fixed_logits.append(lg)

        batch_logits = torch.stack(fixed_logits, dim=0)  # [B, target_dim]
        return batch_x, batch_y, batch_tid, batch_logits

    return _collate


def safe_load_state_dict(model, checkpoint):
    model_state = model.state_dict()
    filtered_checkpoint = {}
    for k, v in checkpoint.items():
        processed_key = "".join(k.split(".")[1:])
        appended_key = "model." + k
        if k in model_state and v.shape == model_state[k].shape:
            filtered_checkpoint[k] = v
        elif (
            processed_key in model_state
            and v.shape == model_state[processed_key].shape
        ):
            filtered_checkpoint[processed_key] = v
        elif (
            appended_key in model_state
            and v.shape == model_state[appended_key].shape
        ):
            filtered_checkpoint[appended_key] = v
    load_result = model.load_state_dict(filtered_checkpoint, strict=False)
    print(
        f"Loaded state dict with safe_load_state_dict with missing {len(load_result.missing_keys)} keys."
    )
    print("Missing keys:", load_result.missing_keys)


class ModelStateContext(ContextDecorator):
    """
    Context manager that preserves the model's train/eval mode.
    Ensures that model state is restored after the context exits,
    preventing state leakage between evaluation phases.
    """

    def __init__(self, model):
        self.model = model
        self.was_training = None

    def __enter__(self):
        # Save the current training mode
        self.was_training = self.model.training
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        # Restore the original training mode
        if self.was_training:
            self.model.train()
        else:
            self.model.eval()
        return False


class ModelInferenceContext(ContextDecorator):
    """
    Context manager for inference operations that may toggle model settings.
    Now includes proper train/eval mode preservation.
    """

    def __init__(self, model, task_id=None, return_task_id=False):
        self.model = model
        self.return_task_id = return_task_id
        self.task_id = task_id
        self.was_training = None

    def __enter__(self):
        # Save the current training mode
        self.was_training = self.model.training

        # Access inner module if wrapped
        active_model = unwrap_model(self.model)

        if self.return_task_id:
            set_task_id(active_model, self.task_id)
        else:
            set_task_id(active_model, None)
            active_model.toggle_combined_logits()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        # Restore the original training mode
        if self.was_training:
            self.model.train()
        else:
            self.model.eval()

        active_model = unwrap_model(self.model)
        # Also restore toggle if it was applied
        if not self.return_task_id:
            active_model.toggle_combined_logits()
        return False


def select_id_threshold_tpr95(id_conf):
    # Lower score => "more OOD". Threshold at 5th percentile of ID scores ⇒ ~TPR=95% on ID.
    if len(id_conf) == 0 or np.all(np.isnan(id_conf)):
        return None
    return np.percentile(id_conf, 5.0)


def log_ood_failure_modes(id_conf, id_gt, ood_conf, ood_gt, threshold=None):
    """
    Calculates False Positive and False Negative rates.

    Args:
        id_conf: Confidence scores of ID data (higher = more confident ID)
        id_gt: Ground truth labels for ID (unused for mask, used for shape)
        ood_conf: Confidence scores of OOD data
        ood_gt: Ground truth labels for OOD
        threshold: (Optional) A fixed float threshold.
                   If None, calculates threshold for FPR@95 on the provided id_conf.
    """
    # 1. Determine Threshold
    if threshold is None:
        # Optimization: Use percentile instead of loop for exact FPR@95
        # We want the threshold where 5% of ID data is < thresh (misclassified as OOD)
        # Assumes: Low Confidence = OOD
        threshold = np.percentile(id_conf, 5)

    # 2. Generate Predictions
    # Samples with confidence < threshold are flagged as OOD (Positive Class)
    id_pred = id_conf < threshold
    ood_pred = ood_conf < threshold

    # 3. Calculate Failure Modes

    # False Positive: ID sample predicted as OOD (1)
    # Since id_pred is 1 when OOD, sum(id_pred) counts ID samples called OOD
    fp_count = np.sum(id_pred)
    fp_rate = fp_count / max(len(id_conf), 1)

    # False Negative: OOD sample predicted as ID (0)
    # Since ood_pred is 0 when ID, sum(~ood_pred) counts OOD samples called ID
    fn_count = np.sum(~ood_pred)
    fn_rate = fn_count / max(len(ood_conf), 1)

    return fp_rate, fn_rate, int(fp_count), int(fn_count)


# Alignment Measurement tool
@torch.no_grad()
def get_embeddings_for_alignment(
    dataloader,
    embedding_model,
    device,
    task_identifier="unknown_task",
    dataset_name="unknown_dataset",
    num_batches=3,
    return_sids=False,
):
    if return_sids:
        assert not (
            isinstance(dataloader.sampler, torch.utils.data.RandomSampler)
        ), (
            "Error: Dataloader must not be shuffled (use shuffle=False) "
            "when return_sids is True to ensure stable sample IDs."
        )
    all_embeddings_list = []
    all_labels = []
    all_sids_list = []
    for i, batch in enumerate(dataloader):
        images, labels = preprocess_batch(batch)
        images = images.to(device)
        try:
            embeddings = embedding_model(images)
            embeddings = embeddings.view(images.shape[0], -1)
            if labels.shape[0] == 0:
                continue
            labels = labels.cpu().numpy()
            all_embeddings_list.append(embeddings.cpu().numpy())
            all_labels.append(labels)
            sids_full_batch = [
                f"{dataset_name}:{labels[j]}:{i}:{j}"
                for j in range(len(labels))
            ]
            sids_to_add = np.array(sids_full_batch, dtype=object)
            all_sids_list.extend(sids_to_add)
        except Exception as e:
            print(
                f"Error during embedding extraction ({task_identifier}, batch {i}): {e}"
            )
            continue
        if i >= num_batches - 1:
            break

    if not all_embeddings_list:
        return None

    embeddings = np.concatenate(all_embeddings_list, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    final_sids = np.array(all_sids_list, dtype=object)
    if return_sids:
        return embeddings, labels, final_sids
    return embeddings, labels


def compute_new_ce_loss(strategy):
    """Compute cross-entropy over only the classes introduced in this task.

    Avalanche's default LwF and weight-alignment loss includes old classes in
    the softmax denominator for new-class samples. Restricting the logits to
    the current classes avoids that unintended source of forgetting.

    Args:
        strategy: Active Avalanche training strategy.

    Returns:
        Scalar cross-entropy loss for the current experience.
    """
    if strategy.experience.current_experience == 0:
        # use the default loss for the first experience
        return strategy.loss
    current_classes = list(strategy.experience.classes_in_this_experience)
    class_to_idx = {c: i for i, c in enumerate(current_classes)}
    mask = torch.tensor(
        [y.item() in class_to_idx for y in strategy.mb_y],
        device=strategy.device,
    )
    new_class_indices = torch.tensor(current_classes, device=strategy.device)
    new_logits_sliced = strategy.mb_output[mask][:, new_class_indices]

    # Remap targets to 0..N-1 relative to this slice
    mapped_targets = torch.tensor(
        [class_to_idx[y.item()] for y in strategy.mb_y[mask]],
        device=strategy.device,
    )
    if not (mask.any()):
        print("Warning: No samples from current classes in the mini-batch.")
        return torch.tensor(0.0, device=strategy.device)
    return F.cross_entropy(new_logits_sliced, mapped_targets)


def compute_true_stream_forgetting(cl_results, n_experiences, return_task_id):
    """
    cl_results: list[dict] of metrics dict returned after each eval phase (one per task).
    n_experiences: total number of experiences
    return_task_id: your config.scenario.return_task_id (affects the task key)
    Returns: (avg_forgetting, per_exp_forgetting_dict)
    """
    # Build per-experience accuracy timeline across phases
    # phase t corresponds to "after finishing task t"
    acc_timeline = {exp_id: [] for exp_id in range(n_experiences)}

    for phase_id, res in enumerate(cl_results):
        # experience index equals task index in your setup
        for exp_id in range(phase_id + 1):  # only experiences seen so far
            if return_task_id:
                # Task + Exp are equal to exp_id in your logs
                key = f"Top1_Acc_Exp/eval_phase/test_stream/Task{exp_id:0>3}/Exp{exp_id:0>3}"
            else:
                # If not multihead, Avalanche often reports just Exp{exp_id}
                key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{exp_id:0>3}"
            val = res.get(key, None)
            # Be defensive about missing keys
            if val is None:
                # Try a fallback without Task prefix (some strategies log like this)
                key2 = f"Top1_Acc_Exp/eval_phase/test_stream/Exp{exp_id:0>3}"
                val = res.get(key2, None)
            if val is not None:
                acc_timeline[exp_id].append(float(val))

    per_exp_forgetting = {}
    for exp_id, series in acc_timeline.items():
        if not series:
            per_exp_forgetting[exp_id] = 0.0
            continue
        if len(series) == 1:
            # First time the exp appears — no forgetting yet
            per_exp_forgetting[exp_id] = 0.0
            continue
        peak_before_last = max(series[:-1])
        last = series[-1]
        per_exp_forgetting[exp_id] = max(0.0, peak_before_last - last)

    # Average across experiences that actually appeared
    valid_exps = [
        v
        for exp_id, v in per_exp_forgetting.items()
        if exp_id < n_experiences - 1  # exclude last
    ]
    avg_forgetting = float(np.mean(valid_exps)) if valid_exps else 0.0
    return avg_forgetting, per_exp_forgetting


def normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-8)


def garbage_collect(release_cuda_cache=True):
    gc.collect()
    if release_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def set_seed(seed):
    """Sets all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # These are crucial for CUDNN reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Enforce deterministic algorithms globally (raises error on
    # non-deterministic ops so they are caught early).
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"All random seeds set to {seed}")


def extract_features(model, batch):
    data, label = preprocess_batch(batch)
    bsz = data.shape[0]
    f = forward_rep(model, data).view(bsz, -1).detach().cpu().numpy()
    label = label.cpu().numpy() if hasattr(label, "cpu") else np.array(label)
    return f, label


class StrategyStateHelper:
    @staticmethod
    def _is_savable(value):
        # 1. Primitives & Tensors (Base Case)
        if isinstance(
            value, (torch.Tensor, np.ndarray, int, float, str, bool, bytes)
        ):
            return True

        # 2. Lists & Tuples (Recursive)
        # This handles nested lists (like y_memory) by checking the first element
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return True
            return StrategyStateHelper._is_savable(value[0])

        # 3. Dictionaries (Recursive)
        if isinstance(value, dict):
            if len(value) == 0:
                return True
            # Check the first value to decide if the dict contains savable data
            return StrategyStateHelper._is_savable(next(iter(value.values())))

        # 4. Fallback: Skip complex objects (Plugins, Loggers) silently
        return False

    @staticmethod
    def save(strategy, filepath):
        print(f"Helper: Saving strategy state to {filepath}...")
        raw_model = unwrap_model(strategy.model)
        state = {
            "model": raw_model.state_dict(),
            "steps": strategy.clock.train_exp_counter,
            "plugins": {},
            "strategy_attrs": {},
        }

        # iCarl's raw-image buffer is redirected to a single shared buffer file
        # (overwritten each task) to avoid ~12 GB duplicated per task checkpoint.
        icarl_plugin = StrategyStateHelper._find_icarl_plugin(strategy)
        _ICARL_BUFFER_ATTRS = {"x_memory", "y_memory", "order"}

        # 1. Save Strategy Attributes (e.g. accumulated losses)
        for attr_name, attr_value in strategy.__dict__.items():
            if StrategyStateHelper._is_savable(attr_value):
                state["strategy_attrs"][attr_name] = attr_value

        # 2. Save Plugin Attributes (e.g. x_memory, y_memory)
        for plugin in strategy.plugins:
            plugin_name = plugin.__class__.__name__
            plugin_state = {"_attrs": {}, "_modules": {}}
            for attr_name, attr_value in plugin.__dict__.items():
                # Redirect iCarl's large image buffer to the shared buffer file.
                if plugin is icarl_plugin and attr_name in _ICARL_BUFFER_ATTRS:
                    continue
                if isinstance(attr_value, nn.Module):
                    # Save state_dict + class info so the module can be reconstructed
                    # even when the plugin attribute is None/uninitialised at load time
                    # (e.g. BiC's bias_layer is created lazily after exp 0).
                    # Pickling full modules fails for modules with local lambdas,
                    # so we save state_dict only and reconstruct from class name.
                    plugin_state["_modules"][attr_name] = {
                        "state_dict": attr_value.state_dict(),
                        "class_module": type(attr_value).__module__,
                        "class_qualname": type(attr_value).__qualname__,
                    }
                elif StrategyStateHelper._is_savable(attr_value):
                    plugin_state["_attrs"][attr_name] = attr_value

            if plugin_state["_attrs"] or plugin_state["_modules"]:
                state["plugins"][plugin_name] = plugin_state

        # 3. Special Case: NCM Class Means
        raw_model = unwrap_model(strategy.model)
        if hasattr(raw_model, "eval_classifier") and hasattr(
            raw_model.eval_classifier, "class_means_dict"
        ):
            state["ncm_class_means"] = (
                raw_model.eval_classifier.class_means_dict
            )

        torch.save(state, filepath)

        # 4. Save the replay buffer (iCarl images + storage_policy) to a single
        #    shared file that is overwritten every task — keeps per-task
        #    checkpoints small and enables correct resume for all replay methods.
        StrategyStateHelper._save_buffer_file(strategy, filepath, icarl_plugin)

    @staticmethod
    def load(strategy, filepath, device="cpu"):
        if not os.path.exists(filepath):
            return False

        print(f"Helper: Loading strategy state from {filepath}...")
        checkpoint = torch.load(filepath, map_location=device)

        raw_model = unwrap_model(strategy.model)

        # Use safe_load_state_dict if available, otherwise fallback to load_state_dict
        if "safe_load_state_dict" in globals():
            safe_load_state_dict(raw_model, checkpoint["model"])
        else:
            raw_model.load_state_dict(checkpoint["model"])

        strategy.clock.train_exp_counter = checkpoint["steps"]

        # Restore Strategy Attributes
        for attr, value in checkpoint.get("strategy_attrs", {}).items():
            setattr(strategy, attr, value)

        # Restore Plugin Attributes
        if "plugins" in checkpoint:
            for plugin in strategy.plugins:
                plugin_name = plugin.__class__.__name__
                if plugin_name in checkpoint["plugins"]:
                    saved_state = checkpoint["plugins"][plugin_name]
                    if "_attrs" in saved_state or "_modules" in saved_state:
                        # New format: state_dicts (+ class info) for nn.Module attrs.
                        for attr_name, module_data in saved_state.get(
                            "_modules", {}
                        ).items():
                            # Support both old flat state_dict and new dict-with-class-info
                            if (
                                isinstance(module_data, dict)
                                and "state_dict" in module_data
                            ):
                                module_sd = module_data["state_dict"]
                                class_module = module_data.get("class_module")
                                class_qualname = module_data.get(
                                    "class_qualname"
                                )
                            else:
                                # Legacy sub-format: flat state_dict, no class info
                                module_sd = module_data
                                class_module = class_qualname = None

                            current = getattr(plugin, attr_name, None)
                            if isinstance(current, nn.Module):
                                current.load_state_dict(module_sd, strict=False)
                            else:
                                # Module not yet initialised (e.g. BiC bias_layer is None
                                # until bias_correction_step runs in after_training_exp).
                                # Training may be skipped entirely (checkpoint exists), so
                                # a deferred hook on after_training_exp would never fire.
                                # Instead, reconstruct the module immediately from the
                                # saved class name + state_dict.
                                reconstructed = (
                                    StrategyStateHelper._reconstruct_module(
                                        class_module, class_qualname, module_sd
                                    )
                                )
                                if reconstructed is not None:
                                    reconstructed.to(device)
                                    setattr(plugin, attr_name, reconstructed)
                                    print(
                                        f"  [StrategyStateHelper] Reconstructed "
                                        f"{plugin.__class__.__name__}.{attr_name} "
                                        f"from {class_qualname}."
                                    )
                                else:
                                    print(
                                        f"  [StrategyStateHelper] WARNING: could not "
                                        f"reconstruct {plugin.__class__.__name__}.{attr_name} "
                                        f"({class_qualname}) — attribute left as None."
                                    )
                        for attr, value in saved_state.get(
                            "_attrs", {}
                        ).items():
                            setattr(plugin, attr, value)
                    else:
                        # Legacy format: flat dict of savable attrs
                        for attr, value in saved_state.items():
                            setattr(plugin, attr, value)

        # Restore NCM Class Means
        raw_model = unwrap_model(strategy.model)
        if (
            "ncm_class_means" in checkpoint
            and hasattr(raw_model, "eval_classifier")
            and hasattr(raw_model.eval_classifier, "replace_class_means_dict")
        ):
            raw_model.eval_classifier.replace_class_means_dict(
                checkpoint["ncm_class_means"]
            )
            print("  Restored NCM Class Means successfully.")

        # Restore replay buffer (iCarl images + storage_policy for BiC/DER)
        StrategyStateHelper._load_buffer_file(strategy, filepath)

        return True

    # ------------------------------------------------------------------
    # Buffer-file helpers (shared across iCarl, DER, BiC, Replay, …)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_buffer_path(checkpoint_path):
        """Derive the shared buffer file path from a per-task checkpoint path.

        e.g. ``experiment_3.pth``  →  ``experiment_buffer.pth``
        """
        dirname = os.path.dirname(checkpoint_path)
        basename = os.path.basename(checkpoint_path)
        new_basename = re.sub(r"_\d+\.pth$", "_buffer.pth", basename)
        if new_basename == basename:  # no trailing _N.pth found
            name, ext = os.path.splitext(basename)
            new_basename = f"{name}_buffer{ext}"
        return os.path.join(dirname, new_basename)

    @staticmethod
    def _find_icarl_plugin(strategy):
        """Return the iCarl plugin by duck-typing (x_memory / y_memory / order)."""
        for p in strategy.plugins:
            if (
                hasattr(p, "x_memory")
                and hasattr(p, "y_memory")
                and hasattr(p, "order")
            ):
                return p
        return None

    @staticmethod
    def _find_storage_policy(strategy):
        """Return the replay storage_policy from the strategy or its plugins."""
        sp = getattr(strategy, "storage_policy", None)
        if sp is not None:
            return sp
        for p in strategy.plugins:
            sp = getattr(p, "storage_policy", None)
            if sp is not None:
                return sp
        return None

    @staticmethod
    def _set_storage_policy(strategy, saved_sp):
        """Replace the storage_policy on the strategy or the first plugin that has one."""
        if hasattr(strategy, "storage_policy"):
            strategy.storage_policy = saved_sp
            return True
        for p in strategy.plugins:
            if hasattr(p, "storage_policy"):
                p.storage_policy = saved_sp
                return True
        return False

    @staticmethod
    def _save_buffer_file(strategy, checkpoint_path, icarl_plugin=None):
        """Save the replay buffer to a single shared file (overwritten each task).

        Separating the buffer from per-task checkpoints prevents ~12 GB of
        exemplar images from being duplicated into every checkpoint file while
        still supporting correct resume for all replay-based methods.
        """
        if icarl_plugin is None:
            icarl_plugin = StrategyStateHelper._find_icarl_plugin(strategy)

        buf_state = {}

        # iCarl: x_memory / y_memory / order (plain CPU tensors — always picklable)
        if icarl_plugin is not None and len(icarl_plugin.x_memory) > 0:
            buf_state["icarl"] = {
                "x_memory": [x.cpu() for x in icarl_plugin.x_memory],
                "y_memory": icarl_plugin.y_memory,
                "order": [
                    o.cpu() if isinstance(o, torch.Tensor) else o
                    for o in icarl_plugin.order
                ],
            }

        # storage_policy (DER, BiC, Replay, …): probe picklability first
        sp = StrategyStateHelper._find_storage_policy(strategy)
        if sp is not None:
            probe = io.BytesIO()
            try:
                torch.save(sp, probe)
                buf_state["storage_policy"] = sp
            except Exception as exc:
                print(
                    f"  [Buffer] WARNING: storage_policy ({type(sp).__name__}) "
                    f"could not be pickled ({type(exc).__name__}: {exc}); "
                    "it will not be restored on checkpoint resume."
                )

        if not buf_state:
            return

        buf_path = StrategyStateHelper._get_buffer_path(checkpoint_path)
        print(f"Helper: Saving buffer to {buf_path}...")
        torch.save(buf_state, buf_path)

    @staticmethod
    def _load_buffer_file(strategy, checkpoint_path):
        """Restore the replay buffer from the shared buffer file if present."""
        buf_path = StrategyStateHelper._get_buffer_path(checkpoint_path)
        if not os.path.exists(buf_path):
            return

        print(f"Helper: Loading buffer from {buf_path}...")
        buf_state = torch.load(buf_path, map_location="cpu")

        # iCarl buffer
        if "icarl" in buf_state:
            icarl_plugin = StrategyStateHelper._find_icarl_plugin(strategy)
            if icarl_plugin is not None:
                bd = buf_state["icarl"]
                icarl_plugin.x_memory = bd["x_memory"]  # list of CPU tensors
                icarl_plugin.y_memory = bd["y_memory"]
                icarl_plugin.order = bd["order"]
                print(
                    f"  Restored iCarl buffer: "
                    f"{len(icarl_plugin.x_memory)} class exemplar sets."
                )

        # storage_policy (DER / BiC / Replay / …)
        if "storage_policy" in buf_state:
            if StrategyStateHelper._set_storage_policy(
                strategy, buf_state["storage_policy"]
            ):
                print("  Restored storage_policy buffer.")
            else:
                print("  [Buffer] WARNING: no storage_policy found to restore.")

    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_module(class_module, class_qualname, state_dict):
        """
        Reconstruct an nn.Module from its class name and state_dict.

        Needed when a plugin attribute is None at load time (e.g. BiC's bias_layer
        is created lazily inside bias_correction_step, which only runs during
        after_training_exp — but training is skipped when a checkpoint exists,
        so a deferred hook would never fire).

        Strategy:
          1. Import the class.
          2. Try cls() with no args.
          3. Try cls(v) for each tensor v in the state_dict (handles modules like
             BiasLayer that need a single positional arg, e.g. the class indices
             stored as the 'clss' buffer in the state_dict).
          4. Return None if all attempts fail.
        """
        if class_module is None or class_qualname is None:
            return None
        try:
            import importlib

            mod = importlib.import_module(class_module)
            cls = getattr(mod, class_qualname)

            # Attempt 1: no-arg constructor
            try:
                module = cls()
                module.load_state_dict(state_dict, strict=False)
                return module
            except (TypeError, RuntimeError):
                pass  # TypeError (needs args) or RuntimeError (shape mismatch)

            # Attempt 2: pass each tensor in the state_dict as the sole positional
            # arg — covers BiasLayer(clss) and similarly structured modules.
            for val in state_dict.values():
                try:
                    module = cls(val)
                    module.load_state_dict(state_dict, strict=False)
                    return module
                except Exception:
                    continue

        except Exception as e:
            print(f"  [StrategyStateHelper] _reconstruct_module failed: {e}")

        return None


class ExperimentGuard:
    """Isolate failures in task-level training and evaluation blocks.

    On failure, the guard prints the traceback, reports the error to W&B when a
    logger is available, and suppresses the exception so later tasks can run.
    Keyboard interrupts are re-raised by default.
    """

    def __init__(
        self,
        description="Operation",
        wandb_logger=None,
        reraise_keyboard_interrupt=True,
    ):
        self.description = description
        self.logger = wandb_logger
        self.reraise_keyboard_interrupt = reraise_keyboard_interrupt

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. No exception happened
        if exc_type is None:
            return False

        # 2. Allow KeyboardInterrupt (Ctrl+C) to still kill the process
        if self.reraise_keyboard_interrupt and issubclass(
            exc_type, KeyboardInterrupt
        ):
            return False  # Do not suppress

        # 3. Handle the Crash
        error_header = f"\n{'!'*40}\n[GUARD] CRITICAL FAILURE IN: {self.description}\n{'!'*40}"

        # Format traceback
        tb_str = "".join(traceback.format_tb(exc_tb))
        full_error_msg = f"{error_header}\nError Type: {exc_type.__name__}\nError Message: {exc_val}\n\nTraceback:\n{tb_str}\n{'='*40}\n"

        # A. Print to Console (Slurm/local logs)
        print(full_error_msg, file=sys.stderr, flush=True)

        # B. Log to WandB (so you see it in the dashboard)
        if self.logger and hasattr(self.logger, "log"):
            try:
                # Log as a text table or simple alert
                self.logger.wandb.alert(
                    title=f"Crash in {self.description}",
                    text=f"Exception: {exc_val}\n\nCheck logs for traceback.",
                    level=self.logger.wandb.AlertLevel.ERROR,
                )
            except:
                pass  # Fail silently if WandB itself is broken

        # 4. Return True to SUPPRESS exception and continue execution
        print(
            f"⚠️ Skipping {self.description} and continuing experiment...\n",
            flush=True,
        )
        return True


class RunningMetric:
    def __init__(self):
        # sums[ood_type][metric] -> float
        self.sums = defaultdict(lambda: defaultdict(float))
        # counts[ood_type][metric] -> int (number of samples or batches weighted)
        self.counts = defaultdict(lambda: defaultdict(float))

    def update(self, ood_type, metric_name, value, weight=1.0):
        self.sums[ood_type][metric_name] += float(value) * float(weight)
        self.counts[ood_type][metric_name] += float(weight)

    def mean(self, ood_type, metric_name):
        denom = self.counts[ood_type].get(metric_name, 0.0)
        return (
            self.sums[ood_type].get(metric_name, 0.0) / denom
            if denom > 0
            else float("nan")
        )

    def summary(self):
        out = {}
        for ood, metrics in self.sums.items():
            out[ood] = {m: self.mean(ood, m) for m in metrics.keys()}
        return out


class AverageIncrementalAccuracy(PluginMetric[float]):
    """
    Computes the Average Accuracy over all incremental stages (B stages).
    Formula: A_bar = (1/B) * sum(A_b)
    where A_b is the stream-level accuracy after the b-th task.
    """

    def __init__(self):
        super().__init__()
        # We use Avalanche's base Accuracy class to do the math safely
        self._current_stream_accuracy = Accuracy()
        self.a_b_scores = []

    def reset(self):
        """Resets the internal accuracy tracker for the current eval stream."""
        self._current_stream_accuracy.reset()

    def result(self):
        """Calculates the average of all recorded stream accuracies."""
        if len(self.a_b_scores) == 0:
            return 0.0
        return sum(self.a_b_scores) / len(self.a_b_scores)

    def before_eval(self, strategy, **kwargs):
        """Called right before the evaluation phase starts."""
        self.reset()

    def after_eval_iteration(self, strategy, **kwargs):
        """Called after every batch in the evaluation stream."""
        # Update the accuracy with the predictions and true labels of the batch
        self._current_stream_accuracy.update(strategy.mb_y, strategy.mb_output)

    def after_eval(self, strategy, **kwargs):
        """Called at the end of the entire evaluation phase."""
        # 1. Get the final accuracy for this specific evaluation stream (A_b)
        current_a_b = self._current_stream_accuracy.result()

        # 2. Append it to our list of historical task accuracies
        self.a_b_scores.append(current_a_b)

        # 3. Compute the overall average accuracy up to this point
        a_bar = self.result()

        # 4. Emit the metric so the logger (e.g., TensorBoard, TextLogger) captures it
        return [
            MetricValue(
                self,
                name="Metric/Average_Incremental_Accuracy",
                value=a_bar,
                x_plot=strategy.clock.train_iterations,
            )
        ]

    def __str__(self):
        return "Average_Incremental_Accuracy"
