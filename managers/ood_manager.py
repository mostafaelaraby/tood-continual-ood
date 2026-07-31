from copy import deepcopy as copy
from dataclasses import dataclass, field

import numpy as np
import torch
from avalanche.benchmarks.utils import AvalancheDataset, concat_datasets
from avalanche.training import ClassBalancedBuffer
from scipy.stats import median_abs_deviation
from torch.utils.data import ConcatDataset, Subset

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import get_postprocessor
from utils import OOD_METRIC_List
from utils.helpers import (
    ModelInferenceContext,
    collate_fn,
    get_class_from_buffer,
    get_dataloader,
    log_ood_failure_modes,
    log_wandb_metrics,
    select_id_threshold_tpr95,
)


class OODPostprocessorManager:
    """
    Manages a SINGLE OOD postprocessor instance.

    This class handles the setup, threshold computation, and evaluation logic,
    using the postprocessor's .inference() method.
    """

    # Base postprocessors calibrate each task's threshold at peak performance
    # and can reuse it at later checkpoints.
    reuse_peak_thresholds = True

    def __init__(
        self,
        config,
        cl_strategy,
        benchmark,
        ood_loader_dict,
        wandb_logger,
        buffer_max_size=20,
    ):
        print("[OOD Manager] Initializing single postprocessor...")
        self.config = config
        self.cl_strategy = cl_strategy
        self.benchmark = benchmark
        self.postprocessor = get_postprocessor(config)
        self.ood_loader_dict = ood_loader_dict
        self.wandb_logger = wandb_logger
        self.num_workers = config.dataset.num_workers

        # State management
        self.thresholds = {}
        self.bias_analyzed_for = set()

        # Aggregated OOD statistics tracking
        self._aggregated_ood_scores = {}

        # Class-balanced replay buffer for postprocessor setup
        self.buffer = ClassBalancedBuffer(
            max_size=buffer_max_size, adaptive_size=True
        )
        self._buffer_updated = set()

    def _ensure_buffer_updated(self, current_task_id):
        """Updates the class-balanced buffer with the current task's dataset."""
        if current_task_id in self._buffer_updated:
            return
        dataset = self.benchmark.train_stream[current_task_id].dataset
        self.buffer.update_from_dataset(dataset)
        self._buffer_updated.add(current_task_id)

    def _make_combined_setup_loader(self, current_task_id, id_train_loader):
        """Combines current task data with old buffer exemplars for postprocessor setup."""
        current_dataset = self.benchmark.train_stream[current_task_id].dataset
        old_exemplars = []
        for task_id in range(current_task_id):
            classes = self.benchmark.train_stream[
                task_id
            ].classes_in_this_experience
            for c in classes:
                exemplar_ds = get_class_from_buffer(self.buffer, c)
                if exemplar_ds is not None:
                    old_exemplars.append(exemplar_ds)

        if not old_exemplars:
            return id_train_loader

        combined = ConcatDataset([current_dataset] + old_exemplars)
        return get_dataloader(
            combined,
            batch_size=self.config.strategy.eval_mb_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

    @torch.enable_grad()
    def _setup_postprocessor(
        self,
        id_train_loader,
        id_val_loader,
        ood_loader_dict,
        num_classes,
        current_task_id=None,
    ):
        """Internal helper to run the setup for the postprocessor instance."""
        model = self.cl_strategy.model
        self.update_num_classes(num_classes)
        self.postprocessor.setup_flag = False

        if current_task_id is not None:
            self._ensure_buffer_updated(current_task_id)
            if current_task_id > 0:
                id_train_loader = self._make_combined_setup_loader(
                    current_task_id, id_train_loader
                )

        with ModelInferenceContext(model):
            self.postprocessor.setup(
                model,
                {"train": id_train_loader, "val": id_val_loader},
                ood_loader_dict,
            )

        if self.config.postprocessor.name == "react":
            self.postprocessor.threshold = np.percentile(
                self.postprocessor.activation_log.flatten(),
                self.postprocessor.percentile,
            )

    def _get_or_cache_threshold(
        self, id_conf, current_task_id, id_data_task_id, invalidate_cache=True
    ):
        """Gets the OOD threshold from cache or calculates and caches a new one."""
        cache_key = id_data_task_id

        if current_task_id == id_data_task_id or invalidate_cache:
            print(
                f"Peak performance for Task {id_data_task_id}. Caching new threshold."
            )
            thr = select_id_threshold_tpr95(np.asarray(id_conf))
            self.thresholds[cache_key] = thr
        else:
            print(
                f"Evaluating old Task {id_data_task_id} data. Using cached threshold."
            )
            thr = self.thresholds.get(cache_key)
            if thr is None:
                print(
                    f"WARNING: Threshold for {cache_key} not found! Recalculating."
                )
                thr = select_id_threshold_tpr95(np.asarray(id_conf))

        return thr

    def setup(
        self, id_train_loader, id_val_loader, num_classes, current_task_id=None
    ):
        """Public method to setup the postprocessor."""
        self._setup_postprocessor(
            id_train_loader,
            id_val_loader,
            self.ood_loader_dict,
            num_classes,
            current_task_id=current_task_id,
        )

    def update_num_classes(self, num_classes):
        """Updates the number of classes in the postprocessor and model."""
        self.postprocessor.num_classes = num_classes
        self.cl_strategy.model.num_classes = num_classes

    def inference(self, data_loader, return_logits=False, return_feature=False):
        """Public wrapper to call the underlying OpenOOD postprocessor's inference."""
        self.cl_strategy.model.is_ood_eval = True
        try:
            return self.postprocessor.inference(
                self.cl_strategy.model,
                data_loader,
                return_logits=return_logits,
                return_feature=return_feature,
            )
        finally:
            # Do not leak OOD-evaluation mode into later ID/CL evaluation if
            # the postprocessor raises while iterating over a data loader.
            self.cl_strategy.model.is_ood_eval = False

    def analyze_bias_vs_distance(self, current_task_id: int):
        """Logs 'Bias' by comparing Current Task's mean score against Past Tasks'."""
        if current_task_id in self.bias_analyzed_for:
            return

        print(
            f"[Analysis] Calculating Oracle Bias for Task {current_task_id}...",
            flush=True,
        )

        current_mean, _ = self._get_ood_conf_stats_from_stream(current_task_id)
        if current_mean is None:
            return

        log_payload = {}
        for past_task_id in range(current_task_id):
            past_mean, _ = self._get_ood_conf_stats_from_stream(past_task_id)
            if past_mean is None:
                continue

            distance = current_task_id - past_task_id
            bias = abs(past_mean - current_mean)
            log_payload[f"Analysis/Bias_vs_Distance/dist_{distance}"] = bias

        if self.wandb_logger and log_payload:
            log_wandb_metrics(self.wandb_logger, log_payload)

        self.bias_analyzed_for.add(current_task_id)

    def _get_ood_conf_stats_from_stream(self, task_id, num_samples=128):
        """Helper to fetch a subset of original training data and calculate stats."""
        try:
            dataset = self.benchmark.train_stream[task_id].dataset
            # Use a fixed-seed RNG to avoid mutating the global numpy state,
            # which would cause non-deterministic AUROC scores across runs.
            rng = np.random.RandomState(seed=task_id)
            indices = rng.choice(
                len(dataset), size=min(num_samples, len(dataset)), replace=False
            )
            subset = Subset(dataset, indices)
            loader = get_dataloader(
                subset,
                batch_size=self.config.strategy.eval_mb_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                shuffle=False,
            )

            with ModelInferenceContext(self.cl_strategy.model):
                _, conf, _ = self.inference(loader)

            return float(np.mean(conf)), float(np.std(conf))

        except Exception as e:
            print(
                f"Warning: [Analysis] Failed to fetch stats for task {task_id}: {e}"
            )
            return None, None

    def eval_for_id_task(
        self,
        id_train_loader,
        id_test_loader,
        current_task_id,
        id_data_task_id,
        return_logits=False,
        return_feature=False,
        all_conf={},
        skip_train_inference=False,
    ):
        """Performs OOD evaluation against a specific in-distribution task.

        Args:
            skip_train_inference: If True, skip the forward pass on training
                data and derive the OOD threshold from test confidence instead.
                This eliminates the most expensive part of the inner eval loop
                for large datasets (e.g. ImageNet with 100 tasks).
        """
        if current_task_id == id_data_task_id and not skip_train_inference:
            self.analyze_bias_vs_distance(current_task_id)
        return_task_id = self.config.scenario.return_task_id

        try:
            with ModelInferenceContext(
                self.cl_strategy.model,
                return_task_id=return_task_id,
                task_id=id_data_task_id,
            ):
                id_data = self.inference(
                    id_test_loader,
                    return_logits=return_logits,
                    return_feature=return_feature,
                )
                if skip_train_inference:
                    train_conf = None
                else:
                    _, train_conf, _ = self.inference(
                        id_train_loader,
                        return_logits=False,
                    )

            if return_logits and return_feature:
                id_preds, id_conf, id_gt, id_logits, id_features = id_data
            elif return_logits:
                id_preds, id_conf, id_gt, id_logits = id_data
            elif return_feature:
                id_preds, id_conf, id_gt, id_features = id_data
                id_logits = None
            else:
                id_preds, id_conf, id_gt = id_data
                id_logits = None
                id_features = None

        except Exception as e:
            print(
                f"Error {e} in ID data for task {id_data_task_id}, skipping.",
                flush=True,
            )
            id_conf = np.array([])
            id_gt = np.array([])
            id_preds = np.array([])
            id_logits = None
            id_features = None
            train_conf = None

        all_conf["ID"] = id_conf
        all_conf["ID_gt"] = id_gt
        all_conf["ID_preds"] = id_preds

        # When train inference is skipped, use test confidence for threshold
        threshold = self._get_or_cache_threshold(
            train_conf if train_conf is not None else id_conf,
            current_task_id,
            id_data_task_id,
            invalidate_cache=not skip_train_inference,
        )

        sids = [f"ID:{id_data_task_id}:{i}" for i in range(len(id_conf))]

        if return_logits and return_feature:
            return (
                sids,
                threshold,
                id_conf,
                train_conf,
                id_preds,
                id_logits,
                id_features,
            )
        if return_logits:
            return sids, threshold, id_conf, train_conf, id_preds, id_logits
        if return_feature:
            return sids, threshold, id_conf, train_conf, id_preds, id_features
        return sids, threshold, id_conf, train_conf, id_preds

    def _map_class_preds_to_task(self, preds, current_task_id):
        """Maps predicted class labels to their corresponding Task IDs."""
        task_preds = np.zeros_like(preds)
        for task_id in range(current_task_id + 1):
            classes_in_task = self.benchmark.train_stream[
                task_id
            ].classes_in_this_experience
            mask = np.isin(preds, classes_in_task)
            task_preds[mask] = task_id
        return task_preds

    def log_ood_scores(
        self,
        task_name,
        ood_split,
        ood_conf,
        all_conf,
        dataset_name=None,
        ood_preds=None,
        current_task_id=None,
    ):
        """Logs OOD stats per dataset including energy metrics and aggregates by nearood/farood."""
        prefix_base = f"OOD_Scores/{task_name}"
        if dataset_name:
            prefix_base += f"/{dataset_name}"

        log_data = {
            f"{prefix_base}/id_mean": all_conf["ID"].mean().item(),
            f"{prefix_base}/id_std": all_conf["ID"].std().item(),
            f"{prefix_base}/{ood_split}_mean": ood_conf.mean().item(),
            f"{prefix_base}/{ood_split}_std": ood_conf.std().item(),
        }

        # Log energy margin statistics if available
        if "ID_margin_energy" in all_conf:
            id_margin = all_conf["ID_margin_energy"]
            ood_margin = all_conf.get("ood_margin_energy")

            log_data[f"{prefix_base}/id_margin_mean"] = float(
                np.mean(id_margin)
            )
            log_data[f"{prefix_base}/id_margin_std"] = float(np.std(id_margin))
            log_data[f"{prefix_base}/id_margin_median"] = float(
                np.median(id_margin)
            )

            if ood_margin is not None:
                log_data[f"{prefix_base}/{ood_split}_margin_mean"] = float(
                    np.mean(ood_margin)
                )
                log_data[f"{prefix_base}/{ood_split}_margin_std"] = float(
                    np.std(ood_margin)
                )
                log_data[f"{prefix_base}/{ood_split}_margin_median"] = float(
                    np.median(ood_margin)
                )

        # Log normalized energy statistics if available
        if "ID_norm_energies" in all_conf:
            id_norm_energies = all_conf["ID_norm_energies"]
            ood_norm_energies = all_conf.get("ood_norm_energies")

            id_max_norm_energy = np.max(id_norm_energies, axis=1)
            log_data[f"{prefix_base}/id_max_norm_energy_mean"] = float(
                np.mean(id_max_norm_energy)
            )
            log_data[f"{prefix_base}/id_max_norm_energy_std"] = float(
                np.std(id_max_norm_energy)
            )
            log_data[f"{prefix_base}/id_max_norm_energy_median"] = float(
                np.median(id_max_norm_energy)
            )

            if ood_norm_energies is not None:
                ood_max_norm_energy = np.max(ood_norm_energies, axis=1)
                log_data[f"{prefix_base}/{ood_split}_max_norm_energy_mean"] = (
                    float(np.mean(ood_max_norm_energy))
                )
                log_data[f"{prefix_base}/{ood_split}_max_norm_energy_std"] = (
                    float(np.std(ood_max_norm_energy))
                )
                log_data[
                    f"{prefix_base}/{ood_split}_max_norm_energy_median"
                ] = float(np.median(ood_max_norm_energy))

        if (
            ood_preds is not None
            and current_task_id is not None
            and self.wandb_logger
        ):
            task_preds = self._map_class_preds_to_task(
                ood_preds, current_task_id
            )
            total_samples = len(task_preds)
            if total_samples > 0:
                counts = np.bincount(task_preds, minlength=current_task_id + 1)
                ratios = counts / total_samples

                dist_prefix = f"Task_Prediction/Distribution/{task_name}"
                if dataset_name:
                    dist_prefix += f"/{dataset_name}"

                log_data[f"{dist_prefix}/Ratio_Predicted_As_Current_Task"] = (
                    ratios[current_task_id]
                )
                log_data[f"{dist_prefix}/Ratio_Predicted_As_Old_Tasks"] = (
                    np.sum(ratios[:current_task_id])
                    if current_task_id > 0
                    else 0.0
                )

        if self.wandb_logger:
            log_wandb_metrics(self.wandb_logger, log_data)

        # Track scores for aggregation by OOD type
        if dataset_name and ood_split:
            agg_key = (task_name, ood_split)
            if agg_key not in self._aggregated_ood_scores:
                self._aggregated_ood_scores[agg_key] = {}

            self._aggregated_ood_scores[agg_key][dataset_name] = {
                "scores": ood_conf,
                "id_scores": all_conf["ID"],
                "mean": ood_conf.mean().item(),
                "std": ood_conf.std().item(),
                "id_mean": all_conf["ID"].mean().item(),
                "id_std": all_conf["ID"].std().item(),
                "id_margin_energy": all_conf.get("ID_margin_energy"),
                "ood_margin_energy": all_conf.get("ood_margin_energy"),
                "id_norm_energies": all_conf.get("ID_norm_energies"),
                "ood_norm_energies": all_conf.get("ood_norm_energies"),
            }

    def log_aggregated_ood_scores(self, current_task_id):
        """
        Logs aggregated OOD statistics grouped by nearood and farood.
        Aggregates across all task combinations (Task-{current_task_id}-ID-*).
        Called after all OOD evaluation is complete for a task.
        Clears the tracking dictionary after logging.
        """
        if not self.wandb_logger:
            keys_to_remove = [
                k
                for k in self._aggregated_ood_scores.keys()
                if k[0].startswith(f"Task-{current_task_id}-ID-")
            ]
            for k in keys_to_remove:
                del self._aggregated_ood_scores[k]
            return

        log_data = {}

        # Group by (id_data_task_id, ood_split) — one aggregate per ID task
        # task_name format: "Task-{current_task_id}-ID-{id_data_task_id}"
        per_id_task_aggregates = {}

        for (
            task_name,
            ood_split,
        ), datasets_dict in list(self._aggregated_ood_scores.items()):
            if not task_name.startswith(f"Task-{current_task_id}-ID-"):
                continue

            if not datasets_dict:
                continue

            id_data_task_id = int(
                task_name[len(f"Task-{current_task_id}-ID-") :]
            )

            agg_key = (id_data_task_id, ood_split)
            if agg_key not in per_id_task_aggregates:
                per_id_task_aggregates[agg_key] = {
                    "ood_scores": [],
                    "id_scores": [],
                    "id_margin_energies": [],
                    "ood_margin_energies": [],
                    "id_norm_energies": [],
                    "ood_norm_energies": [],
                }

            for dataset_name, stats in datasets_dict.items():
                per_id_task_aggregates[agg_key]["ood_scores"].append(
                    stats["scores"]
                )
                per_id_task_aggregates[agg_key]["id_scores"].append(
                    stats["id_scores"]
                )

                if stats.get("id_margin_energy") is not None:
                    per_id_task_aggregates[agg_key][
                        "id_margin_energies"
                    ].append(stats["id_margin_energy"])
                if stats.get("ood_margin_energy") is not None:
                    per_id_task_aggregates[agg_key][
                        "ood_margin_energies"
                    ].append(stats["ood_margin_energy"])
                if stats.get("id_norm_energies") is not None:
                    per_id_task_aggregates[agg_key]["id_norm_energies"].append(
                        stats["id_norm_energies"]
                    )
                if stats.get("ood_norm_energies") is not None:
                    per_id_task_aggregates[agg_key]["ood_norm_energies"].append(
                        stats["ood_norm_energies"]
                    )

        logged_id_stats = set()
        for (
            id_data_task_id,
            ood_split,
        ), score_lists in per_id_task_aggregates.items():
            if score_lists["ood_scores"]:
                aggregated_ood_conf = np.concatenate(score_lists["ood_scores"])
                aggregated_id_conf = np.concatenate(score_lists["id_scores"])

                agg_prefix = (
                    f"OOD_Scores_Aggregated/Task-{current_task_id}"
                    f"/ID-{id_data_task_id}/{ood_split}"
                )
                if id_data_task_id not in logged_id_stats:
                    id_prefix = (
                        f"OOD_Scores_Aggregated/Task-{current_task_id}"
                        f"/ID-{id_data_task_id}"
                    )
                    log_data[f"{id_prefix}/id_mean"] = float(
                        np.mean(aggregated_id_conf)
                    )
                    log_data[f"{id_prefix}/id_std"] = float(
                        np.std(aggregated_id_conf)
                    )
                    logged_id_stats.add(id_data_task_id)
                log_data[f"{agg_prefix}/ood_mean"] = float(
                    np.mean(aggregated_ood_conf)
                )
                log_data[f"{agg_prefix}/ood_std"] = float(
                    np.std(aggregated_ood_conf)
                )
                log_data[f"{agg_prefix}/ood_median"] = float(
                    np.median(aggregated_ood_conf)
                )
                log_data[f"{agg_prefix}/ood_min"] = float(
                    np.min(aggregated_ood_conf)
                )
                log_data[f"{agg_prefix}/ood_max"] = float(
                    np.max(aggregated_ood_conf)
                )
                log_data[f"{agg_prefix}/num_datasets"] = len(
                    score_lists["ood_scores"]
                )

                if score_lists["id_margin_energies"]:
                    aggregated_id_margin = np.concatenate(
                        score_lists["id_margin_energies"]
                    )
                    log_data[f"{agg_prefix}/id_margin_mean"] = float(
                        np.mean(aggregated_id_margin)
                    )
                    log_data[f"{agg_prefix}/id_margin_std"] = float(
                        np.std(aggregated_id_margin)
                    )
                    log_data[f"{agg_prefix}/id_margin_median"] = float(
                        np.median(aggregated_id_margin)
                    )

                if score_lists["ood_margin_energies"]:
                    aggregated_ood_margin = np.concatenate(
                        score_lists["ood_margin_energies"]
                    )
                    log_data[f"{agg_prefix}/{ood_split}_margin_mean"] = float(
                        np.mean(aggregated_ood_margin)
                    )
                    log_data[f"{agg_prefix}/{ood_split}_margin_std"] = float(
                        np.std(aggregated_ood_margin)
                    )
                    log_data[f"{agg_prefix}/{ood_split}_margin_median"] = float(
                        np.median(aggregated_ood_margin)
                    )

                if score_lists["id_norm_energies"]:
                    aggregated_id_norm_energies = np.concatenate(
                        score_lists["id_norm_energies"]
                    )
                    id_max_norm_energy = np.max(
                        aggregated_id_norm_energies, axis=1
                    )
                    log_data[f"{agg_prefix}/id_max_norm_energy_mean"] = float(
                        np.mean(id_max_norm_energy)
                    )
                    log_data[f"{agg_prefix}/id_max_norm_energy_std"] = float(
                        np.std(id_max_norm_energy)
                    )
                    log_data[f"{agg_prefix}/id_max_norm_energy_median"] = float(
                        np.median(id_max_norm_energy)
                    )

                if score_lists["ood_norm_energies"]:
                    aggregated_ood_norm_energies = np.concatenate(
                        score_lists["ood_norm_energies"]
                    )
                    ood_max_norm_energy = np.max(
                        aggregated_ood_norm_energies, axis=1
                    )
                    log_data[
                        f"{agg_prefix}/{ood_split}_max_norm_energy_mean"
                    ] = float(np.mean(ood_max_norm_energy))
                    log_data[
                        f"{agg_prefix}/{ood_split}_max_norm_energy_std"
                    ] = float(np.std(ood_max_norm_energy))
                    log_data[
                        f"{agg_prefix}/{ood_split}_max_norm_energy_median"
                    ] = float(np.median(ood_max_norm_energy))

                print(
                    f"[OOD Aggregation] Task-{current_task_id}/ID-{id_data_task_id}/{ood_split}: "
                    f"mean={np.mean(aggregated_ood_conf):.4f}, "
                    f"std={np.std(aggregated_ood_conf):.4f}, "
                    f"num_datasets={len(score_lists['ood_scores'])}",
                    flush=True,
                )

        if log_data:
            log_wandb_metrics(self.wandb_logger, log_data)

        keys_to_remove = [
            k
            for k in self._aggregated_ood_scores.keys()
            if k[0].startswith(f"Task-{current_task_id}-ID-")
        ]
        for k in keys_to_remove:
            del self._aggregated_ood_scores[k]

    def update_ood_conf_dict(
        self, all_conf, ood_split, dataset_name, ood_conf, ood_gt
    ):
        if ood_split not in all_conf:
            all_conf[ood_split] = {}
            all_conf[ood_split + "_gt"] = {}
        if dataset_name not in all_conf[ood_split]:
            all_conf[ood_split][dataset_name] = {}
            all_conf[ood_split + "_gt"][dataset_name] = {}

        all_conf[ood_split][dataset_name] = ood_conf
        all_conf[ood_split + "_gt"][dataset_name] = ood_gt

        existing_all = all_conf[ood_split].get("ALL", np.array([]))
        all_conf[ood_split]["ALL"] = (
            np.concatenate([existing_all, ood_conf])
            if existing_all.size > 0
            else ood_conf
        )
        existing_all_gt = all_conf[ood_split + "_gt"].get("ALL", np.array([]))
        all_conf[ood_split + "_gt"]["ALL"] = (
            np.concatenate([existing_all_gt, ood_gt])
            if existing_all_gt.size > 0
            else ood_gt
        )

    def eval_ood(
        self,
        dataset_name,
        id_conf,
        current_task_id,
        id_data_task_id,
        ood_split,
        ood_dl,
        task_name,
        all_conf,
        log_wandb_scores=True,
        return_logits=False,
        return_feature=False,
    ):
        try:
            with ModelInferenceContext(self.cl_strategy.model):
                ood_data = self.inference(
                    ood_dl,
                    return_logits=return_logits,
                    return_feature=return_feature,
                )

            if return_logits and return_feature:
                ood_preds, ood_conf, ood_gt, ood_logits, ood_features = ood_data
            elif return_logits:
                ood_preds, ood_conf, ood_gt, ood_logits = ood_data
            elif return_feature:
                ood_preds, ood_conf, ood_gt, ood_features = ood_data
            else:
                ood_preds, ood_conf, ood_gt = ood_data
                ood_logits = None
                ood_features = None

            self.update_ood_conf_dict(
                all_conf, ood_split, dataset_name, ood_conf, ood_gt
            )

            if np.any(np.isnan(ood_conf)) or np.any(np.isinf(ood_conf)):
                print(
                    f"Warning: NaN/Inf in {dataset_name} confidence scores.",
                    flush=True,
                )

        except Exception as e:
            print(f"Error {e} in {dataset_name} dataset, skipping.", flush=True)
            empty = np.array([])
            # Preserve the documented return arity for every requested output
            # combination so callers can handle a failed dataset uniformly.
            result = (empty, empty, empty, [], {})
            if return_logits:
                result += (None,)
            if return_feature:
                result += (None,)
            return result

        if len(ood_conf) < 1000 or len(id_conf) < 1000:
            print(
                f"Warning: Too few samples: {dataset_name} ({len(ood_conf)}), ID ({len(id_conf)})",
                flush=True,
            )

        sids = [
            f"{ood_split}:{dataset_name}:{id_data_task_id}:{i}"
            for i in range(len(ood_conf))
        ]

        metrics, _, _ = self.calculate_and_log_ood_metrics(
            all_conf["ID"],
            all_conf["ID_gt"],
            ood_conf,
            ood_gt,
            task_name,
            dataset_name,
            ood_split,
            threshold=self.thresholds.get(id_data_task_id, None),
        )

        if log_wandb_scores:
            self.log_ood_scores(
                task_name,
                ood_split,
                ood_conf,
                all_conf,
                dataset_name=dataset_name,
                ood_preds=ood_preds,
                current_task_id=current_task_id,
            )

        if return_logits and return_feature:
            return (
                ood_conf,
                ood_gt,
                ood_preds,
                sids,
                metrics,
                ood_logits,
                ood_features,
            )
        if return_logits:
            return ood_conf, ood_gt, ood_preds, sids, metrics, ood_logits
        if return_feature:
            return ood_conf, ood_gt, ood_preds, sids, metrics, ood_features
        return ood_conf, ood_gt, ood_preds, sids, metrics

    def calculate_and_log_ood_metrics(
        self,
        id_conf,
        id_gt,
        ood_conf,
        ood_gt,
        task_name,
        dataset_name,
        ood_split,
        threshold=None,
    ):
        """Computes, logs OOD metrics (AUROC, etc.) and failure modes."""
        conf = np.concatenate([id_conf, ood_conf])
        if ood_gt is None:
            ood_gt = -1 * np.ones_like(ood_conf)
        label = np.concatenate([id_gt, -1 * np.ones_like(ood_gt)])

        metrics_values = compute_all_metrics(
            conf,
            label,
            np.concatenate([np.zeros_like(id_gt), np.zeros_like(ood_gt)]),
        )
        metrics = {
            name: val * 100
            for name, val in zip(OOD_METRIC_List, metrics_values[:-1])
        }

        fp_rate, fn_rate, fp_count, fn_count = log_ood_failure_modes(
            id_conf, id_gt, ood_conf, ood_gt, threshold=threshold
        )

        key_name = f"Metrics/{task_name}/{ood_split}"
        if dataset_name is not None:
            key_name += f"/{dataset_name}"

        log_data = {f"{key_name}/{m}": v for m, v in metrics.items()}
        log_data.update(
            {
                f"{key_name}/failure/fp_rate": fp_rate,
                f"{key_name}/failure/fn_rate": fn_rate,
            }
        )

        if self.wandb_logger:
            log_wandb_metrics(self.wandb_logger, log_data)

        metrics.update(
            {
                "FP_COUNT": fp_count,
                "FN_COUNT": fn_count,
                "FP_RATE": fp_rate * 100,
                "FN_RATE": fn_rate * 100,
                "ERROR_RATE": (fp_rate + fn_rate) * 100,
            }
        )
        return metrics, fp_count, fn_count


# ==============================================================================
# Per-Task Energy Statistics
# ==============================================================================


@dataclass
class TaskEnergyStats:
    """Buffer-derived statistics for one task's energy distribution."""

    mean: float = 0.0
    std: float = 1.0
    median: float = 0.0
    mad: float = 1.0
    n_samples: int = 0


@dataclass
class EnergyStatsStore:
    """All per-task energy statistics + global reference."""

    tasks: dict = field(default_factory=dict)  # task_id -> TaskEnergyStats
    global_ref: TaskEnergyStats = field(default_factory=TaskEnergyStats)

    def update(self, task_id, energies, is_reference=False):
        """Compute and store stats from buffer energies for one task."""
        if len(energies) < 2:
            return

        stats = TaskEnergyStats(
            mean=float(np.mean(energies)),
            std=max(float(np.std(energies)), 1e-6),
            median=float(np.median(energies)),
            mad=max(
                float(median_abs_deviation(energies, scale="normal")), 1e-6
            ),
            n_samples=len(energies),
        )
        self.tasks[task_id] = stats

        if is_reference:
            self.global_ref = TaskEnergyStats(
                mean=stats.mean,
                std=stats.std,
                median=stats.median,
                mad=stats.mad,
                n_samples=stats.n_samples,
            )

    def has(self, task_id):
        return task_id in self.tasks

    def get(self, task_id):
        return self.tasks.get(task_id, TaskEnergyStats())


# ==============================================================================
# Core: Per-Task Energy Computation + Normalization
# ==============================================================================


def compute_per_task_energy(logits, task_class_map, num_tasks):
    """
    Decompose logits into per-task energies.

    Args:
        logits: [N, C] raw model logits
        task_class_map: {task_id: [class_indices]}
        num_tasks: number of tasks seen so far

    Returns:
        [N, T] energy matrix where E[i,t] = logsumexp(logits[i, C_t])
    """
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    N, C = logits_t.shape

    # Build task mask: [T, C] with 0 for valid classes, -inf elsewhere
    mask = torch.full((num_tasks, C), float("-inf"))
    for t in range(num_tasks):
        classes = task_class_map.get(t, [])
        valid = [c for c in classes if c < C]
        if valid:
            mask[t, valid] = 0.0

    # [N, T, C] = [N, 1, C] + [1, T, C]
    masked_logits = logits_t.unsqueeze(1) + mask.unsqueeze(0)

    # [N, T]
    energies = torch.logsumexp(masked_logits, dim=2)
    return energies.numpy()


def normalize_energies(energies, stats_store, method="z_norm"):
    """
    Normalize per-task energies using buffer statistics.

    Normalization is applied before the cross-task maximum. This allows
    different task channels to determine the final ranking for ID and OOD
    samples.

    Args:
        energies: [N, T] raw per-task energies
        stats_store: EnergyStatsStore with buffer statistics
        method: "mean_shift" | "z_norm" | "robust_anchor"

    Returns:
        [N, T] normalized energies
    """
    N, T = energies.shape
    result = energies.copy()
    ref = stats_store.global_ref

    for t in range(T):
        if not stats_store.has(t):
            continue
        s = stats_store.get(t)

        if method == "mean_shift":
            # Shift each task's energy so its buffer mean matches reference
            result[:, t] = energies[:, t] + (ref.mean - s.mean)
        elif method == "zero_mean":
            result[:, t] = energies[:, t] - s.mean
        elif method == "z_norm":
            # Z-normalize then rescale to reference distribution
            z = (energies[:, t] - s.mean) / s.std
            result[:, t] = z * ref.std + ref.mean
        elif method == "robust_anchor":
            # Robust version using median/MAD
            z = (energies[:, t] - s.median) / s.mad
            result[:, t] = z * ref.mad + ref.median
        elif method == "temp_scale":
            # Per-task temperature-scaling baseline.
            # Task-aware *scaling only*: divide each task's energy by its
            # own buffer std (a per-task temperature T_t = s.std), with NO
            # reference re-centering. Isolates whether gains come from the
            # energy decomposition + anchoring or merely from task-aware
            # calibration.
            result[:, t] = energies[:, t] / s.std
        elif method == "none":
            pass

    return result


def compute_tood_score(
    logits,
    task_class_map,
    num_tasks,
    stats_store,
    method="z_norm",
    margin_lambda=0.0,
):
    """
    Compute the complete TOOD score with an optional margin term.

    Base:   S(x) = max_t E_t^norm(x)
    Margin: S(x) = (1 + λ) * E_(1)^norm - λ * E_(2)^norm

    The margin term captures inter-task interaction: ID samples have
    one dominant task (high margin), OOD activates all tasks similarly
    (low margin). With λ > 0, OOD gets penalized for diffuse energy.

    When λ = 0, this reduces to the standard max.

    Args:
        logits: [N, C] raw model logits
        task_class_map: {task_id: [class_indices]}
        num_tasks: number of tasks
        stats_store: EnergyStatsStore
        method: normalization method
        margin_lambda: weight for margin penalty (0 = pure max)

    Returns:
        scores: [N] final OOD scores (higher = more ID)
        task_assignments: [N] which task each sample was assigned to
        norm_energies: [N, T] normalized energies (for diagnostics)
        margin_energy: [N] margin between top-2 task energies
    """
    raw_energies = compute_per_task_energy(logits, task_class_map, num_tasks)
    norm_energies = normalize_energies(raw_energies, stats_store, method)

    task_assignments = norm_energies.argmax(axis=1)
    sorted_energies = np.sort(norm_energies, axis=1)[:, ::-1]
    e_best = sorted_energies[:, 0]
    if sorted_energies.shape[1] < 2:
        # Single task: no second-best channel; margin is undefined (0).
        e_second = e_best
    else:
        e_second = sorted_energies[:, 1]
    margin_energy = e_best - e_second
    if num_tasks < 2 or margin_lambda == 0.0:
        scores = norm_energies.max(axis=1)
    else:
        # S(x) = (1 + λ) * E_(1) - λ * E_(2) = E_(1) + λ * margin
        scores = e_best + margin_lambda * margin_energy

    return scores, task_assignments, norm_energies, margin_energy


# ==============================================================================
# Herding-based exemplar selection
# ==============================================================================


def herding_order(features, k):
    """iCaRL-style greedy herding (Rebuffi et al., 2017).

    Returns the first ``k`` sample indices ordered so that the running mean of
    the selected features best approximates the class mean. We stop after ``k``
    selections because TOOD only keeps a small per-class budget.
    """
    feats = np.asarray(features, dtype=np.float64)
    feats = feats.reshape(len(feats), -1)
    n = len(feats)
    k = min(k, n)
    mu = feats.mean(axis=0)
    selected, mask = [], np.zeros(n, dtype=bool)
    running = np.zeros_like(mu)
    for step in range(k):
        cand = (running[None, :] + feats) / (step + 1)
        dist = np.linalg.norm(mu[None, :] - cand, axis=1)
        dist[mask] = np.inf
        i = int(np.argmin(dist))
        selected.append(i)
        mask[i] = True
        running += feats[i]
    return selected


class HerdingBalancedBuffer:
    """Drop-in for ``ClassBalancedBuffer`` that selects exemplars by herding.

    Exposes ``.buffer`` (an ``AvalancheDataset`` whose ``.targets`` cover all
    stored exemplars) so the existing ``get_class_from_buffer`` helper works
    unchanged. This buffer supports comparisons between herding-based exemplar
    selection and the default random class-balanced calibration buffer.
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self._per_class = {}  # class_id -> (dataset, herding-ordered indices)
        self.buffer = None

    def add_class(self, class_id, dataset, ordered_indices):
        self._per_class[class_id] = (dataset, list(ordered_indices))
        self._rebuild()

    def _rebuild(self):
        n_classes = max(1, len(self._per_class))
        per_class = max(1, self.max_size // n_classes)
        subsets = []
        for ds, order in self._per_class.values():
            sel = order[:per_class]
            if sel:
                subsets.append(AvalancheDataset(ds, indices=sel))
        if subsets:
            self.buffer = concat_datasets(subsets)


# ==============================================================================
# Manager
# ==============================================================================


class CalibratePerTaskOODPostprocessorManager(OODPostprocessorManager):
    """
    Replaces the postprocessor's score with TOOD.

    Instead of calibrating what the postprocessor outputs, we compute
    a fundamentally different score: per-task energy decomposition
    with buffer-based normalization and inter-task margin penalty.

    Score: S(x) = E_(1)^norm + λ * (E_(1)^norm - E_(2)^norm)
           = (1+λ) * E_(1)^norm - λ * E_(2)^norm

    The margin term penalizes samples where multiple tasks have
    similar energy (OOD signature) and rewards samples where one
    task clearly dominates (ID signature). λ=0 recovers pure max.

    Config options:
        calibration_method: "mean_shift" | "z_norm" | "robust_anchor" | "none"
        buffer_max_size: size of replay buffer for statistics
        margin_lambda: weight for margin penalty (0 = pure max, 0.5 = default)
    """

    # TOOD changes the score normalization as new task statistics are added, so
    # historical thresholds must be recomputed in that evolving score space.
    reuse_peak_thresholds = False

    METHODS = {
        "none",
        "mean_shift",
        "z_norm",
        "robust_anchor",
        "zero_mean",
        "temp_scale",
    }
    ANCHORS = {"recent", "oldest", "max_spread", "mean_all"}

    def __init__(
        self,
        config,
        cl_strategy,
        benchmark,
        ood_loader_dict,
        wandb_logger,
        buffer_max_size=20,
        calibration_method="z_norm",
        margin_lambda=0.5,
        anchor="recent",
        buffer_selection="random",
        task_split_factor=1,
        **kwargs,
    ):
        super().__init__(
            config,
            cl_strategy,
            benchmark,
            ood_loader_dict,
            wandb_logger,
            buffer_max_size=buffer_max_size,
        )
        assert (
            calibration_method in self.METHODS
        ), f"Unknown method: {calibration_method}. Choose from {self.METHODS}"
        assert (
            anchor in self.ANCHORS
        ), f"Unknown anchor: {anchor}. Choose from {self.ANCHORS}"
        assert buffer_selection in {"random", "herding"}, (
            f"Unknown buffer_selection: {buffer_selection}. "
            "Choose from {'random', 'herding'}"
        )

        self.calibration_method = calibration_method
        self.margin_lambda = margin_lambda
        self.anchor = anchor  # Task used as the normalization reference.
        self.buffer_selection = buffer_selection  # "random" or "herding".
        # Values greater than one split each task into finer score partitions.
        self.task_split_factor = max(1, int(task_split_factor))
        self._buffer_max_size = buffer_max_size
        if self.buffer_selection == "herding":
            # Replace the random class-balanced buffer with a herding buffer.
            self.buffer = HerdingBalancedBuffer(max_size=buffer_max_size)
        self.stats = EnergyStatsStore()
        self._task_class_map = {}
        self._stats_computed = set()
        print(
            f"[TOOD] anchor={self.anchor}, buffer_selection={self.buffer_selection}, "
            f"task_split_factor={self.task_split_factor}"
        )
        # self.buffer and self._buffer_updated are inherited from parent

    # ------------------------------------------------------------------
    # Task class mapping
    # ------------------------------------------------------------------

    def _ensure_buffer_updated(self, current_task_id):
        """Populate the calibration buffer for ``current_task_id``.

        Random selection delegates to the parent ``ClassBalancedBuffer``.
        Herding extracts penultimate features for the task's training data and
        greedily selects representative exemplars for each class.
        """
        if self.buffer_selection != "herding":
            return super()._ensure_buffer_updated(current_task_id)

        if current_task_id in self._buffer_updated:
            return

        dataset = self.benchmark.train_stream[current_task_id].dataset
        loader = get_dataloader(
            dataset,
            batch_size=self.config.strategy.eval_mb_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )
        with ModelInferenceContext(self.cl_strategy.model):
            out = self.inference(
                loader, return_logits=True, return_feature=True
            )
        # (preds, conf, gt, logits, features)
        gts = np.asarray(out[2]).reshape(-1)
        feats = out[4]
        feats = (
            feats.detach().cpu().numpy()
            if isinstance(feats, torch.Tensor)
            else np.asarray(feats)
        )
        feats = feats.reshape(len(gts), -1)

        n_classes_seen = sum(
            len(self.benchmark.train_stream[t].classes_in_this_experience)
            for t in range(current_task_id + 1)
        )
        per_class = max(1, self._buffer_max_size // max(1, n_classes_seen))

        classes = self.benchmark.train_stream[
            current_task_id
        ].classes_in_this_experience
        for c in classes:
            idx = np.where(gts == c)[0]
            if len(idx) == 0:
                continue
            order_local = herding_order(feats[idx], per_class)
            ds_indices = [int(idx[j]) for j in order_local]
            self.buffer.add_class(c, dataset, ds_indices)

        self._buffer_updated.add(current_task_id)

    def _get_task_class_map(self, current_task_id):
        for t in range(current_task_id + 1):
            if t not in self._task_class_map:
                self._task_class_map[t] = list(
                    self.benchmark.train_stream[t].classes_in_this_experience
                )
        return self._task_class_map

    def _build_pseudo_map(self, current_task_id):
        """Build the per-(pseudo-)task class map used for scoring.

        When ``task_split_factor == 1`` (default) this is the true task ->
        classes map and behaviour is identical to the original TOOD. When it
        is > 1 each real task's classes are split into ``factor`` near-equal
        chunks, producing more energy channels than there are real tasks. This
        simulates overestimating the number of tasks at deployment: TOOD
        decomposes logits into more partitions than truly exist, without
        access to the real boundaries.

        Returns:
            pseudo_map: {pseudo_id: [class_indices]}
            pseudo_to_real: {pseudo_id: real_task_id}
            num_pseudo: number of pseudo-tasks (energy channels)
        """
        real_map = self._get_task_class_map(current_task_id)
        factor = self.task_split_factor
        if factor == 1:
            n = current_task_id + 1
            pseudo_map = {t: list(real_map[t]) for t in range(n)}
            pseudo_to_real = {t: t for t in range(n)}
            return pseudo_map, pseudo_to_real, n

        pseudo_map, pseudo_to_real, p = {}, {}, 0
        for t in range(current_task_id + 1):
            classes = list(real_map[t])
            # stride split keeps chunks balanced and non-empty when possible
            chunks = [classes[i::factor] for i in range(factor)]
            chunks = [c for c in chunks if c]
            for c in chunks:
                pseudo_map[p] = c
                pseudo_to_real[p] = t
                p += 1
        return pseudo_map, pseudo_to_real, p

    def _set_reference(self, num_tasks):
        """Select the calibration reference statistics.

        The reference (``global_ref``) is what every task's energy channel is
        re-centered/-scaled toward in :func:`normalize_energies`.
        """
        mode = self.anchor
        present = [t for t in range(num_tasks) if self.stats.has(t)]
        if not present:
            return

        if mode == "mean_all":
            ts = [self.stats.get(t) for t in present]
            self.stats.global_ref = TaskEnergyStats(
                mean=float(np.mean([s.mean for s in ts])),
                std=max(float(np.mean([s.std for s in ts])), 1e-6),
                median=float(np.mean([s.median for s in ts])),
                mad=max(float(np.mean([s.mad for s in ts])), 1e-6),
                n_samples=int(np.sum([s.n_samples for s in ts])),
            )
            ref_id = None
        elif mode == "oldest":
            ref_id = present[0]
        elif mode == "max_spread":
            # task whose ID energy distribution has the largest spread
            ref_id = max(present, key=lambda t: self.stats.get(t).std)
        else:  # "recent" (default): most recent (pseudo-)task
            ref_id = present[-1]

        if ref_id is not None:
            s = self.stats.get(ref_id)
            self.stats.global_ref = TaskEnergyStats(
                mean=s.mean,
                std=s.std,
                median=s.median,
                mad=s.mad,
                n_samples=s.n_samples,
            )
        print(
            f"[TOOD Stats] anchor='{mode}' -> reference "
            f"mean={self.stats.global_ref.mean:.4f}, "
            f"std={self.stats.global_ref.std:.4f}"
        )

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _make_buffer_loader(self, task_id):
        classes = self.benchmark.train_stream[
            task_id
        ].classes_in_this_experience
        datasets = [get_class_from_buffer(self.buffer, c) for c in classes]
        datasets = [d for d in datasets if d is not None]
        if not datasets:
            return None
        combined = ConcatDataset(datasets)
        if len(combined) < 5:
            return None
        return get_dataloader(
            combined,
            batch_size=self.config.strategy.eval_mb_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

    # ------------------------------------------------------------------
    # Statistics computation
    # ------------------------------------------------------------------

    def _compute_energy_stats(self, current_task_id):
        """
        Run buffer through model, compute per-task energy for each
        task's buffer samples, store statistics.

        The reference task is current_task_id (least drifted).
        """
        if current_task_id in self._stats_computed:
            return

        pseudo_map, pseudo_to_real, num_pseudo = self._build_pseudo_map(
            current_task_id
        )

        for real_t in range(current_task_id + 1):
            loader = self._make_buffer_loader(real_t)
            if loader is None:
                continue

            with ModelInferenceContext(self.cl_strategy.model):
                _, _, _, logits = self.inference(loader, return_logits=True)

            # Per-(pseudo-)task energy for this real task's buffer subset
            all_energies = compute_per_task_energy(
                logits, pseudo_map, num_pseudo
            )

            # Store stats for every pseudo-task that belongs to this real
            # task, evaluated on its OWN energy channel.
            for p in range(num_pseudo):
                if pseudo_to_real[p] != real_t:
                    continue
                self.stats.update(p, all_energies[:, p], is_reference=False)

        # Reference / anchor selection happens once all channels are known.
        self._set_reference(num_pseudo)

        self._stats_computed.add(current_task_id)
        self._log_stats(current_task_id)

    def _log_stats(self, current_task_id):
        """Print and log energy statistics."""
        log_data = {}
        for t in range(current_task_id + 1):
            if self.stats.has(t):
                s = self.stats.get(t)
                print(
                    f"[TOOD Stats] Task {t}: energy_mean={s.mean:.4f}, "
                    f"energy_std={s.std:.4f}, energy_median={s.median:.4f}, "
                    f"n={s.n_samples}"
                )
                log_data[f"TOOD/Task_{t}/energy_mean"] = s.mean
                log_data[f"TOOD/Task_{t}/energy_std"] = s.std

        if (
            self.stats.has(0)
            and self.stats.has(current_task_id)
            and current_task_id > 0
        ):
            gap = self.stats.get(current_task_id).mean - self.stats.get(0).mean
            print(
                f"[TOOD Stats] Energy Gap (T{current_task_id} - T0): {gap:.4f}"
            )
            log_data["TOOD/energy_gap"] = gap

        ref = self.stats.global_ref
        print(
            f"[TOOD Stats] Reference (T{current_task_id}): "
            f"mean={ref.mean:.4f}, std={ref.std:.4f}, "
            f"margin_lambda={self.margin_lambda}"
        )

        if self.wandb_logger and log_data:
            log_wandb_metrics(self.wandb_logger, log_data)

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _score(self, logits, current_task_id, is_id_data=True):
        """
        Compute the TOOD score with an optional margin term.

        Returns:
            scores: [N] higher = more likely ID
            task_assignments: [N] predicted task for each sample
            norm_energies: [N, T] normalized energies per task
            margin_energy: [N] energy margins (difference between top-2 tasks)
        """
        pseudo_map, _, num_tasks = self._build_pseudo_map(current_task_id)

        scores, assignments, norm_energies, margin_energy = (
            compute_tood_score(
                logits,
                pseudo_map,
                num_tasks,
                self.stats,
                self.calibration_method,
                margin_lambda=self.margin_lambda,
            )
        )

        # Diagnostics: task distribution + margin stats
        for t in range(num_tasks):
            frac = (assignments == t).mean()
            if frac > 0.01:
                print(f"  Task {t}: {frac*100:.1f}% of samples", end="")
        print()

        if num_tasks >= 2 and self.margin_lambda > 0:
            sorted_e = np.sort(norm_energies, axis=1)[:, ::-1]
            margins = sorted_e[:, 0] - sorted_e[:, 1]
            print(
                f"  Margin stats: mean={margins.mean():.3f}, "
                f"median={np.median(margins):.3f}, "
                f"p10={np.percentile(margins, 10):.3f}, "
                f"p90={np.percentile(margins, 90):.3f}"
            )

        return scores, assignments, norm_energies, margin_energy

    # ------------------------------------------------------------------
    # Prepare (called before evaluation at each task step)
    # ------------------------------------------------------------------

    def _prepare(self, current_task_id, id_data_task_id):
        self._ensure_buffer_updated(current_task_id)
        if current_task_id > 0:
            return_task_id = self.config.scenario.return_task_id
            with ModelInferenceContext(
                self.cl_strategy.model,
                return_task_id=return_task_id,
                task_id=id_data_task_id,
            ):
                self._compute_energy_stats(current_task_id)

    # ------------------------------------------------------------------
    # Evaluation overrides
    # ------------------------------------------------------------------

    def eval_for_id_task(
        self,
        id_train_loader,
        id_test_loader,
        current_task_id,
        id_data_task_id,
        return_logits=False,
        all_conf=None,
        skip_train_inference=False,
    ):
        if all_conf is None:
            all_conf = {}

        self._prepare(current_task_id, id_data_task_id)

        # The base manager's training confidence is discarded below because
        # TOOD needs logits to rescore it. Avoid doing the same full training
        # pass once for base confidence and again for logits.
        use_tood = (
            current_task_id > 0 and self.calibration_method != "none"
        )
        if (
            use_tood
            and current_task_id == id_data_task_id
            and not skip_train_inference
        ):
            self.analyze_bias_vs_distance(current_task_id)

        # Get base results with logits
        res = super().eval_for_id_task(
            id_train_loader,
            id_test_loader,
            current_task_id,
            id_data_task_id,
            return_logits=True,
            all_conf=all_conf,
            skip_train_inference=skip_train_inference or use_tood,
        )
        sids, threshold, id_conf, train_conf, id_preds, id_logits = res

        # Replace scores with TOOD
        if use_tood:
            print(f"[TOOD] Scoring ID test data (Task {id_data_task_id})...")
            id_conf, id_assignments, id_norm_energies, id_margin = self._score(
                id_logits, current_task_id, is_id_data=True
            )
            all_conf["ID"] = id_conf
            all_conf["ID_norm_energies"] = id_norm_energies
            all_conf["ID_margin_energy"] = id_margin

            if not skip_train_inference and id_train_loader is not None:
                # Re-score training data for threshold
                with ModelInferenceContext(self.cl_strategy.model):
                    _, _, _, train_logits = self.inference(
                        id_train_loader, return_logits=True
                    )
                print("[TOOD] Scoring ID train data...")
                train_conf, _, train_norm_energies, train_margin = self._score(
                    train_logits, current_task_id, is_id_data=True
                )
            else:
                # Use test scores for threshold when train inference is skipped
                train_conf = id_conf

            # Recompute threshold on new scores
            threshold = self._get_or_cache_threshold(
                train_conf,
                current_task_id,
                id_data_task_id,
                invalidate_cache=True,
            )

        if return_logits:
            return sids, threshold, id_conf, train_conf, id_preds, id_logits
        return sids, threshold, id_conf, train_conf, id_preds

    def eval_ood(
        self,
        dataset_name,
        id_conf,
        current_task_id,
        id_data_task_id,
        ood_split,
        ood_dl,
        task_name,
        all_conf,
    ):
        # Get base OOD results with logits
        res = super().eval_ood(
            dataset_name,
            id_conf,
            current_task_id,
            id_data_task_id,
            ood_split,
            ood_dl,
            task_name,
            copy(all_conf),
            log_wandb_scores=False,
            return_logits=True,
        )
        ood_conf, ood_gt, ood_preds, sids, metrics, ood_logits = res

        # Replace OOD scores with TOOD
        if current_task_id > 0 and self.calibration_method != "none":
            print(f"[TOOD] Scoring OOD data ({dataset_name})...")
            ood_conf, ood_assignments, ood_norm_energies, ood_margin = (
                self._score(ood_logits, current_task_id, is_id_data=False)
            )
            all_conf["ood_norm_energies"] = ood_norm_energies
            all_conf["ood_margin_energy"] = ood_margin

            # Recompute metrics with new scores
            metrics, _, _ = self.calculate_and_log_ood_metrics(
                all_conf["ID"],
                all_conf["ID_gt"],
                ood_conf,
                ood_gt,
                task_name,
                dataset_name,
                ood_split,
                threshold=self.thresholds.get(id_data_task_id, None),
            )

        self.log_ood_scores(
            task_name,
            ood_split,
            ood_conf,
            all_conf,
            dataset_name=dataset_name,
            ood_preds=ood_preds,
            current_task_id=current_task_id,
        )
        self.update_ood_conf_dict(
            all_conf, ood_split, dataset_name, ood_conf, ood_gt
        )

        return ood_conf, ood_gt, ood_preds, sids, metrics


def get_ood_manager(
    config, cl_strategy, benchmark, ood_loader_dict, wandb_logger
):
    """Factory that instantiates the right OOD manager based on config.

    Reads ``calibrate_ood_scores`` from the CL experiment config:

        calibrate_ood_scores:
          enabled: False
          method: mean_shift   # mean_shift | z_norm | robust_anchor | zero_mean | none
          buffer_size: 20
          margin_lambda: 0.5
    """
    cal = config.calibrate_ood_scores
    buffer_size = int(cal.buffer_size) if cal and cal.buffer_size else 20

    if cal and cal.enabled:
        method = cal.method or "mean_shift"
        margin_lambda = (
            float(cal.margin_lambda) if cal.margin_lambda is not None else 0.5
        )
        anchor = cal.anchor or "recent"
        buffer_selection = cal.buffer_selection or "random"
        task_split_factor = (
            int(cal.task_split_factor) if cal.task_split_factor else 1
        )
        print(
            f"[OOD Manager] TOOD calibration enabled: method={method}, "
            f"margin_lambda={margin_lambda}, anchor={anchor}, "
            f"buffer_selection={buffer_selection}, "
            f"task_split_factor={task_split_factor}"
        )
        return CalibratePerTaskOODPostprocessorManager(
            config,
            cl_strategy,
            benchmark,
            ood_loader_dict,
            wandb_logger,
            buffer_max_size=buffer_size,
            calibration_method=method,
            margin_lambda=margin_lambda,
            anchor=anchor,
            buffer_selection=buffer_selection,
            task_split_factor=task_split_factor,
        )
    return OODPostprocessorManager(
        config,
        cl_strategy,
        benchmark,
        ood_loader_dict,
        wandb_logger,
        buffer_max_size=buffer_size,
    )
