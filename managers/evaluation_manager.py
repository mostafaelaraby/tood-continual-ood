# Minimal reproduction surface: only CKA + AUROC-deterioration paths are retained.
import numpy as np
import torch.backends.cudnn as cudnn
from avalanche.training.utils import freeze_everything

from utils import OOD_METRIC_List, OOD_TYPES
from utils.helpers import (
    RunningMetric,
    collate_fn_ood,
    garbage_collect,
    get_dataloader,
    get_task_dataloaders,
    log_wandb_metrics,
    set_task_id,
)


class EvaluationManager:
    """
    Runs CL evaluation + OOD evaluation loop against past tasks and OOD datasets.
    """

    def __init__(
        self,
        config,
        strategy,
        benchmark,
        ood_manager,
        recorder_manager,
        ood_loader_dict,
        eval_transform,
        wandb_logger=None,
    ):
        self.config = config
        self.strategy = strategy
        self.benchmark = benchmark
        self.ood_manager = ood_manager
        self.recorder = recorder_manager
        self.ood_loader_dict = ood_loader_dict
        self.eval_transform = eval_transform
        self.wandb_logger = wandb_logger
        self._ood_dl_cache = {}
        self._id_dl_cache = {}
        self._cka_dl_cache = {}

    def evaluate_task(self, current_task_id, n_tasks):
        set_task_id(self.strategy.model, current_task_id)
        self.strategy.model.eval()
        freeze_everything(self.strategy.model)
        self.strategy.model.return_combined_logits = False
        assert not self.strategy.model.training

        previous_cudnn_benchmark = cudnn.benchmark
        cudnn.benchmark = True
        try:
            print("\nEvaluating CL performance on all seen tasks...")
            cl_metrics = self.strategy.eval(
                self.benchmark.test_stream[: current_task_id + 1]
            )
            self.recorder.log_cl_metrics(cl_metrics)

            self._update_cka_metrics(current_task_id)

            print("\n--- Starting OOD Evaluation Loop ---")
            self._run_ood_evaluation_loop(current_task_id, n_tasks)
        finally:
            cudnn.benchmark = previous_cudnn_benchmark
            # Keep PyTorch's CUDA allocator warm between checkpoints. Returning
            # every cached block to the driver forces the next task to repeat
            # expensive allocations and does not reduce live tensor memory.
            garbage_collect(release_cuda_cache=False)

    def _update_cka_metrics(self, current_task_id):
        """Record CKA for each past ID task against the current model state."""
        max_batches_percentage = 0.5
        for task_id in range(current_task_id + 1):
            set_task_id(self.strategy.model, task_id)
            if task_id not in self._cka_dl_cache:
                _, self._cka_dl_cache[task_id] = get_task_dataloaders(
                    self.config,
                    self.benchmark.train_stream,
                    self.benchmark.test_stream,
                    task_id,
                    include_prev_tasks=False,
                    shuffle_train=False,
                    shuffle_test=False,
                    balanced_sampling=True,
                    return_task_id=False,
                    test_only=True,
                )
            id_test_loader = self._cka_dl_cache[task_id]
            max_batches = int(len(id_test_loader) * max_batches_percentage)
            self.recorder.update_cka(
                self.strategy.model,
                id_test_loader,
                data_task_id=task_id,
                current_task_id=current_task_id,
                max_batches=max_batches,
            )

    def _compute_pooled_metrics(
        self, current_task_id, pooled_id_conf, pooled_id_gt, all_conf_scores
    ):
        if not pooled_id_conf:
            return

        flat_id_conf = np.concatenate(pooled_id_conf)
        flat_id_gt = np.concatenate(pooled_id_gt)
        task_name_pooled = f"Task-{current_task_id}-Pooled"

        for ood_type in OOD_TYPES:
            if ood_type not in all_conf_scores:
                continue

            pooled_rm = RunningMetric()
            per_dataset_pooled = {}

            for ds_name in self.ood_loader_dict[ood_type]:
                if ds_name not in all_conf_scores[ood_type]:
                    continue
                gt_key = ood_type + "_gt"
                ood_scores = all_conf_scores[ood_type][ds_name]

                metrics, _, _ = self.ood_manager.calculate_and_log_ood_metrics(
                    flat_id_conf,
                    flat_id_gt,
                    ood_scores,
                    all_conf_scores[gt_key][ds_name],
                    task_name=task_name_pooled,
                    dataset_name=ds_name,
                    ood_split=ood_type,
                )

                per_dataset_pooled[ds_name] = {
                    m: metrics.get(m, 0.0) for m in OOD_METRIC_List
                }
                for metric in OOD_METRIC_List:
                    pooled_rm.update(ood_type, metric, metrics.get(metric, 0))

            pooled_avg = {
                m: pooled_rm.mean(ood_type, m) for m in OOD_METRIC_List
            }
            self.recorder.store_pooled_results(
                current_task_id,
                ood_type,
                pooled_avg,
                per_dataset_metrics=per_dataset_pooled,
            )

    def _run_ood_evaluation_loop(self, current_task_id, n_tasks):
        id_train_loader, id_test_loader = get_task_dataloaders(
            self.config,
            self.benchmark.train_stream,
            self.benchmark.test_stream,
            current_task_id,
            shuffle_train=False,
            include_prev_tasks=True,
            return_task_id=self.config.scenario.return_task_id,
        )

        n_classes = sum(
            len(t.classes_in_this_experience)
            for t in self.benchmark.train_stream[: current_task_id + 1]
        )
        print(
            f"Setting postprocessor for {n_classes} classes at task id {current_task_id}."
        )
        self.ood_manager.setup(id_train_loader, id_test_loader, n_classes)

        all_conf_scores = {}
        pooled_id_conf = []
        pooled_id_gt = []

        print(
            f"Pre-creating train+test dataloaders for {current_task_id + 1} ID tasks..."
        )
        for _idx in range(current_task_id + 1):
            if _idx not in self._id_dl_cache:
                self._id_dl_cache[_idx] = get_task_dataloaders(
                    self.config,
                    self.benchmark.train_stream,
                    self.benchmark.test_stream,
                    _idx,
                    shuffle_train=False,
                    include_prev_tasks=False,
                    return_task_id=self.config.scenario.return_task_id,
                    test_only=False,
                )

        for id_task_idx in range(current_task_id + 1):
            id_task_train_loader, id_task_test_loader = self._id_dl_cache[
                id_task_idx
            ]

            # A historical task's threshold was calibrated on its training
            # data when that task was first evaluated. Re-running the full
            # training set at every later checkpoint is redundant. If an
            # evaluation resumes without that in-memory threshold, retain the
            # old behavior once to reconstruct it.
            skip_train_inference = (
                id_task_idx < current_task_id
                and self.ood_manager.reuse_peak_thresholds
                and id_task_idx in self.ood_manager.thresholds
            )
            _, thr, id_conf, _, _ = self.ood_manager.eval_for_id_task(
                id_task_train_loader,
                id_task_test_loader,
                current_task_id,
                id_task_idx,
                all_conf=all_conf_scores,
                skip_train_inference=skip_train_inference,
            )

            is_last_id = id_task_idx == current_task_id

            for ood_type in OOD_TYPES:
                avg_metrics = self._eval_pair(
                    ood_type,
                    current_task_id,
                    id_task_idx,
                    thr,
                    all_conf_scores,
                    record_ood=is_last_id,
                )
                if id_task_idx == 0:
                    for metric in avg_metrics:
                        log_wandb_metrics(
                            self.wandb_logger,
                            {
                                f"Motivation_Plot_C/Task_0_over_time/{ood_type}/{metric}": avg_metrics[
                                    metric
                                ]
                            },
                        )
                if (
                    current_task_id == n_tasks - 1 or current_task_id == 0
                ) and id_task_idx == 0:
                    suffix = "initial" if current_task_id == 0 else "final"
                    for metric in avg_metrics:
                        log_wandb_metrics(
                            self.wandb_logger,
                            {
                                f"task_0_{suffix}/{ood_type}/{metric}": avg_metrics[
                                    metric
                                ]
                            },
                        )

            if (
                "ID_gt" in all_conf_scores
                and id_conf is not None
                and len(id_conf) > 0
            ):
                pooled_id_conf.append(id_conf)
                pooled_id_gt.append(all_conf_scores["ID_gt"])

        self._compute_pooled_metrics(
            current_task_id, pooled_id_conf, pooled_id_gt, all_conf_scores
        )
        self.recorder.log_ood_summary(current_task_id)

    def _eval_pair(
        self,
        ood_type,
        current_task_id,
        id_data_task_id,
        thr,
        all_conf,
        record_ood=True,
    ):
        rm = RunningMetric()
        cka_recorded = False
        for dataset_name, ood_dl in self.ood_loader_dict[ood_type].items():
            print(f"Evaluating OOD dataset: {dataset_name}...", flush=True)
            ood_dl = self._prepare_ood_dl(ood_dl)
            task_name = f"Task-{current_task_id}-ID-{id_data_task_id}"

            if ood_type in all_conf and dataset_name in all_conf[ood_type]:
                print(f"Using cached OOD confidences for {dataset_name}.")
                ood_conf = all_conf[ood_type][dataset_name]
                ood_gt = all_conf[ood_type + "_gt"][dataset_name]
                metrics, _ = all_conf[ood_type]["dataset_metric"][dataset_name]
                sids = [
                    f"{ood_type}:{dataset_name}:{id_data_task_id}:{i}"
                    for i in range(len(ood_conf))
                ]
                metrics, _, _ = self.ood_manager.calculate_and_log_ood_metrics(
                    all_conf["ID"],
                    all_conf["ID_gt"],
                    ood_conf,
                    ood_gt,
                    task_name,
                    dataset_name,
                    ood_type,
                    threshold=thr,
                )
                all_conf[ood_type]["dataset_metric"][dataset_name] = (
                    metrics,
                    sids,
                )
            else:
                ood_conf, ood_gt, _, sids, metrics = self.ood_manager.eval_ood(
                    dataset_name,
                    all_conf["ID"],
                    current_task_id,
                    id_data_task_id,
                    ood_type,
                    ood_dl,
                    task_name,
                    all_conf,
                )
                if ood_type not in all_conf:
                    all_conf[ood_type] = {}
                if "dataset_metric" not in all_conf[ood_type]:
                    all_conf[ood_type]["dataset_metric"] = {}
                all_conf[ood_type]["dataset_metric"][dataset_name] = (
                    metrics,
                    sids,
                )
            for metric in OOD_METRIC_List:
                val = metrics.get(metric, 0)
                rm.update(ood_type, metric, val)
            # CKA references are keyed by OOD split, not dataset. Recording
            # every dataset under the same key both repeats feature extraction
            # and compares unrelated datasets. Use the first successful
            # dataset as the stable representative for this split.
            if record_ood and not cka_recorded and len(ood_conf) > 0:
                self.recorder.update_cka_ood(
                    self.strategy.model, ood_dl, current_task_id, ood_type
                )
                cka_recorded = True

        self.recorder.log_pair_metrics(
            current_task_id, id_data_task_id, ood_type, all_conf
        )
        avg_metrics = {m: rm.mean(ood_type, m) for m in OOD_METRIC_List}

        per_dataset_metrics = {}
        if ood_type in all_conf and "dataset_metric" in all_conf[ood_type]:
            for ds_name, (ds_metrics, _) in all_conf[ood_type][
                "dataset_metric"
            ].items():
                per_dataset_metrics[ds_name] = {
                    m: ds_metrics.get(m, 0.0) for m in OOD_METRIC_List
                }

        self.recorder.store_task_ood_results(
            current_task_id,
            id_data_task_id,
            ood_type,
            avg_metrics,
            per_dataset_metrics=per_dataset_metrics,
        )
        return avg_metrics

    def _prepare_ood_dl(self, ood_dl):
        cache_key = id(ood_dl.dataset)
        if cache_key in self._ood_dl_cache:
            return self._ood_dl_cache[cache_key]

        ood_dataset = ood_dl.dataset
        ood_dataset.transform_image = self.eval_transform

        prepared = get_dataloader(
            ood_dataset,
            batch_size=self.config.strategy.eval_mb_size,
            num_workers=4,
            collate_fn=collate_fn_ood,
            shuffle=False,
        )
        self._ood_dl_cache[cache_key] = prepared
        return prepared
