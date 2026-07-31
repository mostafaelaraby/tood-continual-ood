# Minimal reproduction surface: only CKA + AUROC-deterioration paths are retained.
import hashlib
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.plotting import (
    plot_auroc_performance_per_task,
    plot_cka_trajectory,
    plot_cka_vs_auroc,
)
from analysis.recorders import CKARecorder
from utils import OOD_METRIC_List, OOD_TYPES
from utils.helpers import (
    RunningMetric,
    compute_true_stream_forgetting,
    log_wandb_metrics,
)


class RecorderManager:
    """
    Manages CKA recording, OOD result storage, and post-experiment analysis
    for AUROC deterioration + CKA trajectory.
    """

    def __init__(self, config, benchmark, wandb_logger):
        self.config = config
        self.benchmark = benchmark
        self.wandb_logger = wandb_logger

        self.recorders = {"cka": CKARecorder()}

        self.cl_results: List[Dict] = []
        self.ood_results: Dict[int, Dict] = {}
        self.pooled_results: Dict[int, Dict] = {}
        self.avg_inc_acc_scores: List[float] = []
        self.auroc_df = pd.DataFrame()

    @staticmethod
    def _calibration_signature(config):
        cal = getattr(config, "calibrate_ood_scores", None)
        if not cal or not getattr(cal, "enabled", False):
            return "calibration_disabled"

        method = str(getattr(cal, "method", "none") or "none")
        buffer_size = str(getattr(cal, "buffer_size", "default"))
        margin_lambda = str(getattr(cal, "margin_lambda", "default"))
        return (
            f"calibration_{method}"
            f"_buffer_{buffer_size}"
            f"_margin_{margin_lambda}"
        )

    @property
    def output_dir(self):
        path = os.path.join(
            self.config.scenario.ckpt_dir,
            self.config.postprocessor.name,
            self._calibration_signature(self.config),
        )
        os.makedirs(path, exist_ok=True)
        return path

    # =========================================================================
    #                          EVAL STATE SAVE / RESUME
    # =========================================================================

    @staticmethod
    def _config_fingerprint(config):
        key_fields = (
            str(config.scenario.n_experiences),
            str(config.scenario.seed),
            str(config.dataset.name),
            str(config.postprocessor.name),
            str(config.strategy.name),
            RecorderManager._calibration_signature(config),
        )
        raw = "|".join(key_fields)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def save_eval_state(self, last_completed_task_id):
        state = {
            "last_completed_task_id": last_completed_task_id,
            "cl_results": self.cl_results,
            "ood_results": self.ood_results,
            "pooled_results": self.pooled_results,
            "recorders": self.recorders,
            "avg_inc_acc_scores": self.avg_inc_acc_scores,
            "config_fingerprint": self._config_fingerprint(self.config),
        }
        state_path = os.path.join(self.output_dir, "eval_state.pkl")
        tmp_path = state_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, state_path)
        print(
            f"Eval state saved: task {last_completed_task_id} -> {state_path}"
        )

    def maybe_resume_eval_state(self):
        resume_eval = getattr(self.config.scenario, "resume_eval", False)
        state_path = os.path.join(self.output_dir, "eval_state.pkl")

        if not resume_eval:
            if os.path.exists(state_path):
                os.remove(state_path)
                print(
                    f"resume_eval=False: removed stale eval state at {state_path}"
                )
            return 0

        if not os.path.exists(state_path):
            print(
                "resume_eval=True but no eval_state.pkl found. Starting fresh."
            )
            return 0

        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load eval state ({e}). Starting fresh.")
            return 0

        saved_fp = state.get("config_fingerprint", "")
        current_fp = self._config_fingerprint(self.config)
        if saved_fp != current_fp:
            print(
                f"WARNING: Eval state fingerprint mismatch "
                f"(saved={saved_fp}, current={current_fp}). Starting fresh."
            )
            os.remove(state_path)
            return 0

        last_completed = state["last_completed_task_id"]
        self.cl_results = state["cl_results"]
        self.ood_results = state["ood_results"]
        self.pooled_results = state["pooled_results"]
        self.recorders = state["recorders"]
        self.avg_inc_acc_scores = state.get("avg_inc_acc_scores", [])

        start_from = last_completed + 1
        print(
            f"Resumed eval state: tasks 0..{last_completed} completed. "
            f"Resuming from task {start_from}."
        )
        self._relog_completed_metrics(last_completed)
        return start_from

    def restore_avg_inc_accuracy(self, avg_inc_acc_plugin):
        if not self.avg_inc_acc_scores:
            return
        avg_inc_acc_plugin.a_b_scores = list(self.avg_inc_acc_scores)
        print(
            f"Restored AverageIncrementalAccuracy with "
            f"{len(avg_inc_acc_plugin.a_b_scores)} entries."
        )

    def _relog_completed_metrics(self, last_completed_task_id):
        if not self.wandb_logger:
            return
        print("Re-logging WandB metrics for completed tasks...")
        for tid in range(last_completed_task_id + 1):
            task_name = f"Task-{tid}"
            wandb_dict = {}

            if tid < len(self.cl_results):
                for key, val in self.cl_results[tid].items():
                    if key.startswith("Top1_Acc_Stream/"):
                        wandb_dict[key] = val

            all_tasks = self.ood_results.get(tid, {}).get("ALL_TASKS", {})
            for ood_type, metrics in all_tasks.items():
                for m, val in metrics.items():
                    if val is not None and not np.isnan(val):
                        wandb_dict[f"{task_name}/{ood_type}/{m}_Last"] = val

            for ood_type in OOD_TYPES:
                step_values = {m: [] for m in OOD_METRIC_List}
                for i in range(tid + 1):
                    prev = (
                        self.ood_results.get(i, {})
                        .get("ALL_TASKS", {})
                        .get(ood_type, {})
                    )
                    for m in OOD_METRIC_List:
                        val = prev.get(m, None)
                        if val is not None and not np.isnan(val):
                            step_values[m].append(val)
                for m in OOD_METRIC_List:
                    if step_values[m]:
                        wandb_dict[
                            f"{task_name}/{ood_type}/{m}_Avg"
                        ] = np.mean(step_values[m])

            for past_task_id in range(tid + 1):
                for ood_type in OOD_TYPES:
                    task_auroc = (
                        self.ood_results.get(tid, {})
                        .get(past_task_id, {})
                        .get(ood_type, {})
                        .get("AUROC", None)
                    )
                    if task_auroc is not None:
                        wandb_dict[
                            f"{task_name}/{ood_type}/AUROC_ID-{past_task_id}"
                        ] = task_auroc

            task_data = self.ood_results.get(tid, {})
            for past_task_id in range(tid + 1):
                pair_data = task_data.get(past_task_id, {})
                per_ds = pair_data.get("_per_dataset", {})
                pair_name = f"Task-{tid}-ID-{past_task_id}"
                for ood_type, ds_dict in per_ds.items():
                    avg_acc = {}
                    for ds_name, ds_metrics in ds_dict.items():
                        for m, val in ds_metrics.items():
                            key = f"{pair_name}/{ood_type}/{ds_name}/{m}"
                            wandb_dict[key] = val
                            avg_acc.setdefault(m, []).append(val)
                    for m, vals in avg_acc.items():
                        wandb_dict[
                            f"{pair_name}/{ood_type}/{m}_AVG"
                        ] = np.mean(vals)

            pooled = self.pooled_results.get(tid, {})
            pooled_name = f"Task-{tid}-Pooled"
            for ood_type, pooled_data in pooled.items():
                per_ds = pooled_data.get("_per_dataset", {})
                for ds_name, ds_metrics in per_ds.items():
                    for m, val in ds_metrics.items():
                        wandb_dict[
                            f"{pooled_name}/{ood_type}/{ds_name}/{m}"
                        ] = val

            if wandb_dict:
                log_wandb_metrics(self.wandb_logger, wandb_dict)
        print(f"Re-logged WandB metrics for tasks 0..{last_completed_task_id}.")

    # =========================================================================
    #                                TRACKING UPDATES
    # =========================================================================

    def log_cl_metrics(self, metrics: Dict):
        self.cl_results.append(metrics)

    def update_cka(
        self, model, loader, data_task_id, current_task_id, max_batches
    ):
        self.recorders["cka"].update(
            model,
            loader,
            data_task_id,
            current_task_id,
            max_batches=max_batches,
        )

    def update_cka_ood(
        self, model, loader, current_task_id, ood_type, max_batches=10
    ):
        self.recorders["cka"].update_ood(
            model, loader, current_task_id, ood_type, max_batches=max_batches
        )

    # =========================================================================
    #                           OOD SUMMARY & STORAGE
    # =========================================================================

    def log_ood_summary(self, current_task_id):
        """Compute Last (stream AUROC at step) + Avg (running mean across steps)."""
        task_name = f"Task-{current_task_id}"
        metrics_by_type = {}

        for ood_type in OOD_TYPES:
            rm = RunningMetric()
            for past_task_id in range(current_task_id + 1):
                task_metrics = (
                    self.ood_results.get(current_task_id, {})
                    .get(past_task_id, {})
                    .get(ood_type, {})
                )
                if not task_metrics:
                    continue
                for metric in OOD_METRIC_List:
                    val = task_metrics.get(metric, 0.0)
                    rm.update(ood_type, metric, val)
            avg_metrics = {m: rm.mean(ood_type, m) for m in OOD_METRIC_List}
            if any(not np.isnan(v) for v in avg_metrics.values()):
                metrics_by_type[ood_type] = avg_metrics

        running_avg_by_type = {}
        for ood_type in OOD_TYPES:
            step_values = {m: [] for m in OOD_METRIC_List}
            for i in range(current_task_id):
                prev = (
                    self.ood_results.get(i, {})
                    .get("ALL_TASKS", {})
                    .get(ood_type, {})
                )
                for metric in OOD_METRIC_List:
                    val = prev.get(metric, None)
                    if val is not None and not np.isnan(val):
                        step_values[metric].append(val)
            curr = metrics_by_type.get(ood_type, {})
            for metric in OOD_METRIC_List:
                val = curr.get(metric, None)
                if val is not None and not np.isnan(val):
                    step_values[metric].append(val)
            running_avg = {
                m: np.mean(step_values[m]) if step_values[m] else np.nan
                for m in OOD_METRIC_List
            }
            if any(not np.isnan(v) for v in running_avg.values()):
                running_avg_by_type[ood_type] = running_avg

        wandb_dict = {}
        for ood_type, metrics in metrics_by_type.items():
            for m in OOD_METRIC_List:
                val = metrics.get(m, None)
                if val is not None and not np.isnan(val):
                    wandb_dict[f"{task_name}/{ood_type}/{m}_Last"] = val
            for past_task_id in range(current_task_id + 1):
                task_auroc = (
                    self.ood_results.get(current_task_id, {})
                    .get(past_task_id, {})
                    .get(ood_type, {})
                    .get("AUROC", None)
                )
                if task_auroc is not None:
                    wandb_dict[
                        f"{task_name}/{ood_type}/AUROC_ID-{past_task_id}"
                    ] = task_auroc
        for ood_type, metrics in running_avg_by_type.items():
            for m in OOD_METRIC_List:
                val = metrics.get(m, None)
                if val is not None and not np.isnan(val):
                    wandb_dict[f"{task_name}/{ood_type}/{m}_Avg"] = val
        log_wandb_metrics(self.wandb_logger, wandb_dict)

        print(f"\n{'='*60}")
        print(f"  OOD Summary after {task_name}")
        print(f"{'='*60}")
        for ood_type in OOD_TYPES:
            last_m = metrics_by_type.get(ood_type, {})
            avg_m = running_avg_by_type.get(ood_type, {})
            if not last_m and not avg_m:
                continue
            print(f"  {ood_type}:")
            if last_m:
                print(
                    f"    Last (step {current_task_id}):   "
                    f"AUROC={last_m.get('AUROC', 0):.4f}  "
                    f"FPR@95={last_m.get('FPR@95', 0):.4f}"
                )
            if avg_m:
                print(
                    f"    Avg  (steps 0..{current_task_id}): "
                    f"AUROC={avg_m.get('AUROC', 0):.4f}  "
                    f"FPR@95={avg_m.get('FPR@95', 0):.4f}"
                )
        print(f"{'='*60}\n")

        if current_task_id not in self.ood_results:
            self.ood_results[current_task_id] = {}
        self.ood_results[current_task_id]["ALL_TASKS"] = metrics_by_type

    def store_task_ood_results(
        self,
        current_task_id,
        past_task_id,
        ood_type,
        metrics,
        per_dataset_metrics=None,
    ):
        if current_task_id not in self.ood_results:
            self.ood_results[current_task_id] = {}
        if past_task_id not in self.ood_results[current_task_id]:
            self.ood_results[current_task_id][past_task_id] = {}
        self.ood_results[current_task_id][past_task_id][ood_type] = metrics
        if per_dataset_metrics:
            if (
                "_per_dataset"
                not in self.ood_results[current_task_id][past_task_id]
            ):
                self.ood_results[current_task_id][past_task_id][
                    "_per_dataset"
                ] = {}
            self.ood_results[current_task_id][past_task_id]["_per_dataset"][
                ood_type
            ] = per_dataset_metrics

    def store_pooled_results(
        self, current_task_id, ood_type, metrics, per_dataset_metrics=None
    ):
        if current_task_id not in self.pooled_results:
            self.pooled_results[current_task_id] = {}
        self.pooled_results[current_task_id][ood_type] = {
            "_avg": metrics,
            "_per_dataset": per_dataset_metrics or {},
        }

    def log_pair_metrics(
        self, current_task_id, id_task_idx, ood_type, all_conf
    ):
        if (
            ood_type not in all_conf
            or "dataset_metric" not in all_conf[ood_type]
        ):
            return
        task_name = f"Task-{current_task_id}-ID-{id_task_idx}"
        datasets_metrics = all_conf[ood_type]["dataset_metric"]
        avg_accumulator = {}
        for dataset_name, (metrics, _) in datasets_metrics.items():
            for metric_name, value in metrics.items():
                log_key = f"{task_name}/{ood_type}/{dataset_name}/{metric_name}"
                if self.wandb_logger:
                    log_wandb_metrics(self.wandb_logger, {log_key: value})
                avg_accumulator.setdefault(metric_name, []).append(value)
        for metric_name, values in avg_accumulator.items():
            avg_val = np.mean(values)
            log_key_avg = f"{task_name}/{ood_type}/{metric_name}_AVG"
            if self.wandb_logger:
                log_wandb_metrics(self.wandb_logger, {log_key_avg: avg_val})

    # =========================================================================
    #                                FINAL ANALYSIS
    # =========================================================================

    def generate_final_report(self, benchmark):
        print("\n--- Final Analysis ---")
        final_metrics = {}
        n_experiences = len(benchmark.train_stream)

        if self.cl_results:
            self._compute_cl_stream_metrics(final_metrics, n_experiences)

        self._log_final_ood_metrics(final_metrics, n_experiences)

        is_joint = self.config.strategy.name.lower() == "joint"
        if not is_joint:
            self._analyze_ood_deterioration(final_metrics, n_experiences)
            self._analyze_cka()

        print("\n--- Final Consolidated Metrics ---")
        log_wandb_metrics(self.wandb_logger, final_metrics)
        for key, value in final_metrics.items():
            print(f"{key}: {value:.4f}")

    def _compute_cl_stream_metrics(self, final_metrics, n_experiences):
        avg_forgetting, _ = compute_true_stream_forgetting(
            self.cl_results, n_experiences, self.config.scenario.return_task_id
        )
        final_metrics["final/StreamForgetting"] = avg_forgetting * 100.0

        # Per-experience accuracy AT THE END of the stream — one value per
        # experience, all distinct. Uses the per-Exp key so that with
        # return_task_id=False we don't end up with N copies of the same
        # stream-mean number. Falls back to the bare Exp key for strategies
        # that don't include the Task prefix.
        accuracies = []
        for task_idx in range(n_experiences):
            t_prefix = (
                f"Task{task_idx:0>3}"
                if self.config.scenario.return_task_id
                else "Task000"
            )
            exp_key = (
                f"Top1_Acc_Exp/eval_phase/test_stream/"
                f"{t_prefix}/Exp{task_idx:0>3}"
            )
            acc = self.cl_results[-1].get(exp_key, None)
            if acc is None:
                acc = self.cl_results[-1].get(
                    f"Top1_Acc_Exp/eval_phase/test_stream/Exp{task_idx:0>3}",
                    None,
                )
            if acc is not None:
                accuracies.append(acc)
                final_metrics[f"Metric/Task{task_idx}-last/accuracy"] = acc

        # Trajectory of accuracy on Exp 0 across phases — read the per-Exp key
        # so this is the actual Task-0 accuracy at each phase, not the
        # Stream mean.
        t0_key = "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"
        t0_fallback = "Top1_Acc_Exp/eval_phase/test_stream/Exp000"
        for tid, cl_metric in enumerate(self.cl_results):
            acc = cl_metric.get(t0_key, None)
            if acc is None:
                acc = cl_metric.get(t0_fallback, None)
            if acc is None:
                continue
            final_metrics[f"Metric/AtT{tid}/Task0/accuracy"] = acc
        last_accuracy = np.mean(accuracies) * 100.0 if accuracies else 0.0
        final_metrics["final/lastAccuracy"] = last_accuracy
        acc_avg_key = "Metric/Average_Incremental_Accuracy"
        avg_accuracy = self.cl_results[-1].get(acc_avg_key, None)
        avg_accuracy = avg_accuracy * 100 if avg_accuracy is not None else 0.0
        final_metrics["final/AvgAccuracy"] = avg_accuracy

    def _log_final_ood_metrics(self, final_metrics, n_experiences):
        last_task_id = n_experiences - 1
        accum = {
            ood_type: {m: [] for m in OOD_METRIC_List} for ood_type in OOD_TYPES
        }
        for b in range(n_experiences):
            all_tasks = self.ood_results.get(b, {}).get("ALL_TASKS", {})
            for ood_type in OOD_TYPES:
                task_metrics = all_tasks.get(ood_type, {})
                for metric in OOD_METRIC_List:
                    val = task_metrics.get(metric, None)
                    if val is not None and not np.isnan(val):
                        accum[ood_type][metric].append(val)

        for ood_type in OOD_TYPES:
            for metric in OOD_METRIC_List:
                if accum[ood_type][metric]:
                    final_metrics[f"final/{ood_type}/{metric}_Avg"] = np.mean(
                        accum[ood_type][metric]
                    )
                last_val = (
                    self.ood_results.get(last_task_id, {})
                    .get("ALL_TASKS", {})
                    .get(ood_type, {})
                    .get(metric, None)
                )
                if last_val is not None and not np.isnan(last_val):
                    final_metrics[f"final/{ood_type}/{metric}_Last"] = last_val

    def _plot_correlation(self, data_df, plot_func, **kwargs):
        if self.auroc_df.empty or data_df.empty:
            return
        for ood_type in OOD_TYPES:
            auroc_sub = self.auroc_df[
                (self.auroc_df["ood_type"] == ood_type)
                & (self.auroc_df["metric"] == "AUROC")
            ].copy()
            merged = pd.merge(auroc_sub, data_df, on="task", how="inner")
            if not merged.empty:
                plot_func(
                    merged,
                    self.output_dir,
                    self.wandb_logger,
                    ood_type=ood_type,
                    **kwargs,
                )

    def _analyze_ood_deterioration(self, final_metrics, n_experiences):
        print("Calculating Final OOD Performance Deterioration...")
        last_task_id = n_experiences - 1
        deterioration_results = []
        metric_signs = {m: -1 if "FPR" in m else 1 for m in OOD_METRIC_List}

        for t in range(n_experiences):
            for ood_type in OOD_TYPES:
                for metric, sign in metric_signs.items():
                    peak = (
                        self.ood_results.get(t, {})
                        .get(t, {})
                        .get(ood_type, {})
                        .get(metric, np.nan)
                    )
                    final = (
                        self.ood_results.get(last_task_id, {})
                        .get(t, {})
                        .get(ood_type, {})
                        .get(metric, np.nan)
                    )
                    if np.isnan(peak):
                        print(
                            f"[Deterioration] WARNING: missing peak for "
                            f"task={t}, ood_type={ood_type}, metric={metric}."
                        )
                    if np.isnan(final):
                        print(
                            f"[Deterioration] WARNING: missing final for "
                            f"task={t}, ood_type={ood_type}, metric={metric}."
                        )
                    det = (peak * sign) - (final * sign)
                    deterioration_results.append(
                        {
                            "task": t,
                            "ood_type": ood_type,
                            "metric": metric,
                            "deterioration": det,
                            "peak_value": peak,
                            "final_value": final,
                        }
                    )
                    if t < last_task_id:
                        final_metrics[
                            f"final/Task_{t}/{ood_type}/{metric}_Deterioration"
                        ] = det

        self.auroc_df = pd.DataFrame(deterioration_results)
        self.auroc_df.to_csv(
            os.path.join(
                self.output_dir, f"{self.config.exp_name}_ood_deterioration.csv"
            ),
            index=False,
        )

        for ood_type in OOD_TYPES:
            for metric in OOD_METRIC_List:
                sub_df = self.auroc_df[
                    (self.auroc_df["ood_type"] == ood_type)
                    & (self.auroc_df["metric"] == metric)
                ]
                past_df = sub_df[sub_df["task"] < last_task_id]
                final_metrics[
                    f"final/{ood_type}/{metric}_Deterioration_AVG"
                ] = (
                    past_df["deterioration"].mean()
                    if not past_df.empty
                    else 0.0
                )
                if not sub_df.empty:
                    plot_auroc_performance_per_task(
                        sub_df,
                        self.output_dir,
                        self.wandb_logger,
                        ood_type,
                        make_plots=self.config.scenario.make_plots,
                    )

    def _analyze_cka(self):
        print("Analyzing CKA Representation Similarity...")
        cka_df = self.recorders["cka"].finalize()
        if cka_df.empty:
            print("No CKA data recorded.")
            return

        cka_df.to_csv(
            os.path.join(
                self.output_dir, f"{self.config.exp_name}_cka_results.csv"
            ),
            index=False,
        )

        plot_cka_trajectory(
            cka_df,
            self.output_dir,
            self.wandb_logger,
            make_plots=self.config.scenario.make_plots,
        )

        drift_df = cka_df[
            cka_df["comparison_type"] == "drift_from_origin"
        ].copy()
        if not drift_df.empty:
            task_cka = (
                drift_df.sort_values("measured_at")
                .groupby("task")
                .last()
                .reset_index()[["task", "cka_similarity"]]
            )
            self._plot_correlation(
                task_cka,
                plot_cka_vs_auroc,
                make_plots=self.config.scenario.make_plots,
            )
