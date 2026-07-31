# Minimal reproduction surface: only CKA + AUROC-deterioration paths are retained.
import os

import matplotlib.pyplot as plt
import pandas as pd
import PIL
import seaborn as sns

from utils.helpers import log_wandb_metrics


def plot_auroc_performance_per_task(
    df: pd.DataFrame,
    output_dir,
    wandb_logger=None,
    ood_type="farood",
    make_plots=True,
):
    if not make_plots:
        return

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f'AUROC Performance and Deterioration per Task for "{ood_type}"',
        fontsize=16,
        y=1.02,
    )

    auroc_melted = df.melt(
        id_vars=["task"],
        value_vars=["peak_value", "final_value"],
        var_name="Metric Type",
        value_name="AUROC Score",
    )

    sns.barplot(
        data=auroc_melted,
        x="task",
        y="AUROC Score",
        hue="Metric Type",
        ax=axes[0],
        palette={"peak_value": "cornflowerblue", "final_value": "salmon"},
    )
    axes[0].set_title("Peak vs. Final AUROC per Task")
    axes[0].set_ylabel("AUROC Score (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(title="Metric Type")

    sns.barplot(
        data=df, x="task", y="deterioration", ax=axes[1], color="lightcoral"
    )
    axes[1].set_title("AUROC Deterioration (Peak − Final)")
    axes[1].set_xlabel("Task Index")
    axes[1].set_ylabel("Deterioration (Percentage Points)")

    plt.tight_layout()
    path = os.path.join(output_dir, f"deterioration_per_task_{ood_type}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    if wandb_logger:
        try:
            image = PIL.Image.open(path)
            log_wandb_metrics(
                wandb_logger,
                {
                    f"Deterioration/{ood_type}/deterioration_per_task_plot": image
                },
            )
        except Exception as e:
            print(f"Failed to log to WandB: {e}")


def plot_cka_vs_auroc(
    df: pd.DataFrame,
    output_dir,
    wandb_logger=None,
    ood_type="farood",
    make_plots=True,
):
    """Scatter plot of CKA similarity vs. AUROC deterioration per task."""
    metric_name = "cka_similarity"
    if metric_name not in df.columns or df[metric_name].isnull().all():
        print(f"Skipping CKA plot: '{metric_name}' not found or all NaN.")
        return

    path = os.path.join(output_dir, f"cka_vs_auroc_{ood_type}.png")

    if make_plots:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x=metric_name,
            y="deterioration",
            hue="task",
            palette="viridis",
            alpha=0.8,
            s=150,
            legend="full",
        )
        plt.title(
            f"CKA Similarity vs. AUROC Deterioration ({ood_type.upper()})",
            fontsize=16,
            pad=15,
        )
        plt.xlabel("CKA Similarity (higher = structure preserved)", fontsize=12)
        plt.ylabel("AUROC Deterioration (%) for Task", fontsize=12)
        plt.xlim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(
            title="Task",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    if wandb_logger:
        try:
            log_dict = {}
            if make_plots:
                image = PIL.Image.open(path)
                log_dict[f"CKA/{ood_type}/cka_vs_auroc_plot"] = image
            for _, row in df.iterrows():
                task_id = int(row["task"])
                log_dict[f"CKA/Task_{task_id}/cka_similarity"] = row[
                    metric_name
                ]
            log_wandb_metrics(wandb_logger, log_dict)
        except Exception as e:
            print(f"Failed to log CKA vs AUROC to WandB: {e}")


def plot_cka_trajectory(
    df: pd.DataFrame,
    output_dir,
    wandb_logger=None,
    make_plots=True,
):
    """
    Two-panel plot showing CKA over time:
      Left  - CKA of task-0 representations vs. later training stages (drift).
      Right - Adjacent-task CKA (T-1 vs T similarity).
    """
    if df.empty:
        return

    path = os.path.join(output_dir, "cka_trajectory.png")

    drift_df = df[df["comparison_type"] == "drift_from_origin"].copy()
    adjacent_df = df[df["comparison_type"] == "adjacent"].copy()
    ood_drift_df = df[df["comparison_type"] == "ood_drift"].copy()

    if drift_df.empty and adjacent_df.empty and ood_drift_df.empty:
        return

    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        task0_drift = drift_df[drift_df["task"] == 0]
        if not task0_drift.empty:
            ax1.plot(
                task0_drift["measured_at"],
                task0_drift["cka_similarity"],
                marker="o",
                linewidth=2,
                markersize=8,
                color="tab:blue",
                label="Task 0 representations",
            )
        ood_colors = {"nearood": "tab:red", "farood": "tab:green"}
        for ood_type in ood_drift_df["task"].unique():
            ood_sub = ood_drift_df[ood_drift_df["task"] == ood_type]
            color = ood_colors.get(ood_type, None)
            ax1.plot(
                ood_sub["measured_at"],
                ood_sub["cka_similarity"],
                marker="^",
                linewidth=2,
                markersize=7,
                linestyle="--",
                color=color,
                label=f"{ood_type} representations",
            )
        ax1.set_xlabel("Evaluation Task", fontsize=12)
        ax1.set_ylabel("CKA Similarity to Original", fontsize=12)
        ax1.set_title("Representation Drift", fontsize=14)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        ax2 = axes[1]
        if not adjacent_df.empty:
            ax2.plot(
                adjacent_df["measured_at"],
                adjacent_df["cka_similarity"],
                marker="s",
                linewidth=2,
                markersize=8,
                color="tab:orange",
                label="Adjacent task CKA",
            )
        ax2.set_xlabel("Task Transition (T-1 to T)", fontsize=12)
        ax2.set_ylabel("CKA Similarity", fontsize=12)
        ax2.set_title("Adjacent Task Similarity", fontsize=14)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    if wandb_logger:
        try:
            log_dict = {}
            for _, row in drift_df.iterrows():
                t_ref = row["task"]
                t_meas = int(row["measured_at"])
                log_dict[f"CKA/Task{t_ref}_drift_at_T{t_meas}"] = row[
                    "cka_similarity"
                ]
            for _, row in adjacent_df.iterrows():
                t_meas = int(row["measured_at"])
                log_dict[f"CKA/Adjacent_T{t_meas-1}_to_T{t_meas}"] = row[
                    "cka_similarity"
                ]
            for _, row in ood_drift_df.iterrows():
                ood_type = row["task"]
                t_meas = int(row["measured_at"])
                log_dict[f"CKA/{ood_type}_drift_at_T{t_meas}"] = row[
                    "cka_similarity"
                ]
            if make_plots:
                image = PIL.Image.open(path)
                log_dict["CKA/trajectory_plot"] = image
            log_wandb_metrics(wandb_logger, log_dict)
        except Exception as e:
            print(f"Failed to log CKA trajectory to WandB: {e}")
