# -*- coding: utf-8 -*-
"""Evaluate a feature-space analogue of TOOD.

At the final checkpoint, this standalone analysis compares two post-hoc,
training-free feature-space OOD scores for each ID task against the OOD pool:

  * **MDS analogue (baseline):** ``S(x) = -min_t d_t(x)`` where ``d_t(x)`` is the
    distance from ``x`` to the nearest class prototype of task ``t`` in
    (L2-normalised) penultimate-feature space. This is the standard
    nearest-prototype detector and is the one that suffers from manifold
    crowding.

  * **Feature TOOD:** per-task distances are first normalized toward
    a reference task using robust per-task ID statistics (median / MAD computed
    from each task's replay/train samples — the same statistics TOOD uses for
    energy), then ``S(x) = -min_t d_t^norm(x)``. This is the direct
    feature-space analogue of TOOD's per-task energy normalization.

It reuses the benchmark, model, and OOD-loader construction from
``evaluate.py`` so the data and checkpoints match the main results. It writes a
CSV and a per-task AUROC comparison plot, and logs scalars to W&B
(project ``rebuttal_feature_analog`` when ``WANDB_PROJECT`` is set).

Usage (same CLI as evaluate.py)::

    python analysis/feature_tood_analog.py --config <cfgs...> [overrides]
"""
import json
import os
import sys

import numpy as np
import torch
import torchvision  # noqa: F401  (preload GL libs before openood -> cv2 import)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avalanche.models.dynamic_modules import avalanche_model_adaptation
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss

from core.setup import get_benchmark, init_model, setup_config
from openood.datasets import get_ood_dataloader
from openood.evaluators.metrics import compute_all_metrics
from utils import device
from utils.factory import get_strategy_class
from utils.helpers import (
    CustomWandbLogger,
    StrategyStateHelper,
    get_task_dataloaders,
    set_seed,
)


def _ckpt_path(config, task_id):
    assert "_OOD_" in config.exp_name, "exp_name must contain '_OOD_'."
    base = config.exp_name.split("_OOD_")[0]
    return os.path.join(config.scenario.ckpt_dir, f"{base}_{task_id}.pth")


@torch.no_grad()
def _extract(model, loader):
    """Return (features [N, D], labels [N]) for every sample in a loader."""
    feats, labels = [], []
    model.eval()
    model.is_ood_eval = True
    for batch in loader:
        if isinstance(batch, dict):
            x, y = batch["data"], batch["label"]
        else:
            x, y = batch[0], batch[1]
        x = x.to(device)
        out = model(x, return_feature=True)
        f = out[1] if isinstance(out, (tuple, list)) else out
        f = f.reshape(f.shape[0], -1)
        feats.append(f.cpu().numpy())
        labels.append(np.asarray(y).reshape(-1))
    model.is_ood_eval = False
    if not feats:
        return np.empty((0, 0)), np.empty((0,))
    return np.concatenate(feats, 0), np.concatenate(labels, 0)


def _l2norm(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _min_dist_to_task(feats, prototypes):
    """Min Euclidean distance from each row of feats to a task's prototypes.

    prototypes: [K, D] class prototypes for one task. Returns [N].
    """
    # ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b ; features & prototypes are L2-normed
    d = np.linalg.norm(feats[:, None, :] - prototypes[None, :, :], axis=2)
    return d.min(axis=1)


def _auroc(id_scores, ood_scores):
    """AUROC with the OpenOOD convention (higher score => more ID)."""
    conf = np.concatenate([id_scores, ood_scores])
    label = np.concatenate(
        [np.ones_like(id_scores), -np.ones_like(ood_scores)]
    )
    pred = np.zeros_like(conf)  # unused by AUROC column
    metrics = compute_all_metrics(conf, label, pred)
    # compute_all_metrics returns [fpr, auroc, aupr_in, aupr_out, ...]
    return float(metrics[1]) * 100.0


def main():
    config = setup_config()
    set_seed(config.scenario.seed)

    project = os.getenv("WANDB_PROJECT", "rebuttal_feature_analog")
    wandb_logger = CustomWandbLogger(
        project_name=project, run_name=config.exp_name, config=config
    )

    benchmark, eval_transform = get_benchmark(config)
    config.ood_dataset.shuffle = False
    ood_loader_dict = get_ood_dataloader(config)

    model, fc_layer = init_model(config, benchmark)
    model = model.to(device)
    dummy_opt = torch.optim.SGD(model.parameters(), lr=0.0)
    strategy, _ = get_strategy_class(
        config, model, dummy_opt, CrossEntropyLoss(),
        EvaluationPlugin(), fc_layer=fc_layer,
    )

    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream
    n_tasks = len(train_stream)

    # Grow the classifier head through every experience, then load the final
    # checkpoint (theta_N) — mirrors evaluate.py's adaptation order.
    for task in train_stream:
        avalanche_model_adaptation(strategy.model, task)
    final_ckpt = _ckpt_path(config, n_tasks - 1)
    if not StrategyStateHelper.load(strategy, final_ckpt, device):
        raise FileNotFoundError(f"Missing final checkpoint: {final_ckpt}")
    model = strategy.model.to(device)

    nw = config.dataset.num_workers

    # --- Per-task prototypes + per-task ID distance statistics (from TRAIN
    #     data, the replay/buffer analogue) and ID TEST features for scoring.
    prototypes = {}          # task_id -> [K, D]
    dist_stats = {}          # task_id -> (median, mad)
    id_test_feats = {}       # task_id -> [N, D]
    for t in range(n_tasks):
        tr_loader, te_loader = get_task_dataloaders(
            config, train_stream, test_stream, t,
            shuffle_train=False, shuffle_test=False, num_workers=nw,
            return_task_id=config.scenario.return_task_id,
        )
        tr_f, tr_y = _extract(model, tr_loader)
        te_f, _ = _extract(model, te_loader)
        tr_f, te_f = _l2norm(tr_f), _l2norm(te_f)
        classes = sorted(set(tr_y.tolist()))
        protos = np.stack([tr_f[tr_y == c].mean(0) for c in classes], 0)
        protos = _l2norm(protos)
        prototypes[t] = protos
        id_test_feats[t] = te_f
        # per-task ID distance distribution from train samples (buffer analog)
        d_tr = _min_dist_to_task(tr_f, protos)
        med = float(np.median(d_tr))
        mad = max(float(np.median(np.abs(d_tr - med))) * 1.4826, 1e-6)
        dist_stats[t] = (med, mad)

    ref_med, ref_mad = dist_stats[n_tasks - 1]  # most-recent-task anchor

    # --- OOD features (pool all near+far OOD datasets for a single graph).
    ood_feats = []
    for split in ood_loader_dict:
        if split in ("val", "id"):
            continue
        loaders = ood_loader_dict[split]
        if not isinstance(loaders, dict):
            continue
        for name, dl in loaders.items():
            f, _ = _extract(model, dl)
            if f.size:
                ood_feats.append(_l2norm(f))
    ood_feats = (
        np.concatenate(ood_feats, 0) if ood_feats else np.empty((0, 1))
    )

    def task_dist_matrix(feats):
        """[N, T] min distance per task, and its TOOD-normalised version."""
        raw = np.stack(
            [_min_dist_to_task(feats, prototypes[t]) for t in range(n_tasks)],
            axis=1,
        )
        norm = raw.copy()
        for t in range(n_tasks):
            med, mad = dist_stats[t]
            norm[:, t] = (raw[:, t] - med) / mad * ref_mad + ref_med
        return raw, norm

    ood_raw, ood_norm = task_dist_matrix(ood_feats)
    # baseline / ours scores: higher => more ID (so negate distance)
    ood_s_mds = -ood_raw.min(axis=1)
    ood_s_ftood = -ood_norm.min(axis=1)

    rows = []
    for i in range(n_tasks):
        id_raw, id_norm = task_dist_matrix(id_test_feats[i])
        id_s_mds = -id_raw.min(axis=1)
        id_s_ftood = -id_norm.min(axis=1)
        auroc_mds = _auroc(id_s_mds, ood_s_mds)
        auroc_ftood = _auroc(id_s_ftood, ood_s_ftood)
        rows.append((i, auroc_mds, auroc_ftood))
        print(
            f"[FeatureTOOD] Task {i}: MDS-analog AUROC={auroc_mds:.2f}  "
            f"Feature-TOOD AUROC={auroc_ftood:.2f}  "
            f"(delta={auroc_ftood - auroc_mds:+.2f})"
        )
        if wandb_logger:
            wandb_logger.wandb.log({
                f"FeatureTOOD/Task_{i}/MDS_AUROC": auroc_mds,
                f"FeatureTOOD/Task_{i}/FeatureTOOD_AUROC": auroc_ftood,
                f"FeatureTOOD/Task_{i}/delta": auroc_ftood - auroc_mds,
            })

    arr = np.array([[r[1], r[2]] for r in rows])
    avg_mds, avg_ftood = arr[:, 0].mean(), arr[:, 1].mean()
    print(
        f"[FeatureTOOD] AVG  MDS-analog={avg_mds:.2f}  "
        f"Feature-TOOD={avg_ftood:.2f}  (delta={avg_ftood - avg_mds:+.2f})"
    )

    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{config.exp_name}_feature_tood.csv")
    with open(csv_path, "w") as f:
        f.write("task,mds_analog_auroc,feature_tood_auroc\n")
        for i, a, b in rows:
            f.write(f"{i},{a:.4f},{b:.4f}\n")
        f.write(f"avg,{avg_mds:.4f},{avg_ftood:.4f}\n")
    print(f"[FeatureTOOD] wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tasks = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tasks, arr[:, 0], "o-", label=f"MDS analog (avg {avg_mds:.1f})")
        ax.plot(tasks, arr[:, 1], "s-",
                label=f"Feature-TOOD (avg {avg_ftood:.1f})")
        ax.set_xlabel("ID task index")
        ax.set_ylabel("OOD AUROC (%)")
        ax.set_title("Per-task feature-distance normalization")
        ax.legend()
        ax.grid(True, alpha=0.3)
        png = os.path.join(out_dir, f"{config.exp_name}_feature_tood.png")
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        print(f"[FeatureTOOD] wrote {png}")
        if wandb_logger:
            import wandb
            wandb_logger.wandb.log({"FeatureTOOD/per_task_auroc": wandb.Image(png)})
    except Exception as e:  # plotting is best-effort
        print(f"[FeatureTOOD] plot skipped: {e}")

    if wandb_logger:
        wandb_logger.wandb.log({
            "FeatureTOOD/AVG_MDS_AUROC": avg_mds,
            "FeatureTOOD/AVG_FeatureTOOD_AUROC": avg_ftood,
            "FeatureTOOD/AVG_delta": avg_ftood - avg_mds,
        })
        wandb_logger.wandb.finish()


if __name__ == "__main__":
    main()
