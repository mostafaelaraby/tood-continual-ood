# -*- coding: utf-8 -*-
"""Run a synthetic experiment that isolates the confidence gap.

We construct a fully controlled class-incremental stream of Gaussian-blob
classes in a low-dimensional space and train a small MLP with an *expanding*
linear head, exactly mirroring the head-expansion dynamics that drive the
Confidence Gap in the real experiments. Because the data geometry is fixed and
known, changes in OOD separability can be attributed to output-score drift
rather than representation change.

The experiment reuses the TOOD scoring code from ``managers.ood_manager``
(``compute_per_task_energy``, ``normalize_energies``,
``compute_tood_score``, ``EnergyStatsStore``), rather than using a
separate implementation.

Outputs (written to ``--out_dir``; logged to W&B project ``rebuttal_toy`` when
``WANDB_PROJECT`` is set):
  * ``toy_confidence_gap.csv`` / ``.png`` — per-task ID energy + the gap.
  * ``toy_auroc_over_tasks.png`` — global-energy vs TOOD AUROC across the stream.
  * ``toy_lambda_sweep.csv`` / ``.png`` — AUROC vs lambda, including a
    high-overlap regime that exposes the lambda failure mode.

Usage::

    python analysis/toy_confidence_gap.py [--n_tasks 8] [--classes_per_task 2]
        [--dim 16] [--separation 4.0] [--epochs 60] [--seed 0]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from managers.ood_manager import (
    EnergyStatsStore,
    compute_per_task_energy,
    normalize_energies,
    compute_tood_score,
)


def auroc(id_scores, ood_scores):
    """Mann-Whitney AUROC; higher score => more ID."""
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n1, n2 = len(id_scores), len(ood_scores)
    if n1 == 0 or n2 == 0:
        return float("nan")
    order = np.argsort(np.concatenate([id_scores, ood_scores]))
    ranks = np.empty(n1 + n2)
    ranks[order] = np.arange(1, n1 + n2 + 1)
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n2)) * 100.0


class MLP(nn.Module):
    """Small MLP backbone with an expanding linear head (CIL head growth)."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 0)  # grows as tasks arrive
        self.hidden = hidden

    def grow(self, n_new):
        old = self.head
        new = nn.Linear(self.hidden, old.out_features + n_new)
        with torch.no_grad():
            if old.out_features > 0:
                new.weight[: old.out_features] = old.weight
                new.bias[: old.out_features] = old.bias
        self.head = new

    def forward(self, x):
        return self.head(self.backbone(x))


def make_blobs(rng, centers, n_per, dim, spread=1.0):
    xs, ys = [], []
    for cls, c in enumerate(centers):
        xs.append(rng.normal(c, spread, size=(n_per, dim)))
        ys.append(np.full(n_per, cls))
    return np.concatenate(xs), np.concatenate(ys)


def run(args, separation, tag, wandb_logger=None):
    rng = np.random.RandomState(args.seed)
    n_classes = args.n_tasks * args.classes_per_task
    # fixed, well-separated class centers on a random orthogonal-ish layout
    centers = rng.normal(0, 1, size=(n_classes, args.dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= separation
    # OOD: the interior/origin region — far from every ID cluster (which live
    # on a sphere of radius `separation`) yet in the network's low-activation
    # interior, so OOD logits/energy stay low (the realistic regime; placing
    # OOD *outside* the sphere makes a ReLU-MLP pathologically overconfident).
    ood_x, _ = make_blobs(
        rng, [np.zeros(args.dim)], 500, args.dim, spread=separation * 0.25
    )

    model = MLP(args.dim)
    opt = None
    task_class_map = {}
    buffer = {}  # class -> features cache (replay analogue for calibration)
    per_task_train = {}
    test_sets = {}

    # metrics across the stream
    gap_rows, auroc_rows = [], []

    for t in range(args.n_tasks):
        cls_ids = list(range(t * args.classes_per_task,
                             (t + 1) * args.classes_per_task))
        task_class_map[t] = cls_ids
        tr_x, tr_local = make_blobs(
            rng, [centers[c] for c in cls_ids], args.n_per, args.dim
        )
        te_x, te_local = make_blobs(
            rng, [centers[c] for c in cls_ids], 200, args.dim
        )
        tr_y = np.array([cls_ids[i] for i in tr_local])
        te_y = np.array([cls_ids[i] for i in te_local])
        per_task_train[t] = (tr_x, tr_y)
        test_sets[t] = (te_x, te_y)

        # grow head + (re)build optimizer over all params
        model.grow(args.classes_per_task)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        # build training set: current task + small replay of old classes
        X = [tr_x]
        Y = [tr_y]
        for c, (bx, by) in buffer.items():
            X.append(bx)
            Y.append(by)
        X = torch.tensor(np.concatenate(X), dtype=torch.float32)
        Y = torch.tensor(np.concatenate(Y), dtype=torch.long)

        model.train()
        for _ in range(args.epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                loss = loss_fn(model(X[idx]), Y[idx])
                loss.backward()
                opt.step()

        # update replay buffer (k random exemplars per current class)
        for c in cls_ids:
            mask = tr_y == c
            xc = tr_x[mask][: args.buffer_per_class]
            buffer[c] = (xc, np.full(len(xc), c))

        # ---- measure at checkpoint theta_t ----
        model.eval()
        num_tasks = t + 1
        with torch.no_grad():
            # buffer-based per-task energy stats (TOOD calibration)
            stats = EnergyStatsStore()
            for bt in range(num_tasks):
                bx = np.concatenate(
                    [buffer[c][0] for c in task_class_map[bt]]
                )
                blog = model(torch.tensor(bx, dtype=torch.float32)).numpy()
                be = compute_per_task_energy(blog, task_class_map, num_tasks)
                stats.update(bt, be[:, bt], is_reference=(bt == num_tasks - 1))

            # confidence gap: mean ID energy of task 0 vs task t (own channel)
            if stats.has(0) and stats.has(t):
                gap_rows.append((t, stats.get(0).mean, stats.get(t).mean))

            # OOD scoring at this checkpoint for each seen ID task
            ood_log = model(torch.tensor(ood_x, dtype=torch.float32)).numpy()
            ood_global = np.log(np.exp(ood_log).sum(1) + 1e-9)
            ood_tood, *_ = compute_tood_score(
                ood_log, task_class_map, num_tasks, stats,
                method="robust_anchor", margin_lambda=args.lam,
            )
            for i in range(num_tasks):
                ix, _ = test_sets[i]
                id_log = model(torch.tensor(ix, dtype=torch.float32)).numpy()
                id_global = np.log(np.exp(id_log).sum(1) + 1e-9)
                id_tood, *_ = compute_tood_score(
                    id_log, task_class_map, num_tasks, stats,
                    method="robust_anchor", margin_lambda=args.lam,
                )
                auroc_rows.append((
                    t, i,
                    auroc(id_global, ood_global),
                    auroc(id_tood, ood_tood),
                ))

    # final model: lambda sweep on task 0 (oldest, worst confidence gap)
    lam_rows = []
    num_tasks = args.n_tasks
    with torch.no_grad():
        stats = EnergyStatsStore()
        for bt in range(num_tasks):
            bx = np.concatenate([buffer[c][0] for c in task_class_map[bt]])
            blog = model(torch.tensor(bx, dtype=torch.float32)).numpy()
            be = compute_per_task_energy(blog, task_class_map, num_tasks)
            stats.update(bt, be[:, bt], is_reference=(bt == num_tasks - 1))
        ood_log = model(torch.tensor(ood_x, dtype=torch.float32)).numpy()
        ix0, _ = test_sets[0]
        id0_log = model(torch.tensor(ix0, dtype=torch.float32)).numpy()
        for lam in np.linspace(0, 1.0, 11):
            id_s, *_ = compute_tood_score(
                id0_log, task_class_map, num_tasks, stats,
                method="robust_anchor", margin_lambda=float(lam))
            ood_s, *_ = compute_tood_score(
                ood_log, task_class_map, num_tasks, stats,
                method="robust_anchor", margin_lambda=float(lam))
            lam_rows.append((float(lam), auroc(id_s, ood_s)))

    return gap_rows, auroc_rows, lam_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=8)
    ap.add_argument("--classes_per_task", type=int, default=2)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--separation", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n_per", type=int, default=400)
    ap.add_argument("--buffer_per_class", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="toy_results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wandb_logger = None
    try:
        import wandb
        project = os.getenv("WANDB_PROJECT", "rebuttal_toy")
        if os.getenv("WANDB_API_KEY"):
            wandb_logger = wandb.init(
                project=project, name=f"toy_sep{args.separation}_seed{args.seed}",
                config=vars(args),
            )
    except Exception as e:
        print(f"[toy] wandb disabled: {e}")

    # well-separated regime (TOOD should succeed)
    gap, auc, lam_ok = run(args, args.separation, "well_sep", wandb_logger)
    # high-overlap regime (exposes lambda failure mode: tasks share energy mass)
    _, _, lam_overlap = run(args, args.separation * 0.35, "overlap", wandb_logger)

    # ---- write CSVs ----
    with open(os.path.join(args.out_dir, "toy_confidence_gap.csv"), "w") as f:
        f.write("task,energy_task0,energy_taskt,gap\n")
        for t, e0, et in gap:
            f.write(f"{t},{e0:.4f},{et:.4f},{et - e0:.4f}\n")
    with open(os.path.join(args.out_dir, "toy_lambda_sweep.csv"), "w") as f:
        f.write("lambda,auroc_well_sep,auroc_high_overlap\n")
        for (l1, a1), (l2, a2) in zip(lam_ok, lam_overlap):
            f.write(f"{l1:.2f},{a1:.4f},{a2:.4f}\n")

    # average AUROC over tasks at each checkpoint (global vs TOOD)
    import collections
    by_t = collections.defaultdict(list)
    for t, i, ag, at in auc:
        by_t[t].append((ag, at))
    auroc_traj = [
        (t, np.mean([x[0] for x in v]), np.mean([x[1] for x in v]))
        for t, v in sorted(by_t.items())
    ]

    print("\n[toy] Confidence gap (energy_task0 - energy_taskt):")
    for t, e0, et in gap:
        print(f"  t={t}: E_T0={e0:.2f}  E_Tt={et:.2f}  gap={et - e0:+.2f}")
    print("\n[toy] AUROC trajectory (avg over ID tasks):")
    for t, ag, at in auroc_traj:
        print(f"  t={t}: global={ag:.1f}  TOOD={at:.1f}  (+{at - ag:.1f})")
    print("\n[toy] Lambda sweep (task 0, final model):")
    for (l, a1), (_, a2) in zip(lam_ok, lam_overlap):
        print(f"  lambda={l:.2f}: well_sep AUROC={a1:.1f}  overlap AUROC={a2:.1f}")

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # confidence gap
        ts = [r[0] for r in gap]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ts, [r[1] for r in gap], "o-", label="ID energy (Task 0)")
        ax.plot(ts, [r[2] for r in gap], "s-", label="ID energy (current Task)")
        ax.set_xlabel("tasks learned"); ax.set_ylabel("mean ID energy")
        ax.set_title("Toy: Confidence Gap emerges with head expansion")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "toy_confidence_gap.png"), dpi=150)

        # auroc trajectory
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r[0] for r in auroc_traj], [r[1] for r in auroc_traj],
                "o-", label="Global energy")
        ax.plot([r[0] for r in auroc_traj], [r[2] for r in auroc_traj],
                "s-", label="TOOD")
        ax.set_xlabel("tasks learned"); ax.set_ylabel("OOD AUROC (%)")
        ax.set_title("Toy: TOOD restores separability lost to the gap")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "toy_auroc_over_tasks.png"), dpi=150)

        # lambda sweep (failure mode)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r[0] for r in lam_ok], [r[1] for r in lam_ok],
                "o-", label="well separated")
        ax.plot([r[0] for r in lam_overlap], [r[1] for r in lam_overlap],
                "s-", label="high overlap (failure mode)")
        ax.axvline(0.5, ls="--", c="grey", alpha=0.6, label="default lambda=0.5")
        ax.set_xlabel("margin lambda"); ax.set_ylabel("OOD AUROC (%) on Task 0")
        ax.set_title("Toy: lambda sensitivity and failure mode")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "toy_lambda_sweep.png"), dpi=150)
        print(f"[toy] wrote plots to {args.out_dir}")

        if wandb_logger:
            wandb_logger.log({
                "toy/confidence_gap": wandb.Image(
                    os.path.join(args.out_dir, "toy_confidence_gap.png")),
                "toy/auroc_over_tasks": wandb.Image(
                    os.path.join(args.out_dir, "toy_auroc_over_tasks.png")),
                "toy/lambda_sweep": wandb.Image(
                    os.path.join(args.out_dir, "toy_lambda_sweep.png")),
            })
    except Exception as e:
        print(f"[toy] plotting skipped: {e}")

    if wandb_logger:
        for t, ag, at in auroc_traj:
            wandb_logger.log({"toy/global_auroc": ag, "toy/tood_auroc": at, "toy/task": t})
        wandb_logger.finish()


if __name__ == "__main__":
    main()
