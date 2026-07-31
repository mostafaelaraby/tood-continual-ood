# TOOD: Task-Aware Out-of-Distribution Score Calibration for Continual Learners

This repository contains the implementation accompanying
[the paper](https://cdn.fourwaves.com/static/media/formdata/ed6c71eb-f6ae-4710-944d-c458f6f289a9/fbd3392c-3314-4168-9007-e18130a6747e.pdf). It combines Avalanche continual-learning benchmarks
with OpenOOD post-hoc detectors to study OOD performance throughout a
class-incremental learning stream.

The camera-ready release focuses on experiment execution and reproducibility.
Training, checkpoint evaluation, metric recording, and the scientific
mechanism analyses are included. Standalone scripts for downloading completed
W&B runs or reformatting results into paper-specific figures and LaTeX tables
are intentionally excluded.

## Installation

### Local system

Linux and macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r docker/requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r docker/requirements.txt
```

The project is run from the repository root; its vendored `openood/` package
is imported directly from that directory. The dependency manifest currently
lives at `docker/requirements.txt` (there is no root-level `requirements.txt`).

## Data and Checkpoints

### Local system

Set `scenario.dataset_dir` in the selected CL config to the directory where
Avalanche should store the ID dataset. OOD data is configured separately in
the matching `*_ood.yml` file; update its `data_dir` and `imglist_pth` values
to match your local OpenOOD data. This repository does not include dataset or
checkpoint download scripts.

## Overview

The pipeline trains a model on a stream of tasks, such as SplitCIFAR10, and
evaluates in-distribution accuracy, forgetting, and OOD AUROC, FPR@95, and
AUPR after each task.

Key capabilities:

- Sets up multiple CL strategies, including Naive, EWC, Replay, iCaRL, BiC,
  DER, and FOSTER.
- Configures ResNet, MobileNet, and ViT model families.
- Runs a CL training loop (multi-GPU via `torchrun`; optional mixed-precision AMP).
- Evaluates OOD detection against near- and far-OOD datasets after each task.
- Measures OOD-performance deterioration and representational alignment over time.
- Records results locally and logs experiment metrics to Weights & Biases.

The pipeline is split into two independent entry points:

| Script        | Responsibility                                                                 |
|---------------|--------------------------------------------------------------------------------|
| `train.py`    | Train each CL task sequentially; write `.pth` checkpoints and `.json` context. |
| `evaluate.py` | Load pretrained checkpoints and run the full OOD evaluation + final reports.  |

Both scripts share configuration, model, benchmark, and optimizer factories via `core/setup.py`.

### Repository Layout

| Path | Purpose |
|------|---------|
| `train.py` | Sequential continual-learning training and checkpoint creation. |
| `evaluate.py` | Checkpoint-only ID and OOD evaluation. |
| `core/`, `managers/`, `cl_extensions/`, `utils/` | Experiment setup and runtime implementation. |
| `analysis/recorders.py`, `analysis/plotting.py` | Evaluation-integrated metric recording and optional diagnostic plots. |
| `analysis/toy_confidence_gap.py` | Controlled confidence-gap experiment. |
| `analysis/feature_tood_analog.py` | Feature-space TOOD analogue experiment. |
| `configs/` | Dataset, network, postprocessor, pipeline, and CL configurations. |
| `openood/` | Vendored OpenOOD components required by the experiments. |

### Paper-to-Code Map

| Paper component | Implementation |
|---|---|
| OOD forgetting, Eq. (2) | [`RecorderManager._analyze_ood_deterioration`](managers/recorder_manager.py#L551) |
| Average incremental AUROC, Eq. (3) | [`RecorderManager.log_ood_summary`](managers/recorder_manager.py#L273) and [`_log_final_ood_metrics`](managers/recorder_manager.py#L504) |
| Per-task energy, Eq. (4) | [`compute_per_task_energy`](managers/ood_manager.py#L926) |
| Mean Shift / Robust Anchor, Eqs. (5–6) | [`normalize_energies`](managers/ood_manager.py#L957) |
| Final score and optional margin, Eqs. (7–8) | [`compute_tood_score`](managers/ood_manager.py#L1009) |
| Algorithm 1 calibration and inference | [`CalibratePerTaskOODPostprocessorManager`](managers/ood_manager.py#L1129) |
| Classification forgetting, Eq. (1) | [`compute_true_stream_forgetting`](utils/helpers.py#L933) |
| Confidence-gap toy experiment, Appendix D.3 | [`analysis/toy_confidence_gap.py`](analysis/toy_confidence_gap.py) |
| Feature-space analogue / manifold crowding | [`analysis/feature_tood_analog.py`](analysis/feature_tood_analog.py) |
| CKA representation analysis | [`CKARecorder`](analysis/recorders.py#L17) |

### Reported Metrics and Artifacts

Evaluation records the quantities needed to reproduce the reported results:

- **ID accuracy** (per task, last, average incremental) and **Stream forgetting** — from the Avalanche eval loop.
- **OOD detection per task**: AUROC, FPR@95, AUPR (near-OOD and far-OOD).
- **AUROC deterioration** (peak − final) per task and averaged — written to `{exp_name}_ood_deterioration.csv` and plotted as `deterioration_per_task_{ood_type}.png`.
- **CKA trajectory**: representation drift from origin, adjacent-task similarity, and OOD representation drift — written to `{exp_name}_cka_results.csv` and plotted as `cka_trajectory.png` + `cka_vs_auroc_{ood_type}.png`.

CSV files are always generated. Set `scenario.make_plots=True` to generate the
evaluation-integrated PNG diagnostics. Final summary metrics are also logged
under the W&B `final/` namespace.

### Configuration Order

Every command uses the same six YAML files, in this order:

| Configuration | Selects | CIFAR-10 example |
|---|---|---|
| ID dataset | Dataset shape and preprocessing | `configs/datasets/cifar10/cifar10.yml` |
| OOD dataset | Near- and far-OOD image lists | `configs/datasets/cifar10/cifar10_ood.yml` |
| OpenOOD network | OpenOOD-compatible network metadata | `configs/networks/resnet18_32x32.yml` |
| Pipeline | Evaluation pipeline and recorder | `configs/pipelines/test/test_ood.yml` |
| Postprocessor | Base OOD score; Energy is `ebo` | `configs/postprocessors/ebo.yml` |
| CL experiment | Strategy, model, schedule, paths, and seed | `configs/cl/cifar10/icarl.yml` |

Files are merged from left to right; command-line overrides have final
precedence. Use the identical ordered set for training and evaluation.

## Path A — Full Reproduction (Train and Evaluate)

The example below runs the paper's CIFAR-10 iCaRL experiment with robust-anchor
TOOD.

### Step 1. Train

```bash
python train.py --config \
  configs/datasets/cifar10/cifar10.yml \
  configs/datasets/cifar10/cifar10_ood.yml \
  configs/networks/resnet18_32x32.yml \
  configs/pipelines/test/test_ood.yml \
  configs/postprocessors/ebo.yml \
  configs/cl/cifar10/icarl.yml \
  --exp_name=cifar10_resnet32_ICaRL_OOD_ebo \
  --calibrate_ood_scores.enabled=True \
  --calibrate_ood_scores.method=robust_anchor \
  --calibrate_ood_scores.buffer_size=200 \
  --calibrate_ood_scores.margin_lambda=0.5
```

For four-GPU training, replace `python train.py` with
`torchrun --nproc_per_node=4 train.py`. Detector and calibration settings do
not change training; they are included so the exact same command options can
be reused for evaluation.

`train.py` writes two artifacts per task to `scenario.ckpt_dir`:

- `{base_name}_{task_id}.pth` — model weights and strategy state.
- `classes_{task_id}.json` — class IDs seen in that task.

### Step 2. Evaluate

```bash
python evaluate.py --config \
  configs/datasets/cifar10/cifar10.yml \
  configs/datasets/cifar10/cifar10_ood.yml \
  configs/networks/resnet18_32x32.yml \
  configs/pipelines/test/test_ood.yml \
  configs/postprocessors/ebo.yml \
  configs/cl/cifar10/icarl.yml \
  --exp_name=cifar10_resnet32_ICaRL_OOD_ebo \
  --calibrate_ood_scores.enabled=True \
  --calibrate_ood_scores.method=robust_anchor \
  --calibrate_ood_scores.buffer_size=200 \
  --calibrate_ood_scores.margin_lambda=0.5
```

`evaluate.py` reads the saved checkpoints and writes results under
`{scenario.ckpt_dir}/{postprocessor.name}/{calibration-signature}/`.

Select the paper variant with these overrides:

| Variant | `enabled` | `method` |
|---|---:|---|
| Uncalibrated Energy | `False` | ignored |
| TOOD Mean Shift | `True` | `mean_shift` |
| TOOD Robust Anchor | `True` | `robust_anchor` |

The paper uses `margin_lambda=0.5` and total calibration-buffer sizes of 200
for CIFAR-10 and 700 for CIFAR-100.

## Path B — Evaluate Pretrained Checkpoints

Use this when the per-task checkpoints already exist.

1. Place the checkpoint directory at the path specified by `scenario.ckpt_dir` in the config.
2. Run the complete Step 2 command above with the same six YAML files and
   overrides that produced the checkpoints. Override `scenario.ckpt_dir` if
   they are stored elsewhere.

Outputs:

- `{exp_name}_ood_deterioration.csv` — per-task peak, final, and deterioration
  values for AUROC, FPR@95, and AUPR.
- `{exp_name}_cka_results.csv` — representation-similarity measurements.
- `eval_state.pkl` — resumable evaluation state.
- PNG diagnostics when `scenario.make_plots=True`.
- A W&B run containing per-step and final summary metrics.

All local evaluation artifacts are written under:

```text
{scenario.ckpt_dir}/{postprocessor.name}/{calibration-signature}/
```

## Weights & Biases

W&B is used only for logging active experiments. Set `WANDB_PROJECT` to choose
the project name and authenticate with `wandb login` or `WANDB_API_KEY` before
running. The release does not include an API exporter for downloading or
post-processing completed runs.

## Scientific Analysis Scripts

Two standalone scripts are retained because they reproduce mechanism
experiments from the paper:

```bash
python analysis/toy_confidence_gap.py
python analysis/feature_tood_analog.py --config [the same YAML configuration set]
```

Each script documents its additional arguments through `--help`.

## Overriding Config Values

Append `--KEY=VALUE` overrides after the configuration files. The leading
`--` is required by the local OpenOOD config parser:

```bash
python train.py --config [six YAML files] --strategy.train_epochs=50
python evaluate.py --config [six YAML files] --scenario.ckpt_dir=/tmp/ckpts
```

## Validation

Run the release tests from the repository root:

```bash
python -m pytest -q Tests
```

## Troubleshooting

| Symptom                                   | Likely cause                                  | Fix                                                 |
|-------------------------------------------|-----------------------------------------------|-----------------------------------------------------|
| `AssertionError: exp_name must contain '_OOD_'` | Config `exp_name` missing separator     | Add `_OOD_` to the `exp_name` field in the YAML.    |
| `FileNotFoundError: ..._0.pth`            | Checkpoint not found at `scenario.ckpt_dir`   | Run `train.py` first, or fix `scenario.ckpt_dir`.   |
| `FileNotFoundError: classes_0.json`       | Context file missing (pre-refactor ckpt dir)  | Rerun `train.py` to regenerate the `classes_*.json`.|
| CUDA out-of-memory during evaluation      | Using an old combined script instead of `evaluate.py` | Use `evaluate.py` — it does not load optimizer state. |
| Diagnostic PNGs are not generated         | `scenario.make_plots=False`                   | Override with `--scenario.make_plots=True`.         |
| W&B not logging                           | W&B authentication is missing                | Set `WANDB_API_KEY` or run `wandb login`.            |
