# Step Models

This directory contains the step-level models used in ViterbiPlanNet experiments.

## Overview

Implemented architectures:

- `MLP_based`
- `Diffusion_based`
- `Diffusion_based_with_task`

The two diffusion variants are inspired by the KEPP step-model formulation:

> Nagasinghe, Kumaranage Ravindu Yasas, Honglu Zhou, Malitha Gunawardhana, Martin Renqiang Min, Daniel Harari, and Muhammad Haris Khan. *Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning of Instructional Videos.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, 18816–18826. https://doi.org/10.1109/CVPR52733.2024.01780.


## Folder structure

```text
step_models/
├── MLP_based/
├── Diffusion_based/
├── Diffusion_based_with_task/
├── calculate_step_results.py
├── calculate_step_best_results.py
└── README.md
```

## Architecture details

### 1) MLP_based

```text
MLP_based/
├── main.py
├── models/
│   ├── step_model.py
│   ├── state_encoder.py
│   └── utils.py
├── dataset/
│   └── dataloader.py
├── tools/parser.py
├── scripts/
│   ├── run_coin.sh
│   ├── run_crosstask.sh
│   └── run_niv.sh
├── sbatch_scripts/
├── checkpoints/
└── logs/
```

- Entry point: `main.py`
- Core network: `models/step_model.py`
- Standard training/evaluation scripts are under `scripts/`.

### 2) Diffusion_based (without task)

```text
Diffusion_based/
├── main.py
├── inference.py
├── model/
│   ├── temporal.py
│   ├── diffusion.py
│   └── helpers.py
├── dataloader/dataloader.py
├── utils/args.py
├── data_lists/
├── steps_list/
├── scripts/
│   ├── run_coin.sh
│   ├── run_crosstask.sh
│   └── run_niv.sh
├── sbatch_scripts/
├── checkpoints/
└── logs/
```

Pipeline:

1. `main.py` trains diffusion step planner.
2. `inference.py` generates step predictions and writes output JSONs.

### 3) Diffusion_based_with_task (with task)

```text
Diffusion_based_with_task/
├── main.py
├── train_mlp.py
├── predict_tasks.py
├── inference.py
├── model/
│   ├── temporal.py
│   ├── diffusion.py
│   └── helpers.py
├── dataloader/dataloader.py
├── utils/args.py
├── data_lists/
├── steps_list/
├── scripts/
│   ├── run_coin.sh
│   ├── run_crosstask.sh
│   └── run_niv.sh
├── sbatch_scripts/
├── checkpoints/
├── checkpoints_mlp/
├── logs/
└── logs_mlp/
```

Pipeline:

1. `main.py`: train diffusion backbone.
2. `train_mlp.py`: train task classifier.
3. `predict_tasks.py`: predict task IDs for samples.
4. `inference.py`: run task-conditioned diffusion inference.

## Quick run examples

```bash
cd step_models/MLP_based && bash scripts/run_niv.sh
cd ../Diffusion_based && bash scripts/run_niv.sh
cd ../Diffusion_based_with_task && bash scripts/run_niv.sh
```

## Results tables

### CrossTask

| Horizon | Method | State Acc | first Acc | last Acc | Max State Acc | Max first Acc | Max last Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| T = 3 | MLP | 62.94% ± 0.22% | 63.27% ± 0.40% | 62.60% ± 0.33% | 63.12% | 63.76% | 62.49% |
| T = 3 | Diffusion | 67.18% ± 0.48% | 68.32% ± 0.38% | 66.04% ± 0.63% | 67.72% | 68.59% | 66.85% |
| T = 3 | Diffusion with task | 67.13% ± 0.30% | 68.45% ± 0.47% | 65.81% ± 0.20% | 67.48% | 68.88% | 66.08% |
| T = 4 | MLP | 62.80% ± 0.19% | 63.22% ± 0.43% | 62.39% ± 0.27% | 63.01% | 63.76% | 62.27% |
| T = 4 | Diffusion | 67.05% ± 0.73% | 68.48% ± 0.90% | 65.63% ± 0.50% | 67.58% | 69.04% | 66.11% |
| T = 4 | Diffusion with task | 66.91% ± 0.60% | 68.31% ± 0.59% | 65.52% ± 0.78% | 67.38% | 68.86% | 65.90% |
| T = 5 | MLP | 63.93% ± 0.18% | 63.23% ± 0.25% | 64.63% ± 0.53% | 64.03% | 63.07% | 64.99% |
| T = 5 | Diffusion | 67.61% ± 0.23% | 69.34% ± 0.64% | 65.88% ± 0.28% | 67.75% | 69.85% | 65.66% |
| T = 5 | Diffusion with task | 66.86% ± 0.61% | 68.31% ± 0.34% | 65.41% ± 0.90% | 67.36% | 68.40% | 66.33% |
| T = 6 | MLP | 62.98% ± 0.14% | 62.75% ± 0.28% | 63.21% ± 0.24% | 63.15% | 63.08% | 63.22% |
| T = 6 | Diffusion | 65.99% ± 0.51% | 68.33% ± 1.03% | 63.64% ± 0.87% | 66.33% | 68.75% | 63.91% |
| T = 6 | Diffusion with task | 66.18% ± 0.54% | 68.24% ± 0.64% | 64.12% ± 0.72% | 66.81% | 68.63% | 64.99% |

### COIN

| Horizon | Method | State Acc | first Acc | last Acc | Max State Acc | Max first Acc | Max last Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| T = 3 | MLP | 46.66% ± 0.25% | 46.66% ± 0.25% | 46.94% ± 0.29% | 46.86% | 46.49% | 47.24% |
| T = 3 | Diffusion | 40.98% ± 1.00% | 40.73% ± 0.91% | 41.23% ± 0.98% | 41.65% | 41.67% | 41.63% |
| T = 3 | Diffusion with task | 41.86% ± 0.82% | 41.10% ± 1.06% | 42.62% ± 0.92% | 42.80% | 41.94% | 43.66% |
| T = 4 | MLP | 46.14% ± 0.34% | 45.81% ± 0.15% | 46.46% ± 0.54% | 46.36% | 45.93% | 46.80% |
| T = 4 | Diffusion | 39.79% ± 3.47% | 40.19% ± 3.35% | 39.39% ± 3.67% | 41.18% | 41.02% | 41.34% |
| T = 4 | Diffusion with task | 40.52% ± 0.90% | 40.65% ± 0.58% | 40.39% ± 1.44% | 41.51% | 41.17% | 41.85% |
| T = 5 | MLP | 43.72% ± 0.33% | 43.51% ± 0.64% | 43.93% ± 0.46% | 44.04% | 44.20% | 43.88% |
| T = 5 | Diffusion | 38.21% ± 0.87% | 39.14% ± 1.30% | 37.28% ± 0.82% | 38.89% | 39.67% | 38.10% |
| T = 5 | Diffusion with task | 41.77% ± 2.15% | 41.79% ± 1.92% | 41.74% ± 2.58% | 43.08% | 42.98% | 43.17% |
| T = 6 | MLP | 40.15% ± 0.17% | 40.37% ± 0.81% | 39.94% ± 0.59% | 40.28% | 40.90% | 39.67% |
| T = 6 | Diffusion | 36.49% ± 1.25% | 37.80% ± 1.30% | 35.19% ± 1.33% | 37.66% | 38.96% | 36.36% |
| T = 6 | Diffusion with task | 38.01% ± 1.85% | 38.82% ± 1.82% | 37.20% ± 2.03% | 39.46% | 40.18% | 38.73% |

### NIV

| Horizon | Method | State Acc | first Acc | last Acc | Max State Acc | Max first Acc | Max last Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| T = 3 | MLP | 44.20% ± 0.90% | 34.59% ± 1.64% | 53.82% ± 0.94% | 44.91% | 36.27% | 53.54% |
| T = 3 | Diffusion | 42.37% ± 10.00% | 40.59% ± 11.56% | 44.15% ± 8.44% | 47.78% | 47.04% | 48.52% |
| T = 3 | Diffusion with task | 47.22% ± 2.70% | 45.41% ± 5.19% | 49.04% ± 0.89% | 48.70% | 48.52% | 48.89% |
| T = 4 | MLP | 42.81% ± 1.05% | 44.39% ± 2.02% | 41.23% ± 1.67% | 43.42% | 46.93% | 39.91% |
| T = 4 | Diffusion | 45.66% ± 2.89% | 45.88% ± 2.89% | 45.44% ± 2.98% | 47.59% | 47.37% | 47.81% |
| T = 4 | Diffusion with task | 45.31% ± 2.72% | 43.86% ± 1.84% | 46.75% ± 4.31% | 48.03% | 44.30% | 51.75% |
| T = 5 | MLP | 42.41% ± 0.86% | 43.21% ± 2.47% | 41.60% ± 0.64% | 43.05% | 45.45% | 40.64% |
| T = 5 | Diffusion | 50.11% ± 1.50% | 49.84% ± 2.03% | 50.37% ± 3.85% | 51.34% | 50.27% | 52.41% |
| T = 5 | Diffusion with task | 47.17% ± 2.67% | 46.95% ± 3.32% | 47.38% ± 2.57% | 50.27% | 51.87% | 48.66% |
| T = 6 | MLP | 43.72% ± 1.22% | 46.49% ± 0.81% | 40.95% ± 2.70% | 44.59% | 46.62% | 42.57% |
| T = 6 | Diffusion | 51.49% ± 3.65% | 53.92% ± 3.78% | 49.05% ± 4.05% | 55.74% | 58.78% | 52.70% |
| T = 6 | Diffusion with task | 52.70% ± 1.55% | 53.11% ± 3.24% | 52.30% ± 2.18% | 54.73% | 56.08% | 53.38% |

### Average Max State Acc across datasets (grouped by Horizon)

| Horizon | Diffusion with task | Diffusion | MLP |
|---|---:|---:|---:|
| T=3 | **52.99%** | *52.38%* | 51.63% |
| T=4 | **52.31%** | *52.12%* | 50.93% |
| T=5 | **53.57%** | *52.66%* | 50.37% |
| T=6 | **53.67%** | *53.24%* | 49.34% |

