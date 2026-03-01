# ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning in Instructional Videos

This repository contains the official PyTorch implementation of **ViterbiPlanNet**, a principled framework that explicitly integrates procedural knowledge into the learning process for procedural planning in instructional videos.

## Overview

Procedural planning aims to predict a sequence of actions that transforms an initial visual state into a desired goal. ViterbiPlanNet introduces a **Differentiable Viterbi Layer (DVL)** that embeds a **Procedural Knowledge Graph (PKG)** directly with the Viterbi decoding algorithm. This design allows the model to learn through graph-based decoding, enabling end-to-end optimization.

The key features of ViterbiPlanNet are:
- **Explicit Procedural Knowledge**: It integrates a Procedural Knowledge Graph (PKG) to guide the planning process.
- **Differentiable Viterbi Layer (DVL)**: It replaces non-differentiable operations in the Viterbi algorithm with smooth relaxations, allowing for end-to-end training.
- **State-of-the-Art Performance**: It achieves state-of-the-art results on CrossTask, COIN, and NIV datasets with significantly fewer parameters than diffusion and LLM-based planners.
- **Sample Efficiency**: The structure-aware training leads to improved sample efficiency and robustness.

## Repository Structure

The repository is organized as follows:

-   `clip/`: Contains the CLIP model implementation, used for feature extraction.
-   `data/`: Holds pre-extracted features for actions and states.
-   `dataset/`: Includes the dataloader for handling the datasets.
-   `models/`: Contains the core implementation of ViterbiPlanNet, including the Differentiable Viterbi Layer and other model components.
-   `scripts/`: Provides shell scripts for training and evaluating the model.
-   `tools/`: Contains various utility scripts, such as argument parsers.
-   `main.py`: The main script for running training and evaluation.
-   `metrics.py`: Implements evaluation metrics such as Success Rate, mAcc, and mIoU.
-   `calculate_results.py`: A script for computing and summarizing results.

## Requirements

The model is implemented in PyTorch. The main dependencies are:

*   Python 3.x
*   PyTorch
*   NumPy
*   TensorBoardX

You can install the required packages using pip:

```bash
pip install torch numpy tensorboardx
```

The project also uses the CLIP model for feature extraction. The necessary files are included in the `clip/` directory.

## Dataset

The project supports three public datasets for procedural planning:
- **CrossTask**
- **COIN**
- **NIV**

The project expects pre-processed features and annotations in a specific format.

1.  **Features**: The `data/` directory is expected to contain pre-extracted features for actions and states (e.g., `coin_action_prompt_features.npy`, `coin_state_prompt_features.npy`).

2.  **Annotations**: The training and validation annotation files are specified via the `--train_json` and `--valid_json` arguments in the run scripts. You will need to download the respective datasets and update the paths in the scripts.

## Training

To train the model, you can use the provided shell scripts in the `scripts/` directory. For example, to train on the COIN dataset, you can run:

```bash
bash scripts/run_coin.sh
```

Before running, you must edit the script to replace the `<PATH>` placeholders for `--train_json` and `--valid_json` with the actual paths to your annotation files.

Example `scripts/run_coin.sh`:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --optimizer 'adam' \
    --lr 0.0009 \
    --dropout 0.2 \
    --batch_size 256 \
    --epochs 500 \
    --max_traj_len 3 \
    --M 2 \
    --aug_range 0 \
    --step_size 40 \
    --lr_decay 0.65 \
    --model_name 'coin' \
    --dataset 'coin' \
    --num_action 778 \
    --num_tasks 180 \
    --img_input_dim 1536 \
    --text_input_dim 768 \
    --embed_dim 128 \
    --train_json '/path/to/your/coin_train.json' \
    --valid_json '/path/to/your/coin_val.json' \
    --saved_path 'checkpoints_coin_viterbi' \
    --seed 7
```

Checkpoints and logs will be saved to the directory specified by `--saved_path` and a `logs/` directory, respectively.

## Evaluation

This work introduces a unified evaluation protocol to ensure fair and consistent comparison between methods. To evaluate a trained model, you can use the `--eval` flag in `main.py`. You will need to provide the path to the trained model and the dataset information.

Example command for evaluation:
```bash
python main.py --eval --dataset coin --saved_path 'checkpoints_coin_viterbi' --seed 7 --max_traj_len 3 --type_of_model viterbi
```
This will load the best model from the specified checkpoint directory and run evaluation on the validation set. The results and predictions will be saved as JSON files.

## Results

The training process saves the best models in the directory specified by `--saved_path`. During evaluation, the model's predictions and performance metrics (Success Rate, mAcc, mIoU) are saved to JSON files in the same directory.