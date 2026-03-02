# [CVPR, 2026] ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning in Instructional Videos

[Luigi Seminara](https://seminaraluigi.altervista.org/)$^{1,\dagger}$, [Davide Moltisanti](https://www.davidemoltisanti.com/research/)$^{2,\*}$, [Antonino Furnari](https://antoninofurnari.github.io/)$^{1,\*}$ 

- (1): University of Catania
- (2): University of Bath
- ($\*$): Equal advising
- ($\dagger$): Work done while visiting University of Bath

arXiv pre-print (Coming very soon!) | Project page (Coming very soon!)

[![Stargazers][stars-shield]][stars-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]

[![LinkedIn][linkedin-shield]][linkedin-url]
[![X][x-shield]][x-url]

![python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

This repository contains the official PyTorch implementation of **ViterbiPlanNet**, a principled framework that explicitly integrates procedural knowledge into the learning process for procedural planning in instructional videos.

## ⁇ Why do you need to use this code?
We establish and open-source a standardized evaluation benchmark, which unifies data splits and evaluation metrics implementations, providing a fair and rigorous comparison of state-of-the-art methods, addressing key inconsistencies in prior work.

## 📢 News
- [February, 2026] We release the ViterbiPlanNet codebase and features.
- [February, 2026] *ViterbiPlanNet* is accepted at **CVPR 2026**.

## 🚧 Coming soon...
- [ ] Code for generating state descriptions
- [ ] Code for extracting state description features [we released pre-extracted state description features in `./data`]
- [ ] Baselines revised code

---
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Data and format](#data-and-format)
- [Contact](#contact)
- [The Evolution of Procedural Planning](#the-evolution-of-procedural-planning)
---

## Overview

Procedural planning aims to predict a sequence of actions that transforms an initial visual state into a desired goal. ViterbiPlanNet introduces a **Differentiable Viterbi Layer (DVL)** that embeds a **Procedural Knowledge Graph (PKG)** directly with the Viterbi decoding algorithm. This design allows the model to learn through graph-based decoding, enabling end-to-end optimization.

The key features of ViterbiPlanNet are:
- **Explicit Procedural Knowledge**: It integrates a Procedural Knowledge Graph (PKG) to guide the planning process.
- **Differentiable Viterbi Layer (DVL)**: It replaces non-differentiable operations in the Viterbi algorithm with smooth relaxations, allowing for end-to-end training.
- **State-of-the-Art Performance**: It achieves state-of-the-art results on CrossTask, COIN, and NIV datasets with significantly fewer parameters than diffusion and LLM-based planners.
- **Sample Efficiency**: The structure-aware training leads to improved sample efficiency and robustness.

## Repository Structure

The repository is organized as follows:

-   `clip/`: Contains the CLIP model implementation, used for the state description features extraction.
-   `data/`: Holds pre-extracted features for actions and state descriptions.
-   `dataset/`: Includes the dataloader for handling the datasets, and contains the datasets and scripts to create splits considering different time horizons.
-   `models/`: Contains the core implementation of ViterbiPlanNet, including the Differentiable Viterbi Layer and other model components.
-   `scripts/`: Provides shell scripts for training and evaluating the model.
-   `tools/`: Contains various utility scripts, such as argument parsers.
-   `main.py`: The main script for running training and evaluation.
-   `metrics.py`: Implements evaluation metrics such as Success Rate, mAcc, and mIoU.
-   `calculate_results.py`: A script for computing and summarizing results.
-   `utils.py`: Contains utility functions for logging and seed setting.

## Datasets
The project supports three public datasets for procedural planning:
- **CrossTask** (see `./dataset/CrossTask`)
- **COIN** (see `./dataset/COIN`)
- **NIV** (see `./dataset/NIV`)

In our project we used the pre-extracted S3D features:

| CrossTask features                                  | COIN features                                 | NIV features                                    | 
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![Download zip](https://custom-icon-badges.demolab.com/badge/-Download-blue?style=for-the-badge&logo=download&logoColor=white "Download zip")](https://drive.google.com/file/d/146VQTl9Pw4VmmlTUt3RVmZtDW8Dcw4uL/view?usp=drive_link) | [![Download zip](https://custom-icon-badges.demolab.com/badge/-Download-blue?style=for-the-badge&logo=download&logoColor=white "Download zip")](https://drive.google.com/file/d/1aQAUQP8jpmZW3iqzVoSeDGJfJOT3-8sv/view?usp=drive_link) | [![Download zip](https://custom-icon-badges.demolab.com/badge/-Download-blue?style=for-the-badge&logo=download&logoColor=white "Download zip")](https://drive.google.com/file/d/1nmsYW0hRGWChCfuTnsvn8lfW2YWmFTfA/view?usp=drive_link) |
| **NOTE**: Unzip and move the folder inside **../dataset/CrossTask** | **NOTE**: Unzip and move the folder inside **../dataset/COIN** | **NOTE**: Unzip and move the folder inside **../dataset/NIV** |


## Training

To train the model, you can use the provided shell scripts in the `scripts/` directory. For example, to train on the CrossTask dataset, you can run:

```bash
bash scripts/run_crosstask.sh
```

Example `scripts/run_crosstask.sh`:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --lr 0.0009 \
    --dropout 0.2 \
    --batch_size 256 \
    --epochs 500 \
    --max_traj_len 3 \                                             # Change to consider a different horizon
    --max_traj_len_test 3 \                                        # During training this must be equal to max_traj_len
    --M 2 \
    --aug_range 0 \
    --step_size 40 \
    --lr_decay 0.65 \
    --model_name 'crosstask' \
    --dataset 'crosstask' \
    --num_action 133 \
    --num_tasks 18 \
    --img_input_dim 1536 \
    --text_input_dim 768 \
    --embed_dim 128 \
    --train_json 'dataset/CrossTask/train_list_3_133.json' \       # Change the training file according to the corresponding horizon 
    --valid_json 'dataset/CrossTask/test_list_3_133.json' \        # Change the test file according to the corresponding horizon  
    --saved_path 'checkpoints' \
    --seed 7
```

Checkpoints and logs will be saved to the directory specified by `--saved_path` and a `logs/` directory, respectively.

## Evaluation

To evaluate a trained model, you can use the `--eval` flag as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size 256 \
    --epochs 500 \
    --max_traj_len 3 \                                             # Change to consider a different horizon
    --max_traj_len_test 3 \                                        # During training this must be equal to max_traj_len
    --M 2 \
    --aug_range 0 \
    --step_size 40 \
    --lr_decay 0.65 \
    --model_name 'crosstask' \
    --dataset 'crosstask' \
    --num_action 133 \
    --num_tasks 18 \
    --img_input_dim 1536 \
    --text_input_dim 768 \
    --embed_dim 128 \
    --train_json 'dataset/CrossTask/train_list_3_133.json' \       # Change the training file according to the corresponding horizon 
    --valid_json 'dataset/CrossTask/test_list_3_133.json' \        # Change the test file according to the corresponding horizon  
    --saved_path 'checkpoints' \
    --seed 7 \
    --eval
```
The results and predictions will be saved as JSON files.

## Data and format
Adding new datasets is possible, and they must respect the following format (example of a sample for horizon 3):
```json
[
    {
        "id": {
            "vid": "GNJfDOoVORM",
            "task": "23521",
            "feature": "./dataset/CrossTask/crosstask_features/processed_data/23521_GNJfDOoVORM.npy",
            "legal_range": [
                [
                    66,     // START TIME
                    72,     // END TIME
                    1       // ACTION ID
                ],
                [
                    73,
                    79,
                    2
                ],
                [
                    80,
                    90,
                    4
                ]
            ],
            "task_id": 0,
            "task_name": "Make Jello Shots",
            "actions": [
                "pour juice",
                "pour jello powder",
                "stir mixture"
            ]
        },
        "instruction_len": 5
    },
...
]
```


## Contact
This repository is created and maintained by [Luigi](https://seminaraluigi.altervista.org/). Technical questions and discussions are encouraged via [GitHub issues](https://github.com/Gigi-G/ViterbiPlanNet/issues), as this allows everyone interested in the work to benefit from the shared information. However, you can always reach us directly via <a href="mailto:luigi.seminara@phd.unict.it?subject=Inquiry about your paper ViterbiPlanNet&cc=antonino.furnari@unict.it;dm2460@bath.ac.uk">email</a>.

---

## The Evolution of Procedural Planning

Procedure planning has its roots in task planning, a domain extensively studied in classical artificial intelligence and robotics. Traditionally, the objective of task planning has been to find a sequence of task-level actions to navigate from a current state to a desired goal state. However, these classical approaches typically relied on hand-defined symbolic planning domains, limiting their application in visually complex, unstructured environments.

As deep learning advanced, the field shifted toward planning directly from high-dimensional pixel observations. This paved the way for modern procedural planning, which attempts to sequence complex, long-horizon tasks in everyday settings using instructional videos.

Within this modern context, two distinct definitions of the problem have emerged:

* **The Visual Goal Formulation [1]:** This definition frames procedure planning around visual start and end states. Given a current visual observation ($v_s$) and a visual goal observation ($v_g$) representing the desired final configuration, the objective is to predict the sequence of high-level actions required to bridge the gap between them.

* **The Natural Language Goal Formulation [2]:** This formulation, termed Visual Planning for Assistance (VPA), generates a sequence of actions based solely on a succinct natural language goal (e.g., "make a shelf") and an untrimmed visual history of the user's progress.

### Our Focus

While the natural language formulation offers interesting applications for virtual assistants [2], our work focuses specifically on the procedure planning problem as defined by [1]. By anchoring our approach in the visual goal formulation, we concentrate on the challenge of learning structured, plannable state and action spaces directly from visual demonstrations, leveraging the conjugate relationships between states and actions to reach a defined visual end-state.

[1] *Chang, Chien-Yi, et al. "Procedure planning in instructional videos." European Conference on Computer Vision. Cham: Springer International Publishing, 2020.*

[2] *Patel, Dhruvesh, et al. "Pretrained language models as visual planners for human assistance." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.*

### Prior Works
***This is the list of paper that I read, feel free to add other works via a pull request.***

- Bi, Jing, Jiebo Luo, and Chenliang Xu. “Procedure Planning in Instructional Videos via Contextual Modeling and Model-Based Policy Learning.” 2021 IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, 15591–600. https://doi.org/10.1109/ICCV48922.2021.01532.
- Chang, Chien-Yi, De-An Huang, Danfei Xu, Ehsan Adeli, Li Fei-Fei, and Juan Carlos Niebles. “Procedure Planning in Instructional Videos.” arXiv:1907.01172. Preprint, arXiv, April 13, 2020. http://arxiv.org/abs/1907.01172.
- Chen, Delong, Willy Chung, Yejin Bang, Ziwei Ji, and Pascale Fung. “WorldPrediction: A Benchmark for High-Level World Modeling and Long-Horizon Procedural Planning.” arXiv:2506.04363. Preprint, arXiv, June 4, 2025. https://doi.org/10.48550/arXiv.2506.04363.
- Fang, Fen, Yun Liu, Ali Koksal, Qianli Xu, and Joo-Hwee Lim. “Masked Diffusion with Task-Awareness for Procedure Planning in Instructional Videos.” arXiv:2309.07409. Preprint, arXiv, September 14, 2023. http://arxiv.org/abs/2309.07409.
- Islam, Md Mohaiminul, Tushar Nagarajan, Huiyu Wang, et al. “Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos.” In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XIX, vol. 15077, edited by Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol. Lecture Notes in Computer Science. Springer, 2024. https://doi.org/10.1007/978-3-031-72655-2_25.
- Li, Zhiheng, Wenjia Geng, Muheng Li, et al. “Skip-Plan: Procedure Planning in Instructional Videos via Condensed Action Space Learning.” 2023 IEEE/CVF International Conference on Computer Vision (ICCV), October 1, 2023, 10263–72. https://doi.org/10.1109/ICCV51070.2023.00945.
- Liu, Jiateng, Sha Li, Zhenhailong Wang, Manling Li, and Heng Ji. A Language-First Approach to Procedure Planning. 2022. https://www.semanticscholar.org/paper/da01ed09a5e57cd5d6f4443e462d200909a7f86e.
- Nagasinghe, Kumaranage Ravindu Yasas, Honglu Zhou, Malitha Gunawardhana, Martin Renqiang Min, Daniel Harari, and Muhammad Haris Khan. “Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning of Instructional Videos.” IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, 2024, 18816–26. https://doi.org/10.1109/CVPR52733.2024.01780.
- Niu, Yulei, Wenliang Guo, Long Chen, Xudong Lin, and Shih-Fu Chang. “SCHEMA: State CHangEs MAtter for Procedure Planning in Instructional Videos.” The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, 2024. https://openreview.net/forum?id=abL5LJNZ49.
- Shi, Lei, Paul Bürkner, and Andreas Bulling. “ActionDiffusion: An Action-Aware Diffusion Model for Procedure Planning in Instructional Videos.” arXiv:2403.08591. Preprint, arXiv, July 20, 2024. http://arxiv.org/abs/2403.08591.
- Wang, Hanlin, Yilu Wu, Sheng Guo, and Limin Wang. “PDPP: Projected Diffusion for Procedure Planning in Instructional Videos.” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, 14836–45. https://doi.org/10.1109/CVPR52729.2023.01425.
- Xu, Yi, Chengzu Li, Han Zhou, et al. “Visual Planning: Let’s Think Only with Images.” arXiv:2505.11409. Preprint, arXiv, May 16, 2025. https://doi.org/10.48550/arXiv.2505.11409.
- Yang, Dejie, Zijing Zhao, and Yang Liu. “PlanLLM: Video Procedure Planning with Refinable Large Language Models.” arXiv:2412.19139. Preprint, arXiv, January 7, 2025. https://doi.org/10.48550/arXiv.2412.19139.
- Zare, Ali, Yulei Niu, Hammad Ayyubi, and Shih-fu Chang. “RAP: Retrieval-Augmented Planner for Adaptive Procedure Planning in Instructional Videos.” arXiv:2403.18600. Preprint, arXiv, September 25, 2024. https://doi.org/10.48550/arXiv.2403.18600.
- Zhao, He, Isma Hadji, Nikita Dvornik, Konstantinos G. Derpanis, Richard P. Wildes, and Allan D. Jepson. “P3 IV: Probabilistic Procedure Planning from Instructional Videos with Weak Supervision.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, 2928–38. https://doi.org/10.1109/CVPR52688.2022.00295.
- Zhou, Yufan, Zhaobo Qi, Lingshuai Lin, et al. “Masked Temporal Interpolation Diffusion for Procedure Planning in Instructional Videos.” arXiv:2507.03393. ICLR 2025, July 4, 2025. https://doi.org/10.48550/arXiv.2507.03393.

[forks-shield]: https://img.shields.io/github/forks/Gigi-G/ViterbiPlanNet.svg?style=for-the-badge
[forks-url]: https://github.com/Gigi-G/ViterbiPlanNet/network/members
[stars-shield]: https://img.shields.io/github/stars/Gigi-G/ViterbiPlanNet.svg?style=for-the-badge
[stars-url]: https://github.com/Gigi-G/ViterbiPlanNet/stargazers
[issues-shield]: https://img.shields.io/github/issues/Gigi-G/ViterbiPlanNet.svg?style=for-the-badge
[issues-url]: https://github.com/Gigi-G/ViterbiPlanNet/issues
[license-shield]: https://img.shields.io/github/license/fGigi-G/ViterbiPlanNet.svg?style=for-the-badge
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://it.linkedin.com/in/luigi-seminara-3bb2a2204
[x-shield]: https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white
[x-url]: https://x.com/Gigii_Gii
