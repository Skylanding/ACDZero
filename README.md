<h3 align="center">
<b>ACDZero: Graph-Embedding-Based Tree Search for Mastering Automated Cyber Defense</b>
<br>
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2601.02196">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://github.com/Skylanding/ACDZero">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
</p>

This repository contains training pipelines and utilities for CAGE Challenge 4 (CybORG-based multi-agent cyber defense) with our ACDZero method.

## CAGE4 Environment

CAGE4 is a multi-agent cyber defense simulation environment built on `CybORG`.

- **Blue agents** perform defensive actions (monitor, analyse, remove, restore, traffic control).
- **Red agents** simulate attacker behavior over multiple steps.
- **Green agents** represent background/benign activity.
- Training is episodic, with observations + rewards per step, and supports methods like `PPO`, `DQN`, `MCTS`, and hybrid variants.

## Project Structure

- `train_distributed.py`: RLlib distributed training entry (`PPO` / `DQN`).
- `train_contextual_bandit.py`: Tabular contextual bandit baseline.
- `train_hybrid_mcts_ppo.py`: Hybrid MuZero-style MCTS + PPO training.
- `train_mcts.py`: MCTS baseline training.
- `training/`: Shared helpers (environment setup, path helpers, observation casting).
- `agents/`: Custom agents and wrappers used by training scripts.
- `shell/`: Utility scripts for launching/monitoring multi-run experiments.
- `graph/`: Graph-based training variant and related code.
- `CybORG/`: Simulator and environment code.
- `experiment/`, `checkpoints/`, `training_data/`: Generated outputs.

## Conda Setup (cage4)

```bash
cd /home/ubuntu/ACDZero
conda create -n cage4 python=3.10 -y
conda activate cage4
pip install -r Requirements.txt
```

If you also use modules under `baseline-train/`, install:

```bash
pip install -r baseline-train/requirements.txt
```

## Common Commands

```bash
python train_distributed.py --algorithm PPO --gpus 0 --iterations 100 --steps 100
python train_contextual_bandit.py --iterations 100 --steps 100
python train_hybrid_mcts_ppo.py --iterations 100 --steps 100 --top-k 10
python train_mcts.py --iterations 100 --steps 100 --simulations 50
```
