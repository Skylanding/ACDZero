#!/usr/bin/env bash
set -e

# Simple wrapper to evaluate MCTS checkpoints using evaluation.py
# Usage: ./eval_mcts.sh [run_name] [max_eps] [workers]
# Defaults: run_name=run_mcts, max_eps=10, workers=4

RUN_NAME=${1:-run_mcts}
MAX_EPS=${2:-10}
WORKERS=${3:-4}

cd "$(dirname "$0")"

mkdir -p weights

# Link checkpoints to weights expected by submission.py
for i in 0 1 2 3 4; do
  SRC="$(pwd)/checkpoints/${RUN_NAME}-${i}_checkpoint.pt"
  DST="weights/gnn_ppo-${i}.pt"
  if [ ! -f "$SRC" ]; then
    echo "Missing checkpoint: ${SRC}"
    echo "Available checkpoints:"
    ls checkpoints
    exit 1
  fi
  ln -sf "$SRC" "$DST"
done

echo "Evaluating checkpoints for ${RUN_NAME} with ${MAX_EPS} episodes, ${WORKERS} workers"
python evaluation.py --distribute "${WORKERS}" --max-eps "${MAX_EPS}"

