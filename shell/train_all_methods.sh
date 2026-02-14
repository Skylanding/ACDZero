#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate cage4

ITERATIONS=${1:-500}
STEPS=${2:-100}
PPO_GPU=${3:-1}
DQN_GPU=${4:-2}
MCTS_GPU=${5:-0}
TOP_K=${6:-10}
NUM_SIMULATIONS=${7:-50}

EXPERIMENT_DIR="experiment"
mkdir -p "$EXPERIMENT_DIR"
OUTPUT_DIR="$EXPERIMENT_DIR/unified_training_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "统一训练脚本 - PPO、DQN、MCTS对比"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "训练迭代: $ITERATIONS"
echo "Episode长度: $STEPS"
echo ""
echo "GPU分配:"
echo "  PPO          -> GPU $PPO_GPU"
echo "  DQN          -> GPU $DQN_GPU"
echo "  MCTS         -> GPU $MCTS_GPU"
echo "  MCTS Top-K:  -> $TOP_K"
echo "  MCTS模拟次数: -> $NUM_SIMULATIONS"
echo "=========================================="
echo ""

mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/training_data"
mkdir -p "$OUTPUT_DIR/visualizations"

PPO_LOG="$OUTPUT_DIR/logs/ppo_training.log"
DQN_LOG="$OUTPUT_DIR/logs/dqn_training.log"
MCTS_LOG="$OUTPUT_DIR/logs/hybrid_mcts_ppo_training.log"

LOG_DIR="$OUTPUT_DIR/logs"
DATA_DIR="$OUTPUT_DIR/training_data"
VIS_DIR="$OUTPUT_DIR/visualizations"

mkdir -p "$DATA_DIR"

echo "启动PPO训练（GPU $PPO_GPU）..."
nohup python train_distributed.py \
    --algorithm PPO \
    --gpus "$PPO_GPU" \
    --iterations $ITERATIONS \
    --steps $STEPS \
    --lr 5e-5 \
    --output "$OUTPUT_DIR/checkpoints" \
    > "$PPO_LOG" 2>&1 &
PPO_PID=$!
echo "  PPO进程ID: $PPO_PID"
echo "  日志文件: $PPO_LOG"
echo ""

echo "启动DQN训练（GPU $DQN_GPU）..."
nohup python train_distributed.py \
    --algorithm DQN \
    --gpus "$DQN_GPU" \
    --iterations $ITERATIONS \
    --steps $STEPS \
    --lr 1e-4 \
    --output "$OUTPUT_DIR/checkpoints" \
    > "$DQN_LOG" 2>&1 &
DQN_PID=$!
echo "  DQN进程ID: $DQN_PID"
echo "  日志文件: $DQN_LOG"
echo ""

echo "启动MCTS (HybridMCTS-PPO) 训练（GPU $MCTS_GPU）..."
nohup python -u train_hybrid_mcts_ppo.py \
    --iterations $ITERATIONS \
    --steps $STEPS \
    --output-dir "$OUTPUT_DIR/checkpoints" \
    --gpu "$MCTS_GPU" \
    --top-k "$TOP_K" \
    --num-simulations "$NUM_SIMULATIONS" \
    --deterministic-red \
    --centralized-obs \
    > "$MCTS_LOG" 2>&1 &
MCTS_PID=$!
echo "  MCTS进程ID: $MCTS_PID"
echo "  日志文件: $MCTS_LOG"
echo ""

echo "$PPO_PID" > "$LOG_DIR/ppo.pid"
echo "$DQN_PID" > "$LOG_DIR/dqn.pid"
echo "$MCTS_PID" > "$LOG_DIR/mcts.pid"

echo "=========================================="
echo "所有训练任务已启动！"
echo "=========================================="
echo ""
echo "进程信息:"
echo "  PPO:          PID=$PPO_PID,  GPU=$PPO_GPU"
echo "  DQN:          PID=$DQN_PID,  GPU=$DQN_GPU"
echo "  MCTS:         PID=$MCTS_PID, GPU=$MCTS_GPU"
echo ""
echo "等待训练启动..."
sleep 5
echo ""

PPO_DATA="$OUTPUT_DIR/checkpoints/training_data/ppo_training_data.csv"
DQN_DATA="$OUTPUT_DIR/checkpoints/training_data/dqn_training_data.csv"
MCTS_DATA="$OUTPUT_DIR/training_data/hybrid_mcts_ppo_training_data.csv"

update_visualizations() {
    local has_data=0

    if [ -f "$PPO_DATA" ] && [ $(wc -l < "$PPO_DATA") -gt 2 ]; then
        has_data=1
        python visualize_training.py \
            --data "$PPO_DATA" \
            --output "$VIS_DIR/ppo_curves.png" 2>/dev/null || true
    fi

    if [ -f "$DQN_DATA" ] && [ $(wc -l < "$DQN_DATA") -gt 2 ]; then
        has_data=1
        python visualize_training.py \
            --data "$DQN_DATA" \
            --output "$VIS_DIR/dqn_curves.png" 2>/dev/null || true
    fi

    if [ -f "$MCTS_DATA" ] && [ $(wc -l < "$MCTS_DATA") -gt 2 ]; then
        has_data=1
        python visualize_training.py \
            --data "$MCTS_DATA" \
            --output "$VIS_DIR/hybrid_mcts_ppo_curves.png" 2>/dev/null || true
    fi

    local compare_files=()
    local compare_names=()

    if [ -f "$PPO_DATA" ] && [ $(wc -l < "$PPO_DATA") -gt 2 ]; then
        compare_files+=("$PPO_DATA")
        compare_names+=("PPO")
    fi

    if [ -f "$DQN_DATA" ] && [ $(wc -l < "$DQN_DATA") -gt 2 ]; then
        compare_files+=("$DQN_DATA")
        compare_names+=("DQN")
    fi

    if [ -f "$MCTS_DATA" ] && [ $(wc -l < "$MCTS_DATA") -gt 2 ]; then
        compare_files+=("$MCTS_DATA")
        compare_names+=("MCTS")
    fi

    if [ ${#compare_files[@]} -gt 1 ]; then
        python visualize_training.py \
            --data "${compare_files[@]}" \
            --names "${compare_names[@]}" \
            --output "$VIS_DIR/comparison.png" \
            --compare 2>/dev/null || true
    fi

    if [ $has_data -eq 1 ]; then
        echo "  ✅ 可视化已更新: $VIS_DIR/"
    fi
}

monitor_loop() {
    local iteration=0
    while true; do
        clear
        echo "=========================================="
        echo "训练监控 - PPO、DQN、MCTS对比 - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        echo ""

        echo "进程状态:"
        if [ -f "$LOG_DIR/ppo.pid" ]; then
            PPO_PID=$(cat "$LOG_DIR/ppo.pid" 2>/dev/null)
            if ps -p $PPO_PID > /dev/null 2>&1; then
                echo "  ✅ PPO:  运行中 (PID: $PPO_PID)"
            else
                echo "  ❌ PPO:  已停止"
            fi
        fi

        if [ -f "$LOG_DIR/dqn.pid" ]; then
            DQN_PID=$(cat "$LOG_DIR/dqn.pid" 2>/dev/null)
            if ps -p $DQN_PID > /dev/null 2>&1; then
                echo "  ✅ DQN:  运行中 (PID: $DQN_PID)"
            else
                echo "  ❌ DQN:  已停止"
            fi
        fi

        if [ -f "$LOG_DIR/mcts.pid" ]; then
            MCTS_PID=$(cat "$LOG_DIR/mcts.pid" 2>/dev/null)
            if ps -p $MCTS_PID > /dev/null 2>&1; then
                echo "  ✅ MCTS: 运行中 (PID: $MCTS_PID)"
            else
                echo "  ❌ MCTS: 已停止"
            fi
        fi

        echo ""
        echo "训练进度:"

        if [ -f "$PPO_DATA" ] && [ $(wc -l < "$PPO_DATA") -gt 1 ]; then
            LAST_LINE=$(tail -n 1 "$PPO_DATA" 2>/dev/null)
            if [ ! -z "$LAST_LINE" ] && [[ ! "$LAST_LINE" =~ ^iteration ]]; then
                LAST_ITER=$(echo "$LAST_LINE" | awk -F',' '{print $1}' | tr -d ' ')
                LAST_REWARD=$(echo "$LAST_LINE" | awk -F',' '{print $2}' | tr -d ' ')
                if [[ "$LAST_ITER" =~ ^[0-9]+$ ]] && [[ "$LAST_REWARD" =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
                    printf "  PPO:  迭代 %4s, 奖励: %8.2f\n" "$LAST_ITER" "$LAST_REWARD"
                fi
            fi
        else
            echo "  PPO:  等待数据..."
        fi

        if [ -f "$DQN_DATA" ] && [ $(wc -l < "$DQN_DATA") -gt 1 ]; then
            LAST_LINE=$(tail -n 1 "$DQN_DATA" 2>/dev/null)
            if [ ! -z "$LAST_LINE" ] && [[ ! "$LAST_LINE" =~ ^iteration ]]; then
                LAST_ITER=$(echo "$LAST_LINE" | awk -F',' '{print $1}' | tr -d ' ')
                LAST_REWARD=$(echo "$LAST_LINE" | awk -F',' '{print $2}' | tr -d ' ')
                if [[ "$LAST_ITER" =~ ^[0-9]+$ ]] && [[ "$LAST_REWARD" =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
                    printf "  DQN:  迭代 %4s, 奖励: %8.2f\n" "$LAST_ITER" "$LAST_REWARD"
                fi
            fi
        else
            echo "  DQN:  等待数据..."
        fi

        if [ -f "$MCTS_DATA" ] && [ $(wc -l < "$MCTS_DATA") -gt 1 ]; then
            LAST_LINE=$(tail -n 1 "$MCTS_DATA" 2>/dev/null)
            if [ ! -z "$LAST_LINE" ] && [[ ! "$LAST_LINE" =~ ^iteration ]]; then
                LAST_ITER=$(echo "$LAST_LINE" | awk -F',' '{print $1}' | tr -d ' ')
                LAST_REWARD=$(echo "$LAST_LINE" | awk -F',' '{print $2}' | tr -d ' ')
                if [[ "$LAST_ITER" =~ ^[0-9]+$ ]] && [[ "$LAST_REWARD" =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
                    printf "  MCTS: 迭代 %4s, 奖励: %8.2f\n" "$LAST_ITER" "$LAST_REWARD"
                fi
            fi
        else
            echo "  MCTS: 等待数据..."
        fi

        echo ""
        echo "GPU使用情况:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
            awk -F',' '{printf "  GPU %s: %3s%% 使用, %s/%s 内存\n", $1, $3, $4, $5}' || echo "  (无法获取GPU信息)"

        echo ""
        echo "最新日志 (最后2行):"
        echo "  PPO:"
        tail -n 2 "$PPO_LOG" 2>/dev/null | sed 's/^/    /' || echo "    (无日志)"
        echo "  DQN:"
        tail -n 2 "$DQN_LOG" 2>/dev/null | sed 's/^/    /' || echo "    (无日志)"
        echo "  MCTS:"
        tail -n 2 "$MCTS_LOG" 2>/dev/null | sed 's/^/    /' || echo "    (无日志)"

        if [ $((iteration % 10)) -eq 0 ]; then
            echo ""
            echo "更新可视化图表..."
            update_visualizations
        fi

        iteration=$((iteration + 1))
        sleep 5
    done
}

echo "=========================================="
echo "开始监控训练进度"
echo "=========================================="
echo "提示: 按 Ctrl+C 停止监控（训练会继续在后台运行）"
echo ""

monitor_loop
