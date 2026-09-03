#!/bin/bash
# ══════════════════════════════════════════════════════════════
# ★ 5K多种子实验脚本 — 审稿P0最高优先
# 用法: bash code/run_5k_seeds.sh [mode] [n_train] [epochs]
# 默认: mode=no_stmn, n_train=5000, epochs=50
# ══════════════════════════════════════════════════════════════
set -e

MODE=${1:-no_stmn}
N_TRAIN=${2:-5000}
EPOCHS=${3:-50}
SEEDS=(42 123 456 789 1024 2048 3000 4096 5555 9999)

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TORCH_HOME=/root/autodl-tmp/.cache/torch

cd /root/autodl-tmp/so2sat_exp

START_TIME=$(date +%s)

echo "╔══════════════════════════════════════════════════╗"
echo "║  5K多种子实验: mode=$MODE, n_train=$N_TRAIN, epochs=$EPOCHS"
echo "║  Seeds: ${SEEDS[*]}"
echo "║  开始时间: $(date)"
echo "╚══════════════════════════════════════════════════╝"

i=0
total=${#SEEDS[@]}
for SEED in "${SEEDS[@]}"; do
    i=$((i + 1))
    echo ""
    echo "★ [$i/$total] Seed=$SEED @ $(date)"
    /root/miniconda3/bin/python -u code/train.py \
        --mode "$MODE" \
        --seed "$SEED" \
        --n_train "$N_TRAIN" \
        --epochs "$EPOCHS" \
        2>&1 | tee -a "5k_${MODE}_seeds.log"
    echo "✓ [$i/$total] Seed=$SEED 完成"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ★ 所有5K多种子实验完成！"
echo "║  总耗时: ${ELAPSED}s ($((ELAPSED / 60))min)"
echo "║  结果在各 out/${MODE}_5k_seed*/ 目录下"
echo "╚══════════════════════════════════════════════════╝"
