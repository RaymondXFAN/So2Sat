#!/bin/bash
# ══════════════════════════════════════════════════════════════
# ★ v3统一代码路径 5K门值实验 — 审稿P2
# full模式(3-分支Ada-MBA含门控) + 5K子集 + 多种子
# 目的: 在统一代码下复现5K门值实验，提供gate_stats数据
# 用法: bash code/run_5k_gate_seeds.sh [epochs]
# ══════════════════════════════════════════════════════════════
set -e

EPOCHS=${1:-50}
SEEDS=(42 123 456 789 1024)

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TORCH_HOME=/root/autodl-tmp/.cache/torch

cd /root/autodl-tmp/so2sat_exp

START_TIME=$(date +%s)

echo "╔══════════════════════════════════════════════════╗"
echo "║  5K门值实验: full模式, 5 seeds, $EPOCHS epochs"
echo "║  Seeds: ${SEEDS[*]}"
echo "╚══════════════════════════════════════════════════╝"

i=0
total=${#SEEDS[@]}
for SEED in "${SEEDS[@]}"; do
    i=$((i + 1))
    echo ""
    echo "★ [$i/$total] Seed=$SEED @ $(date)"
    /root/miniconda3/bin/python -u code/train.py \
        --mode full \
        --seed "$SEED" \
        --n_train 5000 \
        --epochs "$EPOCHS" \
        2>&1 | tee -a "5k_full_seeds.log"
    echo "✓ [$i/$total] Seed=$SEED 完成"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ★ 所有5K门值实验完成！"
echo "║  总耗时: ${ELAPSED}s ($((ELAPSED / 60))min)"
echo "╚══════════════════════════════════════════════════╝"
