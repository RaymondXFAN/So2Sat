#!/bin/bash
# ══════════════════════════════════════════════════════════════
# ★ 320K完整数据多种子实验脚本 — 审稿配对t检验需要
# 用法: bash code/run_320k_seeds.sh [mode] [epochs]
# 默认: mode=no_stmn, epochs=30
# ══════════════════════════════════════════════════════════════
set -e

MODE=${1:-no_stmn}
EPOCHS=${2:-30}
SEEDS=(42 123 456 789 1024)

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TORCH_HOME=/root/autodl-tmp/.cache/torch

cd /root/autodl-tmp/so2sat_exp

echo "╔══════════════════════════════════════════════════╗"
echo "║  320K完整数据多种子: mode=$MODE, epochs=$EPOCHS"
echo "║  Seeds: ${SEEDS[*]}"
echo "╚══════════════════════════════════════════════════╝"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "★ ★ ★ Seed=$SEED ★ ★ ★"
    /root/miniconda3/bin/python -u code/train.py \
        --mode "$MODE" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        2>&1 | tee -a "320k_${MODE}_seeds.log"
    echo "✓ Seed=$SEED 完成"
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ★ 所有320K多种子实验完成！"
echo "╚══════════════════════════════════════════════════╝"
