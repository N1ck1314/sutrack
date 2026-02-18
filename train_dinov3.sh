#!/bin/bash
# DINOv3 ConvNeXt-Tiny 训练脚本

# ====================================
# 配置参数
# ====================================

# 选择训练配置（二选一）
# CONFIG="dinov3_convnext_tiny_224_frozen"  # 快速验证版（冻结encoder）
CONFIG="dinov3_convnext_tiny_224"           # 完整训练版（端到端）

# GPU 设置
NUM_GPUS=1        # GPU 数量（根据你的机器调整）
MODE="single"     # "single" 或 "multiple"

# 保存路径
SAVE_DIR="./checkpoints/train/sutrack_dinov3/${CONFIG}"

# ====================================
# 开始训练
# ====================================

echo "=================================="
echo "DINOv3 ConvNeXt-Tiny 训练"
echo "=================================="
echo "配置文件: experiments/sutrack_dinov3/${CONFIG}.yaml"
echo "保存路径: ${SAVE_DIR}"
echo "GPU模式: ${MODE} (${NUM_GPUS} GPUs)"
echo "=================================="

# 激活环境
source ~/anaconda3/bin/activate sutrack

# 训练命令
if [ "$MODE" == "single" ]; then
    # 单 GPU 训练
    python tracking/train.py \
        --script sutrack_dinov3 \
        --config ${CONFIG} \
        --save_dir ${SAVE_DIR} \
        --mode single \
        --use_lmdb 0
else
    # 多 GPU 训练
    python tracking/train.py \
        --script sutrack_dinov3 \
        --config ${CONFIG} \
        --save_dir ${SAVE_DIR} \
        --mode multiple \
        --nproc_per_node ${NUM_GPUS} \
        --use_lmdb 0
fi

echo ""
echo "=================================="
echo "训练完成！"
echo "检查点保存在: ${SAVE_DIR}"
echo "日志文件: ${SAVE_DIR}/logs/sutrack_dinov3-${CONFIG}.log"
echo "=================================="
