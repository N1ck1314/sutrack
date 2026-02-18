#!/bin/bash
# SUTrack ARV2 训练启动脚本

echo "======================================"
echo "启动 SUTrack ARV2 训练"
echo "======================================"

# 激活conda环境
echo "正在激活conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sutrack

# 检查环境
echo ""
echo "检查Python环境..."
which python
python --version

# 启动训练
echo ""
echo "======================================"
echo "开始训练 sutrack_arv2..."
echo "======================================"
echo ""

# 单GPU训练
python tracking/train.py \
    --script sutrack_arv2 \
    --config sutrack_arv2_t224 \
    --save_dir . \
    --mode single

# 如果需要多GPU训练，使用下面的命令：
# python tracking/train.py \
#     --script sutrack_arv2 \
#     --config sutrack_arv2_t224 \
#     --save_dir . \
#     --mode multiple \
#     --nproc_per_node 2
