#!/bin/bash
# ARTrackV2集成快速验证脚本

echo "======================================"
echo "ARTrackV2集成验证"
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

# 运行验证脚本
echo ""
echo "运行集成验证测试..."
python test_artrackv2_integration.py

echo ""
echo "======================================"
echo "验证完成！"
echo "======================================"
