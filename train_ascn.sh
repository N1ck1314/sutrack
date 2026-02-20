#!/bin/bash
# 训练SUTrack-ASCN模型
# 使用ASCNet的RHDWT下采样和CNCM特征增强

python tracking/train.py \
    --script sutrack_ascn \
    --config sutrack_ascn_t224 \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 4 \
    --use_lmdb 0 \
    --use_wandb 0
