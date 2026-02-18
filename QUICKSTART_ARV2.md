# SUTrack ARV2 快速开始指南

## 🎯 ARTrackV2 核心特性

ARTrackV2通过以下策略实现3.6x速度提升：
- **Pure Encoder架构**：取消帧内自回归，并行处理所有token
- **Appearance Prompts**：外观演化建模（可学习动态模板）
- **Oriented Masking**：限制外观token注意力路径，防信息泄漏
- **Confidence Token**：IoU预测和置信度估计

## 🚀 快速开始

### 1. 激活环境
```bash
conda activate sutrack
```

### 2. 运行验证测试
```bash
cd /home/nick/code/code.sutrack/SUTrack
python test_artrackv2_integration.py
```

### 3. 启动训练

**单GPU训练：**
```bash
bash train_arv2.sh
```

或者手动运行：
```bash
python tracking/train.py \
    --script sutrack_arv2 \
    --config sutrack_arv2_t224 \
    --save_dir . \
    --mode single
```

**多GPU训练（推荐）：**
```bash
python tracking/train.py \
    --script sutrack_arv2 \
    --config sutrack_arv2_t224 \
    --save_dir . \
    --mode multiple \
    --nproc_per_node 2
```

### 4. 测试推理
```bash
python tracking/test.py sutrack_arv2 sutrack_arv2_t224 \
    --dataset depthtrack \
    --threads 0 \
    --num_gpus 1
```

## 📊 配置说明

### 主要配置文件
- `lib/config/sutrack_arv2/config.py` - 基础配置
- `experiments/sutrack_arv2/sutrack_arv2_t224.yaml` - 实验配置

### 关键超参数

#### ARTrackV2模块配置
```yaml
MODEL:
  ARTRACKV2:
    ENABLE: True                      # 启用ARTrackV2
    NUM_APPEARANCE_TOKENS: 4          # 外观token数量
    NUM_TRAJECTORY_TOKENS: 4          # 轨迹token数量（x1,y1,x2,y2）
    ORIENTED_MASKING: True            # 定向注意力掩码
    APPEARANCE_RECON: True            # 外观重建（训练时）
    MASK_RATIO: 0.5                   # 重建masking比例
    CONFIDENCE_LOSS_WEIGHT: 0.5       # IoU损失权重
    APPEARANCE_RECON_LOSS_WEIGHT: 0.3 # 重建损失权重
```

#### 训练配置
```yaml
TRAIN:
  ARTRACKV2:
    USE_REVERSE_AUGMENTATION: True  # 反向序列增强
    REVERSE_PROB: 0.5               # 反向概率
    IOU_LOSS_WEIGHT: 0.5            # IoU损失权重
    APPEARANCE_RECON_LOSS_WEIGHT: 0.3
```

## 🔧 调优建议

### 1. 外观token数量
- **2个tokens**：速度最快，精度略低
- **4个tokens**：平衡选择（推荐）
- **8个tokens**：精度最高，速度稍慢

### 2. Masking比例
- **0.3**：保守，适合初期训练
- **0.5**：推荐值
- **0.7**：激进，更强的正则化

### 3. 损失权重
- `IOU_LOSS_WEIGHT`: 0.3-0.7（推荐0.5）
- `APPEARANCE_RECON_LOSS_WEIGHT`: 0.1-0.5（推荐0.3）

## 📈 预期效果

根据ARTrackV2论文：
- **速度提升**：2-3x FPS（相比原SUTrack）
- **精度保持**：通过外观演化+跨帧自回归维持精度
- **内存效率**：GPU利用率更高

## 🐛 常见问题

### Q1: RuntimeError: attention mask shape error
**已修复**：OrientedMasking现在返回2D mask [N, N]，PyTorch会自动广播

### Q2: 训练速度慢
- 检查是否启用了ARTrackV2：配置中`ENABLE: True`
- 使用多GPU训练
- 减少外观token数量

### Q3: 精度下降
- 增加外观token数量
- 调整损失权重
- 启用Reverse Augmentation

## 📂 输出路径

```
checkpoints/train/sutrack_arv2/sutrack_arv2_t224/
├── SUTRACK_ep0001.pth.tar
├── SUTRACK_ep0002.pth.tar
└── ...

logs/
└── sutrack_arv2-sutrack_arv2_t224.log

tensorboard/train/sutrack_arv2/sutrack_arv2_t224/train/
└── [tensorboard事件文件]
```

## 📚 相关文档

- `INTEGRATION_SUMMARY_ARV2.md` - 完整集成文档
- [ARTrackV2论文](https://artrackv2.github.io/)
- [ARTrack代码](https://github.com/MIV-XJTU/ARTrack)

## 💡 提示

1. **首次训练**：建议先在小数据集上验证
2. **监控训练**：使用tensorboard查看损失曲线
3. **定期备份**：保存重要checkpoint
4. **对比实验**：与原始SUTrack对比速度和精度

## 🎓 核心原理

### Pure Encoder vs 自回归Decoder
```
传统自回归：x1 → y1 → x2 → y2 (串行)
Pure Encoder：[x1, y1, x2, y2] (并行)
```

### Oriented Masking原理
```
Token布局：[confidence | trajectory | appearance | search]

注意力限制：
appearance ✗→ trajectory  (防信息泄漏)
appearance ✓→ search      (学习外观变化)
appearance ✓→ confidence  (质量评估)
```

---
**集成完成时间**: 2026-02-18  
**状态**: ✅ 可用，已修复attention mask维度问题
