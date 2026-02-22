# SUTrack-Select 快速入门

## 简介

SUTrack-Select 是基于 SUTrack 的改进版本，实现了**选择性深度集成 (Selective Depth Integration)**。借鉴 SGLA 的层跳过机制，该模块能够智能地决定每一层是否需要使用深度信息，从而提升推理速度。

## 核心特性

- 🎯 **智能选择**：基于 RGB 特征预测每层是否需要深度信息
- ⚡ **效率优化**：推理时硬跳过不必要的深度处理
- 🔄 **训练友好**：训练时软跳过，保持可微分性
- 📊 **统计分析**：支持深度使用率统计

## 快速测试

### 1. 运行测试脚本

```bash
cd /home/nick/code/code.sutrack/SUTrack
python test_sutrack_select.py
```

**测试内容**：
- ✓ 模型创建
- ✓ 前向传播
- ✓ 选择损失计算
- ✓ 深度使用统计

### 2. 预期输出

```
============================================================
SUTrack-Select 模型测试
============================================================
============================================================
测试 1: 模型创建
============================================================
✓ 成功加载配置文件: experiments/sutrack_select/sutrack_select_t224.yaml
✓ 模型创建成功
✓ 选择性深度集成已启用
  - Encoder use_selective_depth: True
  - Selective depth module: SelectiveDepthIntegration

============================================================
测试 2: 前向传播
============================================================
输入尺寸:
  - Template: torch.Size([2, 3, 112, 112])
  - Search: torch.Size([2, 3, 224, 224])
✓ Encoder 前向传播成功
  - 输出形状: torch.Size([2, XXX, 384])
✓ Decoder 前向传播成功
  - 预测框形状: torch.Size([2, 1, 4])
✓ 所有前向传播测试通过

============================================================
测试 3: 选择损失计算
============================================================
✓ 选择损失计算成功
  - 损失值: 0.XXX
  - 是否需要梯度: True

============================================================
所有测试完成
============================================================
```

## 训练

### 1. 准备预训练权重

确保预训练权重存在：
```bash
ls pretrained/itpn/fast_itpn_tiny_clipl_e1200.pt
```

如果不存在，请下载或使用其他预训练权重。

### 2. 单卡训练

```bash
python lib/train/run_training.py \
    --script sutrack_select \
    --config sutrack_select_t224 \
    --save_dir ./checkpoints/train/sutrack_select/sutrack_select_t224 \
    --mode single
```

### 3. 多卡训练

```bash
python lib/train/run_training.py \
    --script sutrack_select \
    --config sutrack_select_t224 \
    --save_dir ./checkpoints/train/sutrack_select/sutrack_select_t224 \
    --mode multiple \
    --nproc_per_node 4
```

### 4. 训练参数

主要配置项（在 `experiments/sutrack_select/sutrack_select_t224.yaml` 中）：

```yaml
TRAIN:
  BATCH_SIZE: 16          # 批次大小
  EPOCH: 300              # 训练轮数
  LR: 0.0001              # 学习率
  
MODEL:
  ENCODER:
    USE_SELECTIVE_DEPTH: true              # 启用选择性深度
    SELECTIVE_DEPTH_THRESHOLD: 0.5         # 推理阈值
    SELECTION_LOSS_WEIGHT: 0.01            # 选择损失权重
```

## 评估

### 1. LaSOT 数据集

```bash
python tracking/test.py \
    sutrack_select \
    sutrack_select_t224 \
    --dataset lasot \
    --threads 4 \
    --num_gpus 1
```

### 2. GOT-10k 数据集

```bash
python tracking/test.py \
    sutrack_select \
    sutrack_select_t224 \
    --dataset got10k_test \
    --threads 4 \
    --num_gpus 1
```

### 3. TrackingNet 数据集

```bash
python tracking/test.py \
    sutrack_select \
    sutrack_select_t224 \
    --dataset trackingnet \
    --threads 4 \
    --num_gpus 1
```

## 配置调优

### 调整推理阈值

阈值越高，使用深度的频率越低，速度越快但可能损失精度：

```yaml
MODEL:
  ENCODER:
    SELECTIVE_DEPTH_THRESHOLD: 0.7  # 提高阈值，减少深度使用
```

### 调整选择损失权重

损失权重影响模型学习跳过策略的强度：

```yaml
MODEL:
  ENCODER:
    SELECTION_LOSS_WEIGHT: 0.02  # 增加权重，鼓励更多跳过
```

### 启用 Gumbel-Softmax

训练时使用 Gumbel-Softmax 采样（可能提升性能）：

```yaml
MODEL:
  ENCODER:
    USE_GUMBEL_SOFTMAX: true
```

## 性能分析

### 深度使用统计

训练或测试后，可以查看深度使用统计：

```python
from lib.models.sutrack_select import build_sutrack_select
from lib.config.sutrack_select.config import cfg

model = build_sutrack_select(cfg)
# ... 运行推理 ...

stats = model.encoder.body.selective_depth_module.get_depth_usage_stats()
print(f"平均深度使用率: {stats['avg_usage_rate']:.2%}")
print(f"各层使用率: {stats['usage_rate_per_layer']}")
```

### 速度测试

```python
import time
import torch

model.eval()
# 预热
for _ in range(10):
    with torch.no_grad():
        output = model(template_list, search_list, ...)

# 测速
start = time.time()
for _ in range(100):
    with torch.no_grad():
        output = model(template_list, search_list, ...)
end = time.time()

print(f"平均推理时间: {(end - start) / 100 * 1000:.2f} ms")
```

## 常见问题

### Q1: 模型创建失败

**问题**：`ImportError: No module named 'lib.models.sutrack_select'`

**解决**：确保在项目根目录下运行，并且 Python 路径正确：
```bash
cd /home/nick/code/code.sutrack/SUTrack
export PYTHONPATH=$PYTHONPATH:$(pwd)
python test_sutrack_select.py
```

### Q2: 预训练权重加载失败

**问题**：`FileNotFoundError: pretrained/itpn/fast_itpn_tiny_clipl_e1200.pt`

**解决**：
1. 下载预训练权重
2. 或修改配置使用其他权重路径

### Q3: 显存不足

**问题**：`CUDA out of memory`

**解决**：
1. 减少批次大小：`TRAIN.BATCH_SIZE: 8`
2. 使用梯度累积
3. 减少输入尺寸

### Q4: 深度使用率为 0

**问题**：推理时所有层都跳过深度

**解决**：
1. 降低阈值：`SELECTIVE_DEPTH_THRESHOLD: 0.3`
2. 当前实现 depth_feat 为 None，这是预期行为
3. 未来集成真实深度特征后会有实际使用

## 进阶使用

### 自定义深度特征

修改 `fastitpn.py` 中的 `forward_features` 方法：

```python
# 提取或加载深度特征
depth_feat = extract_depth_features(...)  # 自定义函数

# 使用选择性深度模块
xz, layer_prob = self.selective_depth_module(xz, depth_feat, layer_idx)
```

### 集成到其他模型

选择性深度模块是通用的，可以集成到其他 Transformer 模型：

```python
from lib.models.sutrack_select.selective_depth_modules import SelectiveDepthIntegration

# 在模型 __init__ 中
self.selective_depth = SelectiveDepthIntegration(
    dim=embed_dim,
    num_layers=num_layers,
    threshold=0.5
)

# 在 forward 中
for layer_idx, block in enumerate(self.blocks):
    x, prob = self.selective_depth(x, depth_feat, layer_idx)
    x = block(x)
```

## 相关资源

- **完整文档**：[INTEGRATION_SUMMARY_SELECT.md](INTEGRATION_SUMMARY_SELECT.md)
- **SGLA 论文**：Similarity-Guided Layer-Adaptive Vision Transformer
- **SUTrack 项目**：https://github.com/chenxin-dlut/SUTrack

## 联系与反馈

如有问题或建议，请：
1. 查看详细文档 `INTEGRATION_SUMMARY_SELECT.md`
2. 检查测试脚本 `test_sutrack_select.py`
3. 查看配置文件 `experiments/sutrack_select/sutrack_select_t224.yaml`

---
**版本**：v1.0  
**更新时间**：2026-02-22
