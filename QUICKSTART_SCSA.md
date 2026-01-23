# SUTrack-SCSA 快速开始指南

## 简介

SUTrack-SCSA是集成了SCSA（空间-通道协同注意力）机制的SUTrack模型变体，通过协同的空间和通道注意力增强目标跟踪性能。

## 核心特性

- ✅ **SMSA**: 共享多语义空间注意力，捕获多尺度空间特征
- ✅ **PCSA**: 渐进式通道自注意力，建模通道相关性
- ✅ **协同机制**: 空间引导通道，通道缓解多语义差异
- ✅ **即插即用**: 易于集成到现有Transformer架构

## 快速开始

### 1. 环境准备

确保已安装SUTrack的依赖项：
```bash
# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 训练模型

#### 使用Tiny模型（推荐）

适合快速实验和资源受限场景：
```bash
python tracking/train.py \
    --script sutrack_SCSA \
    --config sutrack_scsa_t224 \
    --save_dir ./checkpoints \
    --mode multiple \
    --nproc_per_node 4
```

**特点**:
- 模型大小: 小
- 训练速度: 快
- Batch Size: 32
- 推荐用于：快速原型验证

#### 使用Base模型

更好的性能表现：
```bash
python tracking/train.py \
    --script sutrack_SCSA \
    --config sutrack_scsa_b224 \
    --save_dir ./checkpoints \
    --mode multiple \
    --nproc_per_node 4
```

**特点**:
- 模型大小: 中等
- 训练速度: 中等
- Batch Size: 16
- 推荐用于：性能优化

### 3. 测试模型

#### DepthTrack数据集
```bash
python tracking/test.py sutrack_SCSA sutrack_scsa_t224 \
    --dataset_name depthtrack \
    --threads 4 \
    --num_gpus 4
```

#### LaSOT数据集
```bash
python tracking/test.py sutrack_SCSA sutrack_scsa_t224 \
    --dataset_name lasot \
    --threads 4 \
    --num_gpus 4
```

#### TrackingNet数据集
```bash
python tracking/test.py sutrack_SCSA sutrack_scsa_t224 \
    --dataset_name trackingnet \
    --threads 4 \
    --num_gpus 4
```

### 4. 查看训练日志

训练过程中会显示FPS等性能指标：
```bash
# 实时查看训练日志
tail -f logs/train_sutrack_SCSA_*.log

# 使用tensorboard查看
tensorboard --logdir=./tensorboard
```

## 配置调整

### 调整SCSA参数

编辑配置文件 `experiments/sutrack_SCSA/sutrack_scsa_t224.yaml`:

```yaml
MODEL:
  ENCODER:
    # 启用/禁用SCSA
    USE_SCSA: True
    
    # 通道压缩比例 (影响计算量和性能)
    # 2: 更多通道信息，计算量大
    # 4: 平衡 (推荐)
    # 8: 计算高效，可能损失信息
    SCSA_REDUCTION_RATIO: 4
    
    # 门控激活函数
    # 'sigmoid': 独立的通道门控
    # 'softmax': 通道间竞争性门控
    SCSA_GATE_LAYER: 'sigmoid'
```

### 调整训练参数

```yaml
TRAIN:
  BATCH_SIZE: 32        # 根据GPU内存调整
  EPOCH: 180            # 训练轮数
  LR: 0.0001           # 学习率
  NUM_WORKER: 10       # 数据加载线程数
```

## 常见问题

### Q1: 训练时GPU内存不足？

**解决方案**:
1. 减小batch size
2. 使用更小的模型 (Tiny)
3. 增加SCSA_REDUCTION_RATIO到8

```yaml
TRAIN:
  BATCH_SIZE: 16  # 从32减到16

MODEL:
  ENCODER:
    SCSA_REDUCTION_RATIO: 8  # 从4增加到8
```

### Q2: 如何查看FPS指标？

训练和测试过程中会自动显示FPS。也可以专门测试：

```bash
# 测试时会显示FPS
python tracking/test.py sutrack_SCSA sutrack_scsa_t224 \
    --dataset_name depthtrack \
    --threads 1 \
    --num_gpus 1
```

### Q3: 如何对比SCSA的效果？

```bash
# 1. 训练原始SUTrack
python tracking/train.py --script sutrack --config sutrack_t224

# 2. 训练SUTrack-SCSA
python tracking/train.py --script sutrack_SCSA --config sutrack_scsa_t224

# 3. 在相同数据集上测试对比
python tracking/test.py sutrack sutrack_t224 --dataset_name depthtrack
python tracking/test.py sutrack_SCSA sutrack_scsa_t224 --dataset_name depthtrack
```

### Q4: 如何禁用SCSA进行对比？

编辑配置文件，设置 `USE_SCSA: False`:

```yaml
MODEL:
  ENCODER:
    USE_SCSA: False  # 禁用SCSA
```

## 性能优化建议

### 1. 数据集选择

根据任务选择合适的训练数据集：

```yaml
DATA:
  TRAIN:
    DATASETS_NAME:
    - LASOT              # 通用RGB跟踪
    - GOT10K_vottrain    # 通用RGB跟踪
    - DepthTrack_train   # RGB-D跟踪
    - LasHeR_train       # 红外跟踪
    DATASETS_RATIO:
    - 1
    - 1
    - 2  # 增加RGB-D数据的权重
    - 2
```

### 2. 学习率调整

针对不同模型大小调整学习率：

```yaml
TRAIN:
  LR: 0.0001              # Tiny模型
  # LR: 0.00005           # Base/Large模型
  ENCODER_MULTIPLIER: 0.1 # 编码器学习率倍数
```

### 3. 数据增强

调整数据增强参数以提升鲁棒性：

```yaml
DATA:
  SEARCH:
    CENTER_JITTER: 3.5  # 中心抖动
    SCALE_JITTER: 0.5   # 尺度抖动
```

## 高级用法

### 自定义SCSA模块

如需修改SCSA结构，编辑 `lib/models/sutrack_SCSA/scsa_modules.py`:

```python
# 修改SMSA的卷积核尺寸
class Shareable_Multi_Semantic_Spatial_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],  # 可修改
            gate_layer: str = 'sigmoid',
    ):
        ...

# 修改PCSA的压缩策略
class Progressive_Channel_wise_Self_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            reduction_ratio: int = 4,  # 可修改
    ):
        ...
```

### 多GPU训练

```bash
# 使用4个GPU
python tracking/train.py \
    --script sutrack_SCSA \
    --config sutrack_scsa_t224 \
    --save_dir ./checkpoints \
    --mode multiple \
    --nproc_per_node 4

# 使用8个GPU
python tracking/train.py \
    --script sutrack_SCSA \
    --config sutrack_scsa_t224 \
    --save_dir ./checkpoints \
    --mode multiple \
    --nproc_per_node 8
```

### 断点续训

```bash
python tracking/train.py \
    --script sutrack_SCSA \
    --config sutrack_scsa_t224 \
    --save_dir ./checkpoints \
    --mode multiple \
    --nproc_per_node 4 \
    --resume_path ./checkpoints/SUTRACKSCSA/sutrack_scsa_t224/checkpoint_epoch_100.pth
```

## 实验记录

建议记录每次实验的配置和结果：

```bash
# 创建实验记录文件
cat > experiments/sutrack_SCSA/experiment_log.md << EOF
# 实验记录

## 实验1: Baseline
- 日期: 2026-01-23
- 配置: sutrack_scsa_t224
- SCSA_REDUCTION_RATIO: 4
- SCSA_GATE_LAYER: sigmoid
- 数据集: GOT10K + DepthTrack
- 结果: 
  - Success: XX%
  - Precision: XX%
  - FPS: XX

## 实验2: 调整reduction_ratio
- 日期: 2026-01-24
- 配置: sutrack_scsa_t224 (修改)
- SCSA_REDUCTION_RATIO: 8
- 结果:
  - Success: XX%
  - Precision: XX%
  - FPS: XX
EOF
```

## 参考资料

- **SCSA论文**: https://arxiv.org/pdf/2407.05128
- **详细文档**: [experiments/sutrack_SCSA/README.md](file:///home/nick/code/code.sutrack/SUTrack/experiments/sutrack_SCSA/README.md)
- **集成总结**: [INTEGRATION_SUMMARY_SCSA.md](file:///home/nick/code/code.sutrack/SUTrack/INTEGRATION_SUMMARY_SCSA.md)
- **测试脚本**: [examples/test_scsa.py](file:///home/nick/code/code.sutrack/SUTrack/examples/test_scsa.py)

## 技术支持

如有问题或建议，请：
1. 查看详细文档
2. 检查配置文件
3. 查看训练日志
4. 提交Issue

---

**祝训练顺利！🚀**
