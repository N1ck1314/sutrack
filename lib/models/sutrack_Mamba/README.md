# SUTrack-Mamba

基于 MCITrack (AAAI 2025) 的 Mamba 模块实现，为 SUTrack 添加高效的上下文信息传递能力。

## 🎯 核心思想

参考 MCITrack 论文：**"Exploring Enhanced Contextual Information for Video-Level Object Tracking"**

MCITrack 利用 Mamba 的选择性状态空间模型（SSM）实现：
1. **隐藏状态传递**：通过隐藏状态高效传递上下文信息
2. **线性复杂度**：O(L) vs Transformer 的 O(L²)
3. **长程依赖**：保持长序列的上下文记忆

## 📁 文件结构

```
lib/models/sutrack_Mamba/
├── mamba_module.py       # Mamba 核心模块实现
│   ├── MambaBlock        # 选择性状态空间模型
│   ├── SimpleMambaFusion # 简化版 Mamba 融合（即插即用）
│   └── MambaNeck         # 完整 Mamba Neck（包含交互机制）
├── sutrack.py            # SUTrack-Mamba 主模型
├── encoder.py            # 编码器（复制自标准 sutrack）
├── decoder.py            # 解码器（复制自标准 sutrack）
├── clip.py               # 文本编码器（复制自标准 sutrack）
└── task_decoder.py       # 任务解码器（复制自标准 sutrack）

lib/config/sutrack_Mamba/
└── config.py             # Mamba 模块配置

experiments/sutrack_Mamba/
└── sutrack_mamba_t224.yaml  # Tiny 模型配置
```

## 🚀 快速开始

### 1. 训练

```bash
# Tiny 模型 (224x224)
python tracking/train.py --script sutrack_Mamba --config sutrack_mamba_t224 --save_dir . --mode single
```

### 2. 配置说明

在 `experiments/sutrack_Mamba/sutrack_mamba_t224.yaml` 中：

```yaml
MODEL:
  # Mamba 配置
  USE_MAMBA: True           # 启用/禁用 Mamba 模块
  MAMBA_LAYERS: 2           # Mamba 层数（推荐 2-4）
  MAMBA_D_STATE: 16         # SSM 状态维度（推荐 16）
```

## 🧩 模块说明

### 1. MambaBlock

选择性状态空间模型的核心实现：

```python
# 输入分成两个分支
x, z = input.chunk(2, dim=-1)

# x 分支：1D 卷积 + SSM
x = conv1d(x)
y, h = ssm_step(x, h)  # 状态空间模型

# z 分支：门控
z = silu(z)

# 输出：y * z
output = y * z
```

**优势**：
- 线性复杂度 O(L)
- 保持长程依赖
- 参数高效

### 2. SimpleMambaFusion

简化版，作为即插即用模块添加在 Encoder 之后：

```python
# 在 sutrack.py 中的使用
xz = encoder(...)          # 标准 ViT encoder
xz = mamba_fusion(xz)      # Mamba 上下文增强
```

**参数量**：~2M（2 层）

### 3. MambaNeck（可选）

完整版 MCITrack Neck，包含 Injector + Extractor 交互机制：
- 适合需要更强上下文建模的场景
- 参数量：~10M（4 层）

## 📊 与其他模块对比

| 模块 | 复杂度 | 参数量 | 优势 | 适用场景 |
|------|--------|--------|------|----------|
| **Mamba** | O(L) | 2M | 线性复杂度、长程依赖 | 长序列、视频跟踪 |
| STAtten | O(L²) | 3M | 时空注意力 | 多模态融合 |
| CMA | O(L²) | 2M | 跨模态注意力 | RGB-频域融合 |
| RMT | O(L log L) | 5M | 保持记忆 | 长视频跟踪 |

## 🔧 参数调优

### Mamba 层数 (MAMBA_LAYERS)

```yaml
MAMBA_LAYERS: 2   # 轻量级，推荐用于 Tiny/Small
MAMBA_LAYERS: 4   # 标准配置，推荐用于 Base/Large
```

### SSM 状态维度 (MAMBA_D_STATE)

```yaml
MAMBA_D_STATE: 16   # 标准配置，平衡性能和效率
MAMBA_D_STATE: 32   # 更强的记忆能力，参数量增加
```

### 禁用 Mamba

```yaml
MODEL:
  USE_MAMBA: False  # 回退到标准 SUTrack
```

## 📚 参考文献

```bibtex
@inproceedings{kang2025mcitrack,
  title={Exploring Enhanced Contextual Information for Video-Level Object Tracking},
  author={Kang, Ben and Chen, Xin and Lai, Simiao and Liu, Yang and Liu, Yi and Wang, Dong},
  booktitle={AAAI},
  year={2025}
}
```

## ⚠️ 注意事项

1. **内存占用**：Mamba 使用隐藏状态，会增加内存占用
2. **训练稳定性**：建议使用梯度裁剪（GRAD_CLIP_NORM: 0.1）
3. **与其他模块兼容**：可以与 EUCB、MLKA 等模块组合使用

## 🎓 原理解释

### 为什么 Mamba 适合目标跟踪？

1. **线性复杂度**：处理长序列（视频帧）时效率更高
2. **隐藏状态**：自然地保持跨帧的上下文信息
3. **选择性机制**：动态选择重要信息，过滤噪声

### SSM 公式

```
h' = exp(Δ*A) * h + Δ * B * x   # 状态更新
y = C * h + D * x                # 输出计算
```

其中：
- `h`: 隐藏状态（记忆）
- `Δ, B, C`: 输入相关的动态参数（选择性）
- `A, D`: 固定参数

## 🔍 调试技巧

训练时会显示 Mamba 模块的初始化信息：

```
✅ Mamba模块已成功初始化！
   - mamba_fusion: SimpleMambaFusion
   - 层数: 2
   - 增强状态: ✅ 启用 (use_mamba=True)
   - Mamba参数量: 1.95M
   - 核心机制: 选择性状态空间模型 (SSM) + 线性复杂度
```

## 📈 性能预期

- **速度**：相比 Transformer，长序列时更快
- **精度**：与标准 SUTrack 相当或略优
- **内存**：隐藏状态会增加显存占用

---

**提示**：这是一个即插即用的模块，可以随时通过 `USE_MAMBA: False` 禁用。
