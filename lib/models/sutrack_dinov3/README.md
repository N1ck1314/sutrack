# SUTrack with DINOv3 ConvNeXt-Tiny Encoder

## 简介

这个文件夹包含了使用 **DINOv3 ConvNeXt-Tiny** 作为 backbone 的 SUTrack 实现。

### 主要特点

- **Backbone**: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
- **速度优先**: ConvNeXt-Tiny 比 ViT 更快，适合实时跟踪
- **自动加载**: 从 HuggingFace 自动下载预训练权重，无需手动管理
- **兼容接口**: 完全兼容原 SUTrack 的训练/测试流程

---

## 文件结构

```
sutrack_dinov3/
├── __init__.py           # 模块初始化
├── encoder.py            # DINOv3 ConvNeXt-Tiny encoder 实现
├── decoder.py            # 复用 sutrack_scale 的 decoder
├── task_decoder.py       # 复用 sutrack_scale 的 task_decoder
├── clip.py               # 文本编码器（复用）
└── sutrack.py            # SUTrack 主模型类
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install transformers
```

### 2. 测试接口

运行测试脚本验证模型是否能正常 forward：

```bash
python test_dinov3_encoder.py
```

如果输出显示所有测试通过，说明接口没问题。

### 3. 训练模型

使用 `sutrack_dinov3` 配置训练：

```bash
# 示例：在 GOT10K 上训练
python tracking/train.py --config sutrack_dinov3
```

**注意事项：**

- 第一次运行会从 HuggingFace 自动下载 DINOv3 权重（约 ~120MB），需要联网
- 建议先用 `FREEZE_ENCODER=True` 冻结 backbone 快速验证，再用 `False` 微调

---

## 配置说明

配置文件：`lib/config/sutrack_dinov3/config.py`

### 关键参数

```python
# Encoder 相关
cfg.MODEL.ENCODER.TYPE = "dinov3_convnext_tiny"  # backbone 类型
cfg.MODEL.ENCODER.STRIDE = 32                    # ConvNeXt 输出 stride
cfg.MODEL.ENCODER.CLASS_TOKEN = False            # 是否添加 cls token
cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE = False    # 是否使用 token type embedding

# 训练相关
cfg.TRAIN.FREEZE_ENCODER = False                 # 是否冻结 encoder
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1               # encoder 学习率倍数
```

### 与原 SUTrack 的差异

| 配置项 | 原 SUTrack (Fast-iTPN) | 新版 (DINOv3 ConvNeXt-Tiny) |
|--------|------------------------|------------------------------|
| Encoder Type | `fastitpnb` | `dinov3_convnext_tiny` |
| Pretrain Type | 本地权重路径 | HuggingFace 自动下载 |
| Stride | 14 | 32 |
| Output Dim | 512 | 768 |
| Token Type | 支持前景/背景 mask | 简化版（暂不支持） |

---

## 模型架构

### Encoder 流程

```
Input:
  - template_list: [Tensor(B, 3, 112, 112)]
  - search_list: [Tensor(B, 3, 224, 224)]
  - template_anno_list: [Tensor(B, 4)]

↓

1. 堆叠 template/search → (B*num, 3, H, W)

↓

2. DINOv3 ConvNeXt-Tiny 提特征
   - search (224x224) → (B*num, 49, 768)   # 7x7 feature map
   - template (112x112) → (B*num, 16, 768)  # 4x4 feature map

↓

3. 加位置编码 + (可选) token type embedding

↓

4. 拼接 search + template + text_src

↓

5. (可选) 添加 cls_token

↓

Output: [Tensor(B, L_total, 768)]
```

### Decoder 流程

复用原 SUTrack 的 CENTER/CORNER/MLP decoder，只需要：

- 从 `features[0]` 中提取 search tokens
- Reshape 成 `(B, 768, 7, 7)` 给 decoder
- 输出 bbox 预测

---

## 与原 Fast-iTPN 对比

### 优势

1. **速度更快**: ConvNeXt 是纯卷积结构，推理比 ViT 更快
2. **更简洁**: 不需要手动管理预训练权重，HuggingFace 自动下载
3. **更新的预训练**: DINOv3 在更大数据集上预训练（LVD-1689M）

### 劣势

1. **Stride 更大**: 32 vs 14，特征图分辨率更低（7x7 vs 16x16）
2. **简化版本**: 暂时没有实现前景/背景 mask 和 token_type_indicate
3. **特征维度不同**: 768 vs 512，decoder 需要适配

---

## 已知问题与改进方向

### 当前简化

- **没有前景/背景 mask**: 原 Fast-iTPN 会根据 template_anno 生成前景/背景 token type，当前版本暂时跳过
- **没有 cls_token**: 简化版先不加，后续可以加上
- **文本模态关闭**: 配置中 `MULTI_MODAL_LANGUAGE=False`，后续可以打开

### 改进方向

1. **加回 token_type_indicate**:
   - 在 `encoder.py` 的 `forward` 中实现 `create_mask` 逻辑
   - 根据 template_anno 生成前景/背景权重
   
2. **使用多尺度特征**:
   - DINOv3 ConvNeXt 有 4 个 stage，可以提取多尺度特征
   - 类似 Fast-iTPN 的 FPN 结构
   
3. **尝试更大模型**:
   - `dinov3_convnext_small/base` 精度更高（但速度会慢一些）
   - `dinov3_vits16` 如果想要 ViT 结构

4. **Decoder 改进**:
   - 当前 stride=32 太大，可能丢失细节
   - 考虑加上采样或用多尺度特征

---

## 常见问题

### Q1: 首次运行很慢？

A: 第一次会从 HuggingFace 下载权重（~120MB），需要联网。下载后会缓存到 `~/.cache/huggingface/`。

### Q2: 提示 transformers 版本不对？

A: 需要 `transformers >= 4.56.0`：

```bash
pip install --upgrade transformers
```

### Q3: 显存不够？

A: 尝试：
- 减小 batch_size
- 设置 `cfg.TRAIN.FREEZE_ENCODER = True` 冻结 backbone
- 使用 gradient checkpointing（需要修改 encoder）

### Q4: 精度比原 Fast-iTPN 低？

A: 可能原因：
- Stride 32 太大，特征图太粗糙
- 简化版没有前景/背景 mask
- 需要更多训练 epoch 让模型适应新 backbone

建议：
- 先验证接口正确
- 在小数据集上调参
- 逐步加回原有功能（token_type、多尺度等）

---

## 参考资料

- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [HuggingFace Model Card](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m)
- [原 SUTrack 论文/代码](../sutrack/)

---

## 更新日志

- **2025-XX-XX**: 初始版本，实现基础接口
  - DINOv3 ConvNeXt-Tiny encoder
  - 复用原 decoder/task_decoder
  - 配置文件和测试脚本
