# SUTrack-ASCN 快速入门

## 什么是ASCNet？

ASCNet (Asymmetric Sampling Correction Network) 是一个专门设计用于处理**条纹噪声**和**传感器非均匀性**的网络架构。它的两大核心模块：

1. **RHDWT** - 残差哈尔离散小波变换（下采样）
   - 融合小波的方向先验和CNN的语义特征
   
2. **CNCM** - 列非均匀性校正模块（特征增强）
   - 通过列注意力、空间注意力和自校准三分支增强特征

## 核心优势

✅ **方向先验感知**：RHDWT利用Haar小波捕获条纹的方向性  
✅ **全局列建模**：CNCM能够建模长程依赖，区分条纹和真实结构  
✅ **即插即用**：模块设计独立，可轻松集成到其他网络  
✅ **多场景适用**：红外图像、遥感图像、低光照增强等

## 快速开始

### 1. 环境准备

```bash
conda activate sutrack
```

### 2. 测试模块

验证ASCNet模块是否正常工作：

```bash
python test_ascnet_integration.py
```

预期输出：
```
✅ 所有ASCNet模块测试通过！
```

### 3. 开始训练

```bash
bash train_ascn.sh
```

或手动指定参数：

```bash
python tracking/train.py \
    --script sutrack_ascn \
    --config sutrack_ascn_t224 \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 4
```

### 4. 查看配置

训练启动时会显示详细的ASCNet配置信息：

```
============================================================
🔍 SUTrack-ASCN (ASCNet) 配置确认
============================================================
✓ RHDWT下采样启用状态: 🟢 已启用
✓ 核心机制: 残差哈尔小波变换
  - 模型驱动分支: 固定Haar小波捕获方向先验
  - 残差分支: 步进卷积捕获数据驱动语义
  - 特点: 融合先验知识与深度语义
✓ CNCM模块启用状态: 🟢 已启用
✓ RCSSC块数量: 3
✓ 核心机制: 列非均匀性校正
  - CAB: 列注意力分支（双池化+双重校正）
  - SAB: 空间注意力分支（关键区域增强）
  - SCB: 自校准分支（长程依赖建模）
  - 特点: 全局上下文 + 列特征精细校正
✓ 应用场景: 条纹噪声抑制、传感器非均匀性校正
============================================================
```

## 配置调整

修改 `experiments/sutrack_ascn/sutrack_ascn_t224.yaml`：

```yaml
TRAIN:
  ASCNET:
    USE_RHDWT: True          # 启用/禁用RHDWT下采样
    USE_CNCM: True           # 启用/禁用CNCM模块
    CNCM_NUM_BLOCKS: 3       # CNCM中RCSSC块的数量（1-5）
```

## 模块说明

### RHDWT - 残差哈尔小波下采样

**作用**：替代标准的步进卷积或池化下采样

**优势**：
- 保留小波的方向性先验（适合条纹、雨丝等方向性模式）
- 融合数据驱动的深度语义
- 获得更鲁棒的下采样特征

**使用场景**：
```python
from lib.models.sutrack_ascn.ascnet_modules import RHDWT

# 创建RHDWT层
rhdwt = RHDWT(in_channels=64, out_channels=128)

# 下采样
x = torch.randn(2, 64, 32, 32)
x_down = rhdwt(x)  # [2, 128, 16, 16]
```

### CNCM - 列非均匀性校正模块

**作用**：特征增强，特别适合处理列相关的噪声

**三分支结构**：
1. **CAB**：列注意力（强化列特征）
2. **SAB**：空间注意力（增强关键区域）
3. **SCB**：自校准（建立长程依赖）

**使用场景**：
```python
from lib.models.sutrack_ascn.ascnet_modules import CNCM

# 创建CNCM模块
cncm = CNCM(channels=64, num_blocks=3)

# 特征增强
x = torch.randn(2, 64, 16, 16)
x_enhanced = cncm(x)  # [2, 64, 16, 16]
```

### 完整流水线

```python
from lib.models.sutrack_ascn.ascnet_modules import RHDWT, CNCM

# 构建流水线
rhdwt = RHDWT(in_channels=64, out_channels=128)
cncm = CNCM(channels=128, num_blocks=3)

# 前向传播
x = torch.randn(2, 64, 32, 32)
x_down = rhdwt(x)           # 下采样 [2, 128, 16, 16]
x_enhanced = cncm(x_down)   # 特征增强 [2, 128, 16, 16]
```

## 性能提示

### 内存优化

如果遇到OOM（内存不足），可以：

1. **减少CNCM块数量**：
```yaml
CNCM_NUM_BLOCKS: 2  # 从3改为2
```

2. **减小batch size**：
```yaml
BATCH_SIZE: 16  # 从32改为16
```

### 速度优化

RHDWT相比标准卷积略慢（因为小波分解），如果追求极致速度：

```yaml
USE_RHDWT: False  # 禁用RHDWT，使用标准下采样
USE_CNCM: True    # 保留CNCM增强
```

## 适用数据集

ASCNet特别适合：

1. **RGBD跟踪**
   - DepthTrack：深度图像常有传感器噪声
   - VOT-RGBD：深度通道噪声抑制

2. **RGBT跟踪**
   - RGBT234：红外图像FPA条纹噪声
   - LASHER：热红外跟踪

3. **Event跟踪**
   - VisEvent：事件相机噪声校正

4. **低质量RGB**
   - 夜间跟踪：低光照列噪声
   - 雨天跟踪：雨丝方向性噪声

## 常见问题

### Q1: RHDWT比标准卷积慢吗？

**A**: 略慢（约1.2x），但获得了更强的方向先验建模能力。如果追求极致速度，可禁用RHDWT。

### Q2: CNCM块数量怎么选？

**A**: 
- **3块**（默认）：平衡精度和速度
- **2块**：速度优先
- **4-5块**：精度优先（需要更多内存）

### Q3: 可以只用CNCM不用RHDWT吗？

**A**: 可以！CNCM是独立的特征增强模块，可单独使用：
```yaml
USE_RHDWT: False
USE_CNCM: True
```

### Q4: 如何在其他网络中使用这些模块？

**A**: 非常简单，只需导入：
```python
from lib.models.sutrack_ascn.ascnet_modules import RHDWT, CNCM

# 替换你的下采样层
self.downsample = RHDWT(in_ch, out_ch)

# 在特征增强位置插入
self.enhance = CNCM(channels, num_blocks=3)
```

## 进阶使用

### 自定义RCSSC块

如果需要调整CNCM的内部结构：

```python
from lib.models.sutrack_ascn.ascnet_modules import RCSSC

# 创建单个RCSSC块
rcssc = RCSSC(channels=64)

# 可以堆叠多个RCSSC
x = input_feature
for _ in range(num_blocks):
    x = rcssc(x)
```

### 可视化特征

查看RHDWT的小波分解结果：

```python
import torch
from lib.models.sutrack_ascn.ascnet_modules import HDWT

hdwt = HDWT()
x = torch.randn(1, 3, 224, 224)
subbands = hdwt(x)  # [1, 12, 112, 112]

# 拆分4个子带
ll, lh, hl, hh = subbands.chunk(4, dim=1)

# 可视化（使用matplotlib）
import matplotlib.pyplot as plt
plt.subplot(221); plt.imshow(ll[0, 0].detach().cpu()); plt.title('LL')
plt.subplot(222); plt.imshow(lh[0, 0].detach().cpu()); plt.title('LH')
plt.subplot(223); plt.imshow(hl[0, 0].detach().cpu()); plt.title('HL')
plt.subplot(224); plt.imshow(hh[0, 0].detach().cpu()); plt.title('HH')
```

## 参考资源

- 📄 **集成总结**：[INTEGRATION_SUMMARY_ASCN.md](INTEGRATION_SUMMARY_ASCN.md)
- 🧪 **测试脚本**：`test_ascnet_integration.py`
- 📝 **配置文件**：`lib/config/sutrack_ascn/config.py`
- 🔧 **核心代码**：`lib/models/sutrack_ascn/ascnet_modules.py`

## 下一步

1. ✅ 运行测试验证模块正常工作
2. ✅ 查看配置文件熟悉参数
3. ✅ 开始训练实验
4. 📊 分析结果并调整参数
5. 🎯 在你的目标数据集上微调

祝训练顺利！🚀
