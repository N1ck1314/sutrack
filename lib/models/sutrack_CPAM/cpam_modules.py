"""
CPAM (Channel and Position Attention Mechanism) modules for SUTrack
Based on paper: "ASF-YOLO: A novel YOLO model with attentional scale sequence fusion for cell instance segmentation"

核心模块:
1. ChannelAttention: 通道注意力（关注"看什么"）
2. LocalAttention: 局部位置注意力（关注"看哪里"）
3. CPAM: 联合通道+位置注意力的即插即用模块

适用场景:
- 小目标/密集目标检测与分割
- 多尺度特征融合后的精修
- 需要同时强化语义通道和空间位置的任务
"""

import torch
import torch.nn as nn
import math


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    
    核心思想:
    - 对每个通道做全局平均池化
    - 使用1D卷积建模相邻通道的相关性（无降维）
    - 生成通道权重，抑制冗余通道、强化关键语义通道
    
    相比 SENet:
    - 不做维度压缩，减少信息损失
    - 使用局部通道交互（1D Conv）而非全连接
    - 更适合细粒度任务和小目标
    """
    def __init__(self, channel, b=1, gamma=2):
        """
        Args:
            channel: 输入特征通道数
            b: 卷积核大小计算的偏移量
            gamma: 卷积核大小计算的缩放因子
        """
        super(ChannelAttention, self).__init__()
        
        # 自适应计算卷积核大小（基于通道数）
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        # 全局平均池化：(B, C, H, W) -> (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 1D卷积：建模通道间的局部相关性
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             padding=(kernel_size - 1) // 2, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            通道加权后的特征: (B, C, H, W)
        """
        # 全局平均池化
        y = self.avg_pool(x)  # (B, C, 1, 1)
        
        # 准备1D卷积：(B, C, 1, 1) -> (B, 1, C)
        y = y.squeeze(-1)  # (B, C, 1)
        y = y.transpose(-1, -2)  # (B, 1, C)
        
        # 1D卷积建模通道相关性
        y = self.conv(y)  # (B, 1, C)
        
        # 恢复形状并生成权重
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        
        # 通道加权
        return x * y.expand_as(x)


class LocalAttention(nn.Module):
    """
    局部位置注意力模块
    
    核心思想:
    - 沿宽度和高度方向分别建模（保留位置信息）
    - 对横向/纵向结构敏感
    - 精准强化目标所在的空间位置
    
    相比普通 Spatial Attention:
    - 不会破坏几何结构（不用全局池化）
    - 定位更稳定
    - 适合细长结构和方向性明显的目标
    """
    def __init__(self, channel, reduction=16):
        """
        Args:
            channel: 输入特征通道数
            reduction: 通道压缩比例
        """
        super(LocalAttention, self).__init__()
        
        # 通道压缩（降低计算量）
        self.conv_1x1 = nn.Conv2d(in_channels=channel, 
                                  out_channels=channel // reduction,
                                  kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        
        # 水平和垂直方向的权重生成
        self.F_h = nn.Conv2d(in_channels=channel // reduction, 
                            out_channels=channel,
                            kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction,
                            out_channels=channel,
                            kernel_size=1, stride=1, bias=False)
        
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            位置加权后的特征: (B, C, H, W)
        """
        _, _, h, w = x.size()
        
        # 水平方向池化：沿宽度求平均 (B, C, H, 1)
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        
        # 垂直方向池化：沿高度求平均 (B, C, 1, W)
        x_w = torch.mean(x, dim=2, keepdim=True)
        
        # 拼接后处理
        x_cat = torch.cat((x_h, x_w), dim=3)  # (B, C, H+W, 1) or (B, C, 1, H+W)
        x_cat = self.relu(self.bn(self.conv_1x1(x_cat)))
        
        # 分割为水平和垂直
        x_cat_h, x_cat_w = x_cat.split([h, w], 3)
        
        # 生成水平和垂直权重
        s_h = self.sigmoid_h(self.F_h(x_cat_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_w))
        
        # 位置加权（水平 × 垂直）- 添加数值稳定性
        # 避免两个sigmoid直接相乘导致极小值
        out = x * (s_h.expand_as(x) * s_w.expand_as(x) + 1e-6)
        return out


class CPAM(nn.Module):
    """
    Channel and Position Attention Mechanism (CPAM)
    
    核心创新:
    1. 联合通道注意力与位置注意力
    2. 先筛选语义通道，再强化空间位置
    3. 即插即用，适合多尺度特征融合后的精修
    
    计算流程:
    input1 → ChannelAttention → + input2 → LocalAttention → output
    
    适用场景:
    - 小目标/密集目标检测
    - 多尺度特征融合后的再筛选
    - 需要同时关注"看什么"和"看哪里"的任务
    """
    def __init__(self, channel, reduction=16, b=1, gamma=2):
        """
        Args:
            channel: 输入特征通道数
            reduction: LocalAttention的通道压缩比例
            b, gamma: ChannelAttention的卷积核大小参数
        """
        super(CPAM, self).__init__()
        
        # 通道注意力：筛选"看什么"
        self.channel_att = ChannelAttention(channel, b=b, gamma=gamma)
        
        # 位置注意力：强化"看哪里"
        self.local_att = LocalAttention(channel, reduction=reduction)
        
    def forward(self, input1, input2):
        """
        Args:
            input1: 主特征 (B, C, H, W)
            input2: 辅助特征 (B, C, H, W)，通常是跳跃连接或融合特征
        Returns:
            增强后的特征: (B, C, H, W)
        """
        # 步骤1: 对input1进行通道筛选
        input1 = self.channel_att(input1)
        
        # 步骤2: 与input2融合
        x = input1 + input2
        
        # 步骤3: 对融合结果进行位置强化
        x = self.local_att(x)
        
        return x


class CPAM_SingleInput(nn.Module):
    """
    CPAM 单输入版本（用于单一特征的增强）
    """
    def __init__(self, channel, reduction=16, b=1, gamma=2):
        super(CPAM_SingleInput, self).__init__()
        self.channel_att = ChannelAttention(channel, b=b, gamma=gamma)
        self.local_att = LocalAttention(channel, reduction=reduction)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """Xavier/Kaiming 初始化以增强数值稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            增强后的特征: (B, C, H, W)
        """
        # 添加输入检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("\u26a0️  CPAM 输入检测到 NaN/Inf!")
            return x  # 直接返回原始特征
        
        # 保存原始特征
        identity = x
        
        try:
            # 通道注意力
            x_ch = self.channel_att(x)
            if torch.isnan(x_ch).any():
                print("\u26a0️  ChannelAttention 输出 NaN, 跳过")
                return identity
            
            # 位置注意力
            x_out = self.local_att(x_ch)
            if torch.isnan(x_out).any():
                print("\u26a0️  LocalAttention 输出 NaN, 跳过")
                return identity
            
            # 加权残差连接 - 增大残差权重
            x = x_out * 0.2 + identity * 0.8  # 更保守的融合
            
            # 最终检查
            if torch.isnan(x).any():
                print("\u26a0️  CPAM 最终输出 NaN, 返回原始特征")
                return identity
                
        except Exception as e:
            print(f"\u26a0️  CPAM 处理错误: {e}, 返回原始特征")
            return identity
        
        return x


if __name__ == '__main__':
    print("="*60)
    print("Testing CPAM Modules")
    print("="*60)
    
    # 测试 ChannelAttention
    print("\n[1] Testing ChannelAttention...")
    x = torch.randn(2, 64, 32, 32)
    ch_att = ChannelAttention(64)
    out = ch_att(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in ch_att.parameters()) / 1e3:.2f}K")
    
    # 测试 LocalAttention
    print("\n[2] Testing LocalAttention...")
    x = torch.randn(2, 64, 32, 32)
    local_att = LocalAttention(64, reduction=16)
    out = local_att(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in local_att.parameters()) / 1e3:.2f}K")
    
    # 测试 CPAM (双输入)
    print("\n[3] Testing CPAM (dual-input)...")
    input1 = torch.randn(2, 64, 32, 32)
    input2 = torch.randn(2, 64, 32, 32)
    cpam = CPAM(64)
    out = cpam(input1, input2)
    print(f"   Input1: {input1.shape}, Input2: {input2.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in cpam.parameters()) / 1e3:.2f}K")
    
    # 测试 CPAM_SingleInput
    print("\n[4] Testing CPAM_SingleInput...")
    x = torch.randn(2, 384, 14, 14)  # SUTrack typical feature
    cpam_single = CPAM_SingleInput(384)
    out = cpam_single(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in cpam_single.parameters()) / 1e3:.2f}K")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
