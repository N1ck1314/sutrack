"""
ASCNet核心模块实现
包括：
1. RHDWT - 残差哈尔离散小波变换（下采样）
2. CNCM - 列非均匀性校正模块
3. RCSSC - 残差列空间自校正块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import pywt


class HDWT(nn.Module):
    """
    哈尔离散小波变换 (Haar Discrete Wavelet Transform)
    将特征分解为4个子带: LL(低频), LH(水平高频), HL(垂直高频), HH(对角高频)
    """
    def __init__(self):
        super(HDWT, self).__init__()
        # Haar小波的固定滤波器（不可学习）
        self.requires_grad = False
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            subbands: [B, 4*C, H/2, W/2] - 拼接后的4个子带
        """
        # 使用PyWavelets进行2D离散小波变换
        # 由于pywt作用于numpy，这里使用手动实现的Haar滤波器
        return self.dwt_2d_haar(x)
    
    def dwt_2d_haar(self, x):
        """
        手动实现的2D Haar小波变换（适用于PyTorch）
        """
        B, C, H, W = x.shape
        
        # 定义Haar小波滤波器
        # 低通滤波器 (averaging)
        h_low = torch.tensor([[0.5, 0.5]], device=x.device, dtype=x.dtype)
        # 高通滤波器 (differencing)  
        h_high = torch.tensor([[0.5, -0.5]], device=x.device, dtype=x.dtype)
        
        # 先对行进行滤波
        x_low = F.conv2d(x.view(B*C, 1, H, W), 
                         h_low.view(1, 1, 1, 2), 
                         stride=(1, 2), padding=0)
        x_high = F.conv2d(x.view(B*C, 1, H, W), 
                          h_high.view(1, 1, 1, 2), 
                          stride=(1, 2), padding=0)
        
        # 再对列进行滤波
        x_ll = F.conv2d(x_low, 
                        h_low.t().view(1, 1, 2, 1), 
                        stride=(2, 1), padding=0)
        x_lh = F.conv2d(x_low, 
                        h_high.t().view(1, 1, 2, 1), 
                        stride=(2, 1), padding=0)
        x_hl = F.conv2d(x_high, 
                        h_low.t().view(1, 1, 2, 1), 
                        stride=(2, 1), padding=0)
        x_hh = F.conv2d(x_high, 
                        h_high.t().view(1, 1, 2, 1), 
                        stride=(2, 1), padding=0)
        
        # Reshape回 [B, C, H/2, W/2]
        x_ll = x_ll.view(B, C, H//2, W//2)
        x_lh = x_lh.view(B, C, H//2, W//2)
        x_hl = x_hl.view(B, C, H//2, W//2)
        x_hh = x_hh.view(B, C, H//2, W//2)
        
        # 在通道维度拼接
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)


class RHDWT(nn.Module):
    """
    残差哈尔离散小波变换 (Residual Haar DWT)
    
    双分支结构：
    - 模型驱动分支：使用固定的Haar小波捕获方向先验
    - 残差分支：使用步进卷积捕获数据驱动的语义
    """
    def __init__(self, in_channels, out_channels):
        super(RHDWT, self).__init__()
        
        # 模型驱动分支：Haar小波
        self.hdwt = HDWT()
        # HDWT输出4倍通道，需要降维
        self.conv_model = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 残差分支：标准步进卷积
        self.conv_res = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, out_channels, H/2, W/2]
        """
        # 模型驱动分支
        x_dwt = self.hdwt(x)  # [B, 4C, H/2, W/2]
        x_model = self.conv_model(x_dwt)  # [B, out_channels, H/2, W/2]
        
        # 残差分支
        x_res = self.conv_res(x)  # [B, out_channels, H/2, W/2]
        
        # 逐元素相加
        out = x_model + x_res
        
        return out


class CAB(nn.Module):
    """
    列注意力分支 (Column Attention Branch)
    
    使用(H, 1)核的列池化来强化列特征
    """
    def __init__(self, channels):
        super(CAB, self).__init__()
        self.channels = channels
        
        # 共享的1x1卷积
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1, 1, 0),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 两个信道注意力分支
        self.ca_avg = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.Sigmoid()
        )
        
        self.ca_max = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 列平均池化和列最大池化 (H, 1)核
        x_avg = F.adaptive_avg_pool2d(x, (1, W))  # [B, C, 1, W]
        x_max = F.adaptive_max_pool2d(x, (1, W))  # [B, C, 1, W]
        
        # 拼接
        x_cat = torch.cat([x_avg, x_max], dim=1)  # [B, 2C, 1, W]
        
        # 共享卷积
        x_shared = self.shared_conv(x_cat)  # [B, 2C, 1, W]
        
        # 分割成两个分支
        x_a, x_m = torch.split(x_shared, C, dim=1)
        
        # 分别通过信道注意力
        weight_avg = self.ca_avg(x_a)  # [B, C, 1, W]
        weight_max = self.ca_max(x_m)  # [B, C, 1, W]
        
        # 广播并加权
        out = x * weight_avg + x * weight_max
        
        return out


class SAB(nn.Module):
    """
    空间注意力分支 (Spatial Attention Branch)
    
    通过全局池化增强关键区域的空间相关性
    """
    def __init__(self, channels):
        super(SAB, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        # 沿通道维度池化
        x_avg = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_max, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 拼接
        x_cat = torch.cat([x_avg, x_max], dim=1)  # [B, 2, H, W]
        
        # 生成空间掩码
        mask = self.conv(x_cat)  # [B, 1, H, W]
        
        # 加权
        out = x * mask
        
        return out


class SCB(nn.Module):
    """
    自校准分支 (Self-Calibrated Branch)
    
    通过下采样-卷积-上采样建立长程依赖
    """
    def __init__(self, channels):
        super(SCB, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            weight: [B, C, H, W] - 调制权重
        """
        B, C, H, W = x.shape
        
        # 下采样
        x_down = F.avg_pool2d(x, 2, 2)  # [B, C, H/2, W/2]
        
        # 卷积提取上下文
        x_context = self.conv(x_down)  # [B, C, H/2, W/2]
        
        # 上采样回原始分辨率
        x_up = F.interpolate(x_context, size=(H, W), mode='bilinear', align_corners=False)
        
        # 残差连接 + Sigmoid
        weight = torch.sigmoid(x + x_up)
        
        return weight


class RCSSC(nn.Module):
    """
    残差列空间自校正块 (Residual Column Spatial Self-Correction Block)
    
    融合CAB、SAB、SCB三个分支
    """
    def __init__(self, channels):
        super(RCSSC, self).__init__()
        
        self.cab = CAB(channels)
        self.sab = SAB(channels)
        self.scb = SCB(channels)
        
        # 融合1x1卷积
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        # 三个分支
        ca = self.cab(x)  # 列注意力
        sa = self.sab(x)  # 空间注意力
        sc = self.scb(x)  # 自校准权重
        
        # 融合CA和SA
        fusion = self.fusion_conv(torch.cat([sa, ca], dim=1))
        
        # 用SC调制
        out = fusion * sc
        
        # 残差连接
        out = out + x
        
        return out


class CNCM(nn.Module):
    """
    列非均匀性校正模块 (Column Non-uniformity Correction Module)
    
    堆叠多个RCSSC块进行特征增强
    """
    def __init__(self, channels, num_blocks=3):
        super(CNCM, self).__init__()
        
        self.num_blocks = num_blocks
        
        # 多个RCSSC块（顺序堆叠）
        self.blocks = nn.ModuleList([
            RCSSC(channels) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        out = x
        
        # 顺序通过所有RCSSC块
        for block in self.blocks:
            out = block(out)
        
        return out
