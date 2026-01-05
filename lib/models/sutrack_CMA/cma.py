"""
Cross-Modal Attention (CMA) Block
用于融合RGB特征和频域特征的跨模态注意力机制
"""
import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class CMA_Block(nn.Module):
    """
    Cross-Modal Attention Block for fusing RGB and frequency domain features.
    
    Args:
        in_channel: 输入通道数
        hidden_channel: 隐藏层通道数
        out_channel: 输出通道数
    """
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(CMA_Block, self).__init__()

        # Query projection for RGB features
        self.conv1 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        # Key projection for frequency features
        self.conv2 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        # Value projection for frequency features
        self.conv3 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )

        # Attention scaling factor
        self.scale = hidden_channel ** -0.5

        # Output projection with normalization and activation
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                hidden_channel, out_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        """
        Args:
            rgb: RGB特征 (B, C, H, W)
            freq: 频域特征 (B, C, H, W)
        Returns:
            output: 融合后的特征 (B, C, H, W)
        """
        _, _, h, w = rgb.size()

        # Project to Q, K, V
        q = self.conv1(rgb)      # Query from RGB
        k = self.conv2(freq)     # Key from frequency
        v = self.conv3(freq)     # Value from frequency

        # Reshape for attention computation
        # q: (B, C, H*W) -> (B, H*W, C)
        q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3)).transpose(-2, -1)
        # k: (B, C, H*W)
        k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3))

        # Compute attention scores: Q @ K^T
        attn = torch.matmul(q, k) * self.scale  # (B, H*W, H*W)
        m = attn.softmax(dim=-1)

        # Apply attention to values
        # v: (B, C, H*W) -> (B, H*W, C)
        v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3)).transpose(-2, -1)
        # z: (B, H*W, C)
        z = torch.matmul(m, v)
        
        # Reshape back to spatial format
        z = z.view(z.size(0), h, w, -1)
        z = z.permute(0, 3, 1, 2).contiguous()

        # Residual connection + output projection
        output = rgb + self.conv4(z)

        return output


class FrequencyFilter(nn.Module):
    """
    频域滤波器，用于从RGB特征中提取频域信息
    """
    def __init__(self, channels):
        super(FrequencyFilter, self).__init__()
        self.channels = channels
        
        # 可学习的频域权重
        self.freq_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            freq_feature: 频域特征 (B, C, H, W)
        """
        # 2D FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # 应用可学习的频域权重
        x_fft = x_fft * self.freq_weight
        
        # 逆FFT回到空间域
        freq_feature = torch.fft.irfft2(x_fft, s=(x.size(2), x.size(3)), norm='ortho')
        
        return freq_feature


class CMA_Module(nn.Module):
    """
    CMA模块，包含频域滤波器和跨模态注意力块
    用于在SUTrack模型中融合空间域和频域特征
    """
    def __init__(self, in_channel, hidden_channel=None, out_channel=None):
        super(CMA_Module, self).__init__()
        
        if hidden_channel is None:
            hidden_channel = in_channel // 2
        if out_channel is None:
            out_channel = in_channel
            
        self.freq_filter = FrequencyFilter(in_channel)
        self.cma_block = CMA_Block(in_channel, hidden_channel, out_channel)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, C, H, W)
        Returns:
            output: 融合后的特征 (B, C, H, W)
        """
        # 提取频域特征
        freq_feature = self.freq_filter(x)
        
        # 跨模态注意力融合
        output = self.cma_block(x, freq_feature)
        
        return output


if __name__ == '__main__':
    # 测试代码
    in_channel = 64
    hidden_channel = 32
    out_channel = 64
    h = 64
    w = 64

    # 测试CMA_Block
    block = CMA_Block(in_channel, hidden_channel, out_channel)
    rgb_input = torch.rand(1, in_channel, h, w)
    freq_input = torch.rand(1, in_channel, h, w)
    output = block(rgb_input, freq_input)
    print("CMA_Block Test:")
    print("  RGB Input size:", rgb_input.size())
    print("  Freq Input size:", freq_input.size())
    print("  Output size:", output.size())
    
    # 测试CMA_Module
    print("\nCMA_Module Test:")
    module = CMA_Module(in_channel, hidden_channel, out_channel)
    x = torch.rand(2, in_channel, h, w)
    output = module(x)
    print("  Input size:", x.size())
    print("  Output size:", output.size())
