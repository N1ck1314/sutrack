"""
SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention
Reference: https://arxiv.org/pdf/2407.05128

包含两个核心模块:
1. SMSA - Shareable Multi-Semantic Spatial Attention (共享多语义空间注意力)
2. PCSA - Progressive Channel-wise Self-Attention (渐进式通道自注意力)
"""
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F


class Shareable_Multi_Semantic_Spatial_Attention(nn.Module):
    """
    共享多语义空间注意力 (SMSA)
    
    通过通道分组 + 多尺度共享的 1D depthwise 卷积，
    在不引入高计算量的情况下显式建模不同语义层级的空间结构。
    """
    def __init__(
            self,
            dim: int,  # 输入特征的维度
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],  # 分组卷积核的大小
            gate_layer: str = 'sigmoid',  # 门控层的激活函数类型
    ):
        super(Shareable_Multi_Semantic_Spatial_Attention, self).__init__()
        self.dim = dim

        # 确保输入特征的维度能被4整除
        assert self.dim % 4 == 0, '输入特征的维度应能被4整除。'
        # 计算每个分组的通道数
        self.group_chans = group_chans = self.dim // 4

        # 定义局部深度可分离卷积层，用于处理局部特征
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        # 定义小尺寸全局深度可分离卷积层，用于捕捉小范围的全局特征
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        # 定义中尺寸全局深度可分离卷积层，用于捕捉中等范围的全局特征
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        # 定义大尺寸全局深度可分离卷积层，用于捕捉大范围的全局特征
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        # 根据门控层类型选择激活函数，用于生成注意力权重
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        # 定义用于高度方向的组归一化层，对特征进行归一化处理
        self.norm_h = nn.GroupNorm(4, dim)
        # 定义用于宽度方向的组归一化层，对特征进行归一化处理
        self.norm_w = nn.GroupNorm(4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的批量大小、通道数、高度和宽度
        b, c, h_, w_ = x.size()
        
        # 在宽度维度上求平均，得到高度方向的特征
        x_h = x.mean(dim=3)
        # 将高度方向的特征按通道分组
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)

        # 在高度维度上求平均，得到宽度方向的特征
        x_w = x.mean(dim=2)
        # 将宽度方向的特征按通道分组
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        # 计算高度方向的注意力图，先将分组后的特征拼接，再进行归一化和激活操作
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        # 调整高度方向注意力图的形状，使其与输入特征的维度匹配
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        # 计算宽度方向的注意力图，先将分组后的特征拼接，再进行归一化和激活操作
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        # 调整宽度方向注意力图的形状，使其与输入特征的维度匹配
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        # 将输入特征与高度和宽度方向的注意力图相乘，增强重要区域的特征
        # 使用 * 创建新张量，避免原地操作
        out = x * x_h_attn * x_w_attn

        return out


class Progressive_Channel_wise_Self_Attention(nn.Module):
    """
    渐进式通道自注意力 (PCSA)
    
    引入通道维度上的单头自注意力，并结合渐进压缩策略，
    在保留空间先验的同时显式建模通道相似性，缓解多语义子特征之间的不一致问题。
    """
    def __init__(
            self,
            dim: int,  # 输入特征的通道数
            reduction_ratio: int = 4,  # 通道压缩比例
    ):
        super(Progressive_Channel_wise_Self_Attention, self).__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        self.reduced_dim = max(dim // reduction_ratio, 32)  # 确保压缩后至少有32个通道
        
        # 渐进式通道压缩
        self.compress = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, self.reduced_dim, 1, bias=False),
            nn.BatchNorm2d(self.reduced_dim),
            nn.ReLU(inplace=True)
        )
        
        # Channel-wise self-attention (QKV for channel dimension)
        self.query_conv = nn.Conv2d(self.reduced_dim, self.reduced_dim, 1)
        self.key_conv = nn.Conv2d(self.reduced_dim, self.reduced_dim, 1)
        self.value_conv = nn.Conv2d(self.reduced_dim, self.reduced_dim, 1)
        
        # 恢复通道维度
        self.expand = nn.Conv2d(self.reduced_dim, dim, 1, bias=False)
        
        # 注意力缩放因子
        self.scale = self.reduced_dim ** -0.5
        
        # Sigmoid门控
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # 渐进式压缩: [B, C, H, W] -> [B, C', 1, 1]
        compressed = self.compress(x)
        
        # 生成 Q, K, V: [B, C', 1, 1]
        query = self.query_conv(compressed).view(b, self.reduced_dim, 1)  # [B, C', 1]
        key = self.key_conv(compressed).view(b, self.reduced_dim, 1)  # [B, C', 1]
        value = self.value_conv(compressed).view(b, self.reduced_dim, 1)  # [B, C', 1]
        
        # Channel-wise self-attention: [B, C', C']
        # Q^T @ K
        attn = torch.bmm(query.transpose(1, 2), key) * self.scale  # [B, 1, 1]
        attn = F.softmax(attn, dim=-1)
        
        # Attention @ V
        out = torch.bmm(value, attn)  # [B, C', 1]
        out = out.view(b, self.reduced_dim, 1, 1)
        
        # 恢复通道维度: [B, C', 1, 1] -> [B, C, 1, 1]
        out = self.expand(out)
        
        # Sigmoid门控并广播到空间维度
        ca_weight = self.sigmoid(out)
        
        # 应用通道注意力 (避免原地操作)
        out = x * ca_weight
        
        return out


class SCSA(nn.Module):
    """
    SCSA模块: 空间-通道协同注意力
    
    组合形式: SCSA(X) = PCSA(SMSA(X))
    - 先应用SMSA提取多语义空间注意力
    - 再应用PCSA缓解多语义差异并建模通道相关性
    """
    def __init__(
            self,
            dim: int,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            gate_layer: str = 'sigmoid',
            reduction_ratio: int = 4,
    ):
        super(SCSA, self).__init__()
        
        # 空间注意力 (SMSA)
        self.smsa = Shareable_Multi_Semantic_Spatial_Attention(
            dim=dim,
            group_kernel_sizes=group_kernel_sizes,
            gate_layer=gate_layer
        )
        
        # 通道注意力 (PCSA)
        self.pcsa = Progressive_Channel_wise_Self_Attention(
            dim=dim,
            reduction_ratio=reduction_ratio
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 协同注意力: 空间引导 -> 通道缓解
        x = self.smsa(x)
        x = self.pcsa(x)
        return x


if __name__ == '__main__':
    # 测试SMSA
    print("Testing SMSA...")
    smsa = Shareable_Multi_Semantic_Spatial_Attention(dim=32)
    input_smsa = torch.randn(1, 32, 50, 50)
    output_smsa = smsa(input_smsa)
    print(f'SMSA Input size: {input_smsa.size()}')
    print(f'SMSA Output size: {output_smsa.size()}')
    
    # 测试PCSA
    print("\nTesting PCSA...")
    pcsa = Progressive_Channel_wise_Self_Attention(dim=32, reduction_ratio=4)
    input_pcsa = torch.randn(1, 32, 50, 50)
    output_pcsa = pcsa(input_pcsa)
    print(f'PCSA Input size: {input_pcsa.size()}')
    print(f'PCSA Output size: {output_pcsa.size()}')
    
    # 测试完整SCSA
    print("\nTesting SCSA...")
    scsa = SCSA(dim=32, reduction_ratio=4)
    input_scsa = torch.randn(1, 32, 50, 50)
    output_scsa = scsa(input_scsa)
    print(f'SCSA Input size: {input_scsa.size()}')
    print(f'SCSA Output size: {output_scsa.size()}')
