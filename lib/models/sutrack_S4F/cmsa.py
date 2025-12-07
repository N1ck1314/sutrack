"""
Cross-Modal Spatial Awareness (CMSA) Module
基于S4Fusion论文的跨模态空间感知模块

参考论文: S4Fusion: Saliency-Aware Selective State Space Model for Infrared and Visible Image Fusion
GitHub: https://github.com/zipper112/S4Fusion

核心功能：
1. 块标记（Patch Mark）：空间位置对齐
2. 交织（Interleaving）：模态信息空间融合
3. 交叉扫描：状态空间模型处理
4. 特征恢复：重建融合特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def rearrange(tensor, pattern, **axes_lengths):
    """简化版的einops.rearrange功能"""
    if pattern == 'b (h w) c -> b c h w':
        b, hw, c = tensor.shape
        h = axes_lengths['h']
        w = axes_lengths['w']
        return tensor.permute(0, 2, 1).reshape(b, c, h, w)
    elif pattern == 'b c h w -> b (h w) c':
        b, c, h, w = tensor.shape
        return tensor.reshape(b, c, -1).permute(0, 2, 1)
    elif pattern == 'b (n m) c -> b n m c':
        b, nm, c = tensor.shape
        n = nm // 2  # 假设m=2
        m = 2
        return tensor.reshape(b, n, m, c)
    elif pattern == 'b n m c -> b (n m) c':
        b, n, m, c = tensor.shape
        return tensor.reshape(b, n*m, c)
    elif pattern == 'b d l -> b l d':
        return tensor.permute(0, 2, 1)
    elif pattern == 'b l d -> b d l':
        return tensor.permute(0, 2, 1)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")


class PatchMark(nn.Module):
    """
    块标记模块：为不同模态的对应位置块分配相同标记
    实现空间位置的跨模态关联
    """
    def __init__(self, dim, num_patches):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        
        # 位置编码生成
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches, dim) * 0.02
        )
        
        # 模态区分编码
        self.modal_token_rgb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.modal_token_x = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
    def forward(self, rgb_feat, x_feat):
        """
        Args:
            rgb_feat: [B, N, C] RGB模态特征
            x_feat: [B, N, C] 第二模态特征（Depth/Thermal等）
        Returns:
            marked_rgb: [B, N, C] 标记后的RGB特征
            marked_x: [B, N, C] 标记后的第二模态特征
        """
        B, N, C = rgb_feat.shape
        
        # 添加位置编码
        pos_embed = self.position_embedding[:, :N, :]
        
        # 添加模态标记和位置编码
        marked_rgb = rgb_feat + pos_embed + self.modal_token_rgb
        marked_x = x_feat + pos_embed + self.modal_token_x
        
        return marked_rgb, marked_x


class Interleaving(nn.Module):
    """
    交织模块：将两种模态特征沿不同方向展平并交替合并
    实现模态信息的空间维度对齐
    """
    def __init__(self, h, w):
        super().__init__()
        self.h = h
        self.w = w
        
    def forward(self, rgb_feat, x_feat):
        """
        Args:
            rgb_feat: [B, N, C] where N = H*W
            x_feat: [B, N, C]
        Returns:
            interleaved: [B, 2*N, C] 交织后的序列
        """
        B, N, C = rgb_feat.shape
        H, W = self.h, self.w
        
        # 重塑为2D特征图
        rgb_2d = rearrange(rgb_feat, 'b (h w) c -> b c h w', h=H, w=W)
        x_2d = rearrange(x_feat, 'b (h w) c -> b c h w', h=H, w=W)
        
        # 四个方向扫描：水平、垂直、对角线等
        # 简化版本：水平和垂直交织
        
        # 水平方向交织
        h_rgb = rearrange(rgb_2d, 'b c h w -> b (h w) c')
        h_x = rearrange(x_2d, 'b c h w -> b (h w) c')
        
        # 交替合并：RGB, X, RGB, X, ...
        interleaved = torch.stack([h_rgb, h_x], dim=2)  # [B, N, 2, C]
        interleaved = rearrange(interleaved, 'b n m c -> b (n m) c')  # [B, 2N, C]
        
        return interleaved


class SelectiveStateSpace(nn.Module):
    """
    选择性状态空间模块（简化版SSM）
    基于Mamba/S4的核心思想，实现高效的序列建模
    """
    def __init__(self, dim, d_state=16, expand_factor=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_inner = int(dim * expand_factor)
        
        # 输入投影
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        
        # SSM参数（简化版本）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner
        )
        
        # 状态空间参数
        self.x_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        
        # 激活函数
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        Args:
            x: [B, L, C] 输入序列
        Returns:
            out: [B, L, C] 输出序列
        """
        B, L, C = x.shape
        
        # 输入投影并分离
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # [B, L, d_inner]
        
        # 卷积处理
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)
        x = rearrange(x, 'b d l -> b l d')
        x = self.act(x)
        
        # 简化的SSM处理（使用注意力机制近似）
        # 完整SSM需要复杂的状态空间计算
        dt = self.dt_proj(x)  # [B, L, d_inner]
        x = x * torch.sigmoid(dt)  # 门控机制
        
        # 门控输出
        x = x * self.act(z)
        
        # 输出投影
        out = self.out_proj(x)
        
        return out


class CrossSS2D(nn.Module):
    """
    交叉SS2D模块：基于状态空间模型的跨模态信息交互
    使用模态私有参数处理交织序列
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        
        # RGB模态的SSM
        self.ssm_rgb = SelectiveStateSpace(dim, d_state)
        
        # 第二模态的SSM
        self.ssm_x = SelectiveStateSpace(dim, d_state)
        
        # 跨模态交互
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, interleaved_feat):
        """
        Args:
            interleaved_feat: [B, 2N, C] 交织后的特征
        Returns:
            rgb_out: [B, N, C] 处理后的RGB特征
            x_out: [B, N, C] 处理后的第二模态特征
        """
        B, L, C = interleaved_feat.shape
        N = L // 2
        
        # 分离两个模态
        rgb_feat = interleaved_feat[:, 0::2, :]  # [B, N, C]
        x_feat = interleaved_feat[:, 1::2, :]    # [B, N, C]
        
        # 各自通过SSM处理
        rgb_ssm = self.ssm_rgb(rgb_feat)
        x_ssm = self.ssm_x(x_feat)
        
        # 跨模态注意力交互
        rgb_out, _ = self.cross_attn(
            query=rgb_ssm,
            key=x_ssm,
            value=x_ssm
        )
        rgb_out = self.norm(rgb_feat + rgb_out)
        
        x_out, _ = self.cross_attn(
            query=x_ssm,
            key=rgb_ssm,
            value=rgb_ssm
        )
        x_out = self.norm(x_feat + x_out)
        
        return rgb_out, x_out


class CMSA(nn.Module):
    """
    跨模态空间感知模块（Cross-Modal Spatial Awareness）
    
    完整流程：
    1. Patch Mark：空间位置标记
    2. Interleaving：模态交织
    3. Cross SS2D：状态空间交互
    4. Recovering：特征恢复
    """
    def __init__(self, dim, h, w, d_state=16, use_ssm=True):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.num_patches = h * w
        self.use_ssm = use_ssm
        
        # 1. 块标记
        self.patch_mark = PatchMark(dim, self.num_patches)
        
        # 2. 交织
        self.interleaving = Interleaving(h, w)
        
        # 3. 交叉SS2D
        if use_ssm:
            self.cross_ss2d = CrossSS2D(dim, d_state)
        else:
            # 如果不使用SSM，使用标准的交叉注意力
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=8,
                batch_first=True
            )
        
        # 4. 恢复层
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, rgb_feat, x_feat):
        """
        完整的CMSA前向传播
        
        Args:
            rgb_feat: [B, N, C] RGB模态特征（N = H*W）
            x_feat: [B, N, C] 第二模态特征（Depth/Thermal/Event等）
            
        Returns:
            fused_feat: [B, N, C] 融合后的特征
        """
        B, N, C = rgb_feat.shape
        identity_rgb = rgb_feat
        identity_x = x_feat
        
        # 1. Patch Mark：空间位置标记
        marked_rgb, marked_x = self.patch_mark(rgb_feat, x_feat)
        
        # 2. Interleaving：模态交织
        interleaved = self.interleaving(marked_rgb, marked_x)
        
        # 3. Cross SS2D：跨模态交互
        if self.use_ssm:
            rgb_out, x_out = self.cross_ss2d(interleaved)
        else:
            # 使用标准交叉注意力
            rgb_out, _ = self.cross_attn(marked_rgb, marked_x, marked_x)
            x_out, _ = self.cross_attn(marked_x, marked_rgb, marked_rgb)
        
        # 4. Recovering：特征恢复和融合
        rgb_out = self.norm1(rgb_out + identity_rgb)
        x_out = self.norm2(x_out + identity_x)
        
        # 自适应门控融合
        gate_weights = self.gate(torch.cat([rgb_out, x_out], dim=-1))
        fused_feat = gate_weights * rgb_out + (1 - gate_weights) * x_out
        
        # 或者使用拼接融合
        concat_feat = torch.cat([rgb_out, x_out], dim=-1)
        fused_feat = self.out_proj(concat_feat)
        
        return fused_feat


class MultiModalFusionWithCMSA(nn.Module):
    """
    多模态融合模块（集成CMSA）
    用于替代SUTrack中的简单torch.cat拼接
    """
    def __init__(self, dim, h, w, use_ssm=True, fusion_mode='cmsa'):
        super().__init__()
        self.dim = dim
        self.fusion_mode = fusion_mode
        
        if fusion_mode == 'cmsa':
            self.cmsa = CMSA(dim, h, w, use_ssm=use_ssm)
        elif fusion_mode == 'concat':
            # 简单拼接后的卷积融合
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
        
    def forward(self, rgb_feat, x_feat):
        """
        Args:
            rgb_feat: [B, C, H, W] 或 [B, N, C]
            x_feat: [B, C, H, W] 或 [B, N, C]
        Returns:
            fused: 融合后的特征
        """
        # 处理输入维度
        if rgb_feat.dim() == 4:  # [B, C, H, W]
            B, C, H, W = rgb_feat.shape
            rgb_feat = rearrange(rgb_feat, 'b c h w -> b (h w) c')
            x_feat = rearrange(x_feat, 'b c h w -> b (h w) c')
            reshape_back = True
        else:  # [B, N, C]
            reshape_back = False
            B, N, C = rgb_feat.shape
            H = W = int(math.sqrt(N))
        
        if self.fusion_mode == 'cmsa':
            fused = self.cmsa(rgb_feat, x_feat)
        elif self.fusion_mode == 'concat':
            # 简单拼接
            rgb_2d = rearrange(rgb_feat, 'b (h w) c -> b c h w', h=H, w=W)
            x_2d = rearrange(x_feat, 'b (h w) c -> b c h w', h=H, w=W)
            concat = torch.cat([rgb_2d, x_2d], dim=1)
            fused_2d = self.fusion_conv(concat)
            fused = rearrange(fused_2d, 'b c h w -> b (h w) c')
        
        # 恢复原始维度
        if reshape_back:
            fused = rearrange(fused, 'b (h w) c -> b c h w', h=H, w=W)
        
        return fused
