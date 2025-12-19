"""
RMT (Retentive Multi-scale Transformer) Module
基于RetNet的视觉Retention机制，实现高效的空间建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RetNetRelPos2d(nn.Module):
    """二维相对位置编码，用于视觉Retention"""
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        # 计算位置编码的角度，基于论文中常用的公式
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        # 计算衰减因子
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        """生成2D衰减矩阵"""
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w], indexing='ij')
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = mask.abs().sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_decay(self, l: int):
        """生成1D衰减矩阵"""
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        """
        前向传播
        Args:
            slen: (H, W) 特征图尺寸
            activate_recurrent: 是否启用循环模式
            chunkwise_recurrent: 是否启用分块循环模式
        """
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(slen[0], slen[1], -1)
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])
            retention_rel_pos = ((sin, cos), (mask_h, mask_w))
        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(slen[0], slen[1], -1)
            mask = self.generate_2d_decay(slen[0], slen[1])
            retention_rel_pos = ((sin, cos), mask)
        return retention_rel_pos


def rotate_every_two(x):
    """旋转操作：用于位置编码"""
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    out = x.flatten(-2)
    return out


def theta_shift(x, sin, cos):
    """使用sin和cos对输入特征进行位置调制"""
    return (x * cos) + (rotate_every_two(x) * sin)


class DWConv2d(nn.Module):
    """深度卷积，用于局部特征增强"""
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        输入: x (B, H, W, C)
        输出: (B, H, W, C)
        """
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        return x


class VisionRetentionChunk(nn.Module):
    """
    视觉Retention模块（分块版本）
    Manhattan Self-Attention with Decomposed MaSA
    """
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        
        # 局部位置编码增强（LCE）
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos):
        """
        前向传播
        Args:
            x: (B, H, W, C) 输入特征
            rel_pos: 相对位置编码
        Returns:
            output: (B, H, W, C) 输出特征
        """
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos
        
        # 投影Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 局部增强
        lepe = self.lepe(v)
        
        k *= self.scaling
        
        # 重塑为多头格式
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        
        # 应用位置调制
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)
        
        # 宽度方向的注意力（Decomposed MaSA - Width）
        qr_w = qr.transpose(1, 2)  # (B, H, num_heads, W, key_dim)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (B, H, num_heads, W, W)
        qk_mat_w = qk_mat_w + mask_w
        qk_mat_w = torch.softmax(qk_mat_w, dim=-1)
        v = torch.matmul(qk_mat_w, v)
        
        # 高度方向的注意力（Decomposed MaSA - Height）
        qr_h = qr.permute(0, 3, 1, 2, 4)  # (B, W, num_heads, H, key_dim)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)  # (B, W, num_heads, H, head_dim)
        
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (B, W, num_heads, H, H)
        qk_mat_h = qk_mat_h + mask_h
        qk_mat_h = torch.softmax(qk_mat_h, dim=-1)
        output = torch.matmul(qk_mat_h, v)
        
        # 重塑回原始维度
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (B, H, W, C)
        
        # 添加局部增强
        output = output + lepe
        
        # 输出投影
        output = self.out_proj(output)
        
        return output


class RMTBlock(nn.Module):
    """
    RMT Block: Vision Retention + FFN
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.retention = VisionRetentionChunk(dim, num_heads)
        
        # FFN
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        
        # DropPath
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, rel_pos):
        """
        Args:
            x: (B, H, W, C)
            rel_pos: 相对位置编码
        """
        # Retention
        x = x + self.drop_path(self.retention(self.norm1(x), rel_pos))
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
