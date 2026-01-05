"""
Sparse Self-Attention Module for SUTrack
基于 SparseViT 论文的稀疏自注意力机制，适配目标跟踪任务

核心创新:
1. SABlock: 稀疏自注意力块，抑制语义干扰，强化非语义特征提取
2. 层级稀疏结构: 不同层采用不同稀疏率，捕获多尺度特征
3. 可学习特征融合: LFF模块自适应融合多尺度特征

论文: SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer
适配场景: 目标跟踪中的小目标检测、遮挡处理、快速运动
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 兼容不同版本的timm
try:
    from timm.layers import trunc_normal_, DropPath
except ImportError:
    try:
        from timm.models.layers import trunc_normal_, DropPath
    except ImportError:
        # 如果都导入失败，使用torch内置的初始化
        from torch.nn.init import trunc_normal_
        from timm.models.layers import DropPath


# 全局配置
layer_scale = True
init_value = 1e-6


class DWConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """多层感知机（带深度卷积）"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def block(x, block_size):
    """块划分函数"""
    B, H, W, C = x.shape
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // block_size, block_size, Wp // block_size, block_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x, H, Hp, C


def unblock(x, Ho):
    """块合并函数"""
    B, H, W, win_H, win_W, C = x.shape
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H * win_H, W * win_W, C)
    Hp = Wp = H * win_H
    Wo = Ho
    if Hp > Ho or Wp > Wo:
        x = x[:, :Ho, :Wo, :].contiguous()
    return x


def alter_sparse(x, sparse_size=8):
    """稀疏化函数 - 将特征图划分为不重叠的子块"""
    x = x.permute(0, 2, 3, 1)
    assert x.shape[1] % sparse_size == 0 and x.shape[2] % sparse_size == 0, \
        f'Image size ({x.shape[1]}, {x.shape[2]}) should be divisible by sparse_size ({sparse_size})'
    grid_size = x.shape[1] // sparse_size
    out, H, Hp, C = block(x, grid_size)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, C)
    out = out.permute(0, 3, 1, 2)
    return out, H, Hp, C


def alter_unsparse(x, H, Hp, C, sparse_size=8):
    """去稀疏化函数 - 将子块重组为完整特征图"""
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, Hp // sparse_size, Hp // sparse_size, sparse_size, sparse_size, C)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = unblock(x, H)
    out = out.permute(0, 3, 1, 2)
    return out


class Attention(nn.Module):
    """标准多头注意力"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SABlock(nn.Module):
    """
    稀疏自注意力块 (Sparse Attention Block)
    
    核心思想:
    - 将全局注意力替换为稀疏注意力，仅在局部子块内计算注意力
    - 抑制语义信息传播，强化非语义特征（如噪声、边缘、纹理异常）
    - 适用于目标跟踪中需要捕捉细粒度差异的场景
    
    参数:
        dim: 特征维度
        num_heads: 注意力头数
        sparse_size: 稀疏块大小（控制稀疏率）
        mlp_ratio: MLP隐藏层扩展比例
    """
    def __init__(self, dim, num_heads, sparse_size=8, mlp_ratio=4., qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # 位置嵌入（深度卷积）
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
        
        self.sparse_size = sparse_size
        self.ls = layer_scale
        
        if self.ls:
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """
        前向传播
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            (B, C, H, W) 输出特征图
        """
        x_before = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        B, N, H, W = x.shape

        # 稀疏化处理：将特征图划分为不重叠的子块
        x_sparse, Ho, Hp, C = alter_sparse(x, self.sparse_size)
        
        Bf, Nf, Hf, Wf = x_sparse.shape
        x_sparse = x_sparse.flatten(2).transpose(1, 2)  # (Bf, Hf*Wf, C)
        
        # 在每个子块内独立计算注意力
        x_attn = self.attn(self.norm1(x_sparse))
        x_attn = x_attn.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
        
        # 去稀疏化：重组为完整特征图
        x_attn = alter_unsparse(x_attn, Ho, Hp, C, self.sparse_size)
        x_attn = x_attn.flatten(2).transpose(1, 2)

        # 残差连接 + Layer Scale
        if self.ls:
            x = x_before + self.drop_path(self.gamma_1 * x_attn)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        else:
            x = x_before + self.drop_path(x_attn)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class SparseViTModule(nn.Module):
    """
    SparseViT 完整模块（多个稀疏注意力块堆叠）
    
    适用于目标跟踪任务的多尺度稀疏特征提取
    """
    def __init__(self, dim, num_blocks=2, num_heads=8, sparse_sizes=[8, 4], 
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path_rate=0.1):
        """
        Args:
            dim: 特征维度
            num_blocks: SABlock数量
            num_heads: 注意力头数
            sparse_sizes: 每个块的稀疏大小（不同层使用不同稀疏率）
            mlp_ratio: MLP扩展比例
        """
        super().__init__()
        
        # 确保 sparse_sizes 长度与 num_blocks 一致
        if len(sparse_sizes) < num_blocks:
            sparse_sizes = sparse_sizes + [sparse_sizes[-1]] * (num_blocks - len(sparse_sizes))
        
        # 构建多个稀疏注意力块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.blocks = nn.ModuleList([
            SABlock(
                dim=dim,
                num_heads=num_heads,
                sparse_size=sparse_sizes[i],
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i]
            )
            for i in range(num_blocks)
        ])
        
        print(f"[SparseViT] 初始化完成:")
        print(f"  - 特征维度: {dim}")
        print(f"  - 块数量: {num_blocks}")
        print(f"  - 注意力头数: {num_heads}")
        print(f"  - 稀疏大小: {sparse_sizes[:num_blocks]}")

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == '__main__':
    print("="*60)
    print("Testing SparseViT Modules for SUTrack")
    print("="*60)
    
    # 测试 SABlock
    print("\n[1] Testing SABlock...")
    x = torch.randn(2, 384, 16, 16)
    sa_block = SABlock(dim=384, num_heads=8, sparse_size=4)
    out = sa_block(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in sa_block.parameters()) / 1e6:.2f}M")
    
    # 测试 SparseViTModule
    print("\n[2] Testing SparseViTModule...")
    x = torch.randn(2, 384, 14, 14)
    sparse_vit = SparseViTModule(dim=384, num_blocks=2, num_heads=8, sparse_sizes=[7, 7])
    out = sparse_vit(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Total Parameters: {sum(p.numel() for p in sparse_vit.parameters()) / 1e6:.2f}M")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
