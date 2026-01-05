"""
Adaptive Sparse Self-Attention (ASSA) modules for SUTrack
Based on paper: "Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration"
GitHub: https://github.com/joshyZhou/AST

核心模块:
1. ASSA (Adaptive Sparse Self-Attention): 自适应稀疏注意力
2. FRFN (Feature Refinement Feed-forward Network): 特征细化前馈网络
3. TVConv (Spatially-Variant Convolution): 空间变体卷积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class _ConvBlock(nn.Sequential):
    """
    卷积块：Conv + LayerNorm + ReLU
    用于 TVConv 的权重生成网络
    """
    def __init__(self, in_planes, out_planes, h, w, kernel_size=3, stride=1, bias=False):
        padding = (kernel_size - 1) // 2
        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),
            nn.LayerNorm([out_planes, h, w]),
            nn.ReLU(inplace=True)
        )


class TVConv(nn.Module):
    """
    Spatially-Variant Convolution (空间变体卷积)
    
    核心思想:
    - 通过 Affinity Maps 捕获输入图像中不同空间区域的固有属性
    - 使用位置映射 (position map) 生成位置特定的卷积核权重
    - 训练时生成核可根据这些亲和性对齐自适应调整
    
    优势:
    - 比标准卷积更灵活,能根据图像内容自适应调整
    - 参数量增加较少
    """
    def __init__(self,
                 channels,
                 TVConv_k=3,
                 stride=1,
                 TVConv_posi_chans=4,
                 TVConv_inter_chans=64,
                 TVConv_inter_layers=3,
                 TVConv_Bias=False,
                 h=14,
                 w=14,
                 **kwargs):
        super(TVConv, self).__init__()

        # 注册缓冲区变量
        self.register_buffer("TVConv_k", torch.as_tensor(TVConv_k))
        self.register_buffer("TVConv_k_square", torch.as_tensor(TVConv_k**2))
        self.register_buffer("stride", torch.as_tensor(stride))
        self.register_buffer("channels", torch.as_tensor(channels))
        self.register_buffer("h", torch.as_tensor(h))
        self.register_buffer("w", torch.as_tensor(w))

        self.bias_layers = None
        out_chans = self.TVConv_k_square * self.channels

        # 位置映射参数
        self.posi_map = nn.Parameter(torch.Tensor(1, TVConv_posi_chans, h, w))
        nn.init.ones_(self.posi_map)

        # 权重生成网络
        self.weight_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, out_chans, TVConv_inter_layers, h, w)
        if TVConv_Bias:
            self.bias_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, channels, TVConv_inter_layers, h, w)

        # Unfold 用于提取局部区域
        self.unfold = nn.Unfold(TVConv_k, 1, (TVConv_k-1)//2, stride)

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers, h, w):
        """创建卷积层序列"""
        layers = [_ConvBlock(in_chans, inter_chans, h, w, bias=False)]
        for i in range(num_inter_layers):
            layers.append(_ConvBlock(inter_chans, inter_chans, h, w, bias=False))
        layers.append(nn.Conv2d(
            in_channels=inter_chans,
            out_channels=out_chans,
            kernel_size=3,
            padding=1,
            bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        # 生成位置特定的卷积权重
        weight = self.weight_layers(self.posi_map)
        weight = weight.view(1, self.channels, self.TVConv_k_square, self.h, self.w)
        
        # Unfold 提取局部区域并应用权重
        out = self.unfold(x).view(x.shape[0], self.channels, self.TVConv_k_square, self.h, self.w)
        out = (weight * out).sum(dim=2)  # 加权求和
        
        if self.bias_layers is not None:
            bias = self.bias_layers(self.posi_map)
            out = out + bias
        return out


class FRFN(nn.Module):
    """
    Feature Refinement Feed-forward Network
    
    核心机制: Enhance-and-Ease (增强 + 缓解)
    
    1. Enhance: 使用 TVConv 强化关键信息
    2. Ease: 通过通道压缩缓解冗余
    
    与 ASSA 在空间维度的稀疏互补
    整体从"选 token"升级为"选信息流"
    """
    def __init__(self, dim, h=14, w=14, ffn_expansion_factor=2.66, bias=False):
        super(FRFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # 第一层: 扩展通道
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # TVConv: 空间变体卷积用于特征细化
        self.dwconv = TVConv(
            channels=hidden_features * 2,
            h=h,
            w=w,
            TVConv_k=3,
            TVConv_posi_chans=4,
            TVConv_inter_chans=min(64, hidden_features // 2),
            TVConv_inter_layers=2
        )

        # 第二层: 压缩通道 (Ease 机制)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, C, H, W)
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2  # Gate机制
        x = self.project_out(x)
        return x


class ASSA(nn.Module):
    """
    Adaptive Sparse Self-Attention
    
    核心创新:
    1. 双分支结构:
       - Sparse Branch (ReLU²): 抑制低相关区域
       - Dense Branch (Softmax): 保持信息完整性
    
    2. 自适应加权融合:
       A = (ω₁·SSA + ω₂·DSA) V
       
    3. 优势:
       - 不依赖 Top-K / 超像素 / 固定阈值
       - 自适应学习"哪些关系该稀疏、哪些该保留"
       - 泛化性强,适用于多种低层视觉任务
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., h=14, w=14):
        super(ASSA, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.h = h
        self.w = w

        # QKV 映射
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 自适应权重参数 (可学习)
        self.alpha = nn.Parameter(torch.ones(1))  # Sparse branch 权重
        self.beta = nn.Parameter(torch.ones(1))   # Dense branch 权重
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: (B, N, C) where N = H*W
        Returns:
            x: (B, N, C)
        """
        B, N, C = x.shape
        
        # 生成 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)

        # === Sparse Branch: ReLU² Attention ===
        # ReLU² 会抑制负值和小正值,保留高置信交互
        attn_sparse = F.relu(attn) ** 2
        attn_sparse = attn_sparse / (attn_sparse.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化
        attn_sparse = self.attn_drop(attn_sparse)
        
        # === Dense Branch: Softmax Attention ===
        # 标准 Softmax 保证全局信息流
        attn_dense = F.softmax(attn, dim=-1)
        attn_dense = self.attn_drop(attn_dense)

        # === 自适应加权融合 ===
        # 归一化权重
        alpha_norm = torch.sigmoid(self.alpha)
        beta_norm = torch.sigmoid(self.beta)
        weight_sum = alpha_norm + beta_norm + 1e-8
        
        attn_final = (alpha_norm / weight_sum) * attn_sparse + (beta_norm / weight_sum) * attn_dense

        # 应用注意力到 V
        x = (attn_final @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class ASSA_TransformerBlock(nn.Module):
    """
    ASSA Transformer Block
    
    结构:
    x → LN → ASSA → + → LN → FRFN → +
    ↓________________↑       ↓________↑
    
    核心组件:
    1. ASSA: 自适应稀疏注意力 (空间维度稀疏)
    2. FRFN: 特征细化前馈网络 (通道维度压缩)
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., h=14, w=14,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(ASSA_TransformerBlock, self).__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = ASSA(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            attn_drop=attn_drop, 
            proj_drop=drop,
            h=h,
            w=w
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        self.mlp = FRFN(dim=dim, h=h, w=w, ffn_expansion_factor=mlp_ratio, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, N, C)
        Returns:
            x: (B, N, C)
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        # ASSA
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # FRFN (需要转换为 4D)
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        x_2d = self.mlp(x_2d)
        x = x + self.drop_path(x_2d.flatten(2).transpose(1, 2))
        
        return x


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("Testing ASSA Modules")
    print("="*60)
    
    # 测试 TVConv
    print("\n[1] Testing TVConv...")
    x = torch.rand(2, 64, 14, 14)
    tvconv = TVConv(channels=64, h=14, w=14)
    out = tvconv(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in tvconv.parameters()) / 1e6:.2f}M")
    
    # 测试 FRFN
    print("\n[2] Testing FRFN...")
    x = torch.rand(2, 384, 14, 14)
    frfn = FRFN(dim=384, h=14, w=14)
    out = frfn(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in frfn.parameters()) / 1e6:.2f}M")
    
    # 测试 ASSA
    print("\n[3] Testing ASSA...")
    x = torch.rand(2, 196, 384)  # (B, N, C)
    assa = ASSA(dim=384, num_heads=8, h=14, w=14)
    out = assa(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in assa.parameters()) / 1e6:.2f}M")
    print(f"   Alpha: {assa.alpha.item():.4f}, Beta: {assa.beta.item():.4f}")
    
    # 测试 ASSA_TransformerBlock
    print("\n[4] Testing ASSA_TransformerBlock...")
    x = torch.rand(2, 196, 384)
    block = ASSA_TransformerBlock(dim=384, num_heads=8, h=14, w=14)
    out = block(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in block.parameters()) / 1e6:.2f}M")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
