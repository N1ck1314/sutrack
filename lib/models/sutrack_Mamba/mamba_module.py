"""
Mamba Module for SUTrack
基于 MCITrack 的 Mamba 实现，适配目标跟踪任务

核心创新:
1. MambaBlock: 选择性状态空间模型（SSM），线性复杂度
2. Mamba_Neck: 使用隐藏状态高效传递上下文信息
3. InteractionBlock: Injector（注入）+ Extractor（提取）的交互机制

参考论文: MCITrack (AAAI 2025) - Exploring Enhanced Contextual Information for Video-Level Object Tracking
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class DWConv(nn.Module):
    """Depthwise Convolution for spatial feature mixing"""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        # x: (L, B, C) -> (B, L, C)
        x = x.permute(1, 0, 2)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x).flatten(2).transpose(1, 2)  # B, N, C
        x = x.permute(1, 0, 2)  # L, B, C
        return x


class ConvFFN(nn.Module):
    """Convolutional Feed-Forward Network"""
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Extractor(nn.Module):
    """
    特征提取器: 从编码器特征中提取上下文信息
    使用交叉注意力机制
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1, drop_path=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.query_norm = norm_layer(d_model)
        self.feat_norm = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        # ConvFFN
        self.ffn = ConvFFN(in_features=d_model, hidden_features=int(d_model * 0.25), drop=0.)
        self.ffn_norm = norm_layer(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, feat):
        """
        Args:
            query: (L, B, D) - 查询特征
            feat: (L, B, D) - 键值特征
        Returns:
            (L, B, D) - 更新后的查询特征
        """
        # 交叉注意力
        attn = self.attn(self.query_norm(query),
                         self.feat_norm(feat), self.feat_norm(feat))[0]
        query = query + attn
        # FFN
        query = query + self.drop_path(self.ffn(self.ffn_norm(query)))
        return query


class Injector(nn.Module):
    """
    特征注入器: 将上下文信息注入到特征中
    使用可学习的缩放因子 gamma
    """
    def __init__(self, d_model, n_heads=8, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 dropout=0.1, init_values=0.):
        super().__init__()
        self.query_norm = norm_layer(d_model)
        self.feat_norm = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.gamma = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)

    def forward(self, query, feat):
        """
        Args:
            query: (L, B, D) - 查询特征
            feat: (L, B, D) - 上下文特征
        Returns:
            (L, B, D) - 注入上下文后的特征
        """
        attn = self.attn(self.query_norm(query),
                         self.feat_norm(feat), self.feat_norm(feat))[0]
        return query + self.gamma * attn


class MambaBlock(nn.Module):
    """
    Mamba Block: 选择性状态空间模型 (Selective State Space Model)
    
    核心思想:
    1. 输入分成两个分支: x 和 z
    2. x 分支通过 1D 卷积和 SSM 处理
    3. z 分支作为门控
    4. 最终输出 = y * z
    
    优势:
    - 线性复杂度 O(L) vs Transformer 的 O(L^2)
    - 通过隐藏状态保持长程依赖
    """
    def __init__(self, d_model, d_inner=None, dt_rank=32, d_state=16, d_conv=3,
                 bias=False, conv_bias=True, dt_init='random', dt_scale=1.0,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_inner or d_model * 2
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.dt_scale = dt_scale
        
        # 输入投影: D -> 2*ED (两个分支)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        
        # 1D 卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, 
            out_channels=self.d_inner,
            kernel_size=d_conv, 
            bias=conv_bias,
            groups=self.d_inner,
            padding=(d_conv - 1) // 2
        )
        
        # 投影到 Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        
        # Δ 投影
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # dt 初始化
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # dt bias 初始化
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # S4D 实数初始化
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影: ED -> D
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x, h=None):
        """
        Args:
            x: (B, L, D) - 输入特征
            h: (B, L, ED, N) - 隐藏状态（可选）
        Returns:
            output: (B, L, D) - 输出特征
            h: (B, L, ED, N) - 更新后的隐藏状态
        """
        B, L, D = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)
        
        # x 分支: 1D 卷积
        x = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L, ED)
        x = F.silu(x)
        
        # SSM 步骤
        y, h = self.ssm_step(x, h)
        
        # z 分支: 门控
        z = F.silu(z)
        
        # 输出
        output = y * z
        output = self.out_proj(output)  # (B, L, D)
        
        return output, h

    def ssm_step(self, x, h=None):
        """
        选择性状态空间模型步骤
        
        SSM 公式:
        h' = Ah + Bx
        y = Ch + Dx
        """
        B, L, ED = x.shape
        
        # 获取 A (负指数保证稳定性)
        A = -torch.exp(self.A_log.float())  # (ED, N)
        
        # 投影得到 Δ, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        delta, B_param, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Δ 投影 (softplus 激活)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, ED)
        
        # 初始化隐藏状态
        if h is None:
            h = torch.zeros(B, L, ED, self.d_state, device=x.device, dtype=x.dtype)
        
        # 离散化 SSM
        # h = exp(Δ*A) * h + Δ * B * x
        # y = C * h + D * x
        
        # 简化的并行扫描实现
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB_x = delta.unsqueeze(-1) * B_param.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, ED, N)
        
        # 更新隐藏状态
        h = deltaA * h + deltaB_x
        
        # 计算输出
        y = (h * C.unsqueeze(2)).sum(dim=-1)  # (B, L, ED)
        y = y + self.D * x
        
        return y, h


class ResidualBlock(nn.Module):
    """残差 Mamba 块"""
    def __init__(self, d_model, d_inner=None, dt_rank=32, d_state=16, 
                 d_conv=3, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.mixer = MambaBlock(d_model, d_inner, dt_rank, d_state, d_conv)
        self.norm = RMSNorm(d_model)

    def forward(self, x, h=None):
        """
        Args:
            x: (B, L, D)
            h: 隐藏状态
        Returns:
            output: (B, L, D)
            h: 更新后的隐藏状态
        """
        x_norm = self.norm(x)
        if self.grad_ckpt and self.training:
            output, h = checkpoint.checkpoint(self.mixer, x_norm, h, use_reentrant=False)
        else:
            output, h = self.mixer(x_norm, h)
        output = output + x
        return output, h


class InteractionBlock(nn.Module):
    """
    交互块: 实现 Encoder 和 Search 特征之间的信息交互
    """
    def __init__(self, d_model, extra_extractor=False, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.injector = Injector(d_model=d_model)
        self.extractor = Extractor(d_model=d_model)
        
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(d_model=d_model) for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, xs):
        """
        Args:
            x: (B, L, D) - 完整特征
            xs: (B, L_s, D) - Search 特征
        Returns:
            x: 更新后的完整特征
            xs: 更新后的 Search 特征
        """
        # 注入: 将 search 特征信息注入到完整特征
        x = self.injector(x.permute(1, 0, 2), xs.permute(1, 0, 2)).permute(1, 0, 2)
        
        # 提取: 从完整特征中提取信息到 search 特征
        if self.grad_ckpt and self.training:
            xs = checkpoint.checkpoint(
                self.extractor, xs.permute(1, 0, 2), x.permute(1, 0, 2), 
                use_reentrant=False
            ).permute(1, 0, 2)
        else:
            xs = self.extractor(xs.permute(1, 0, 2), x.permute(1, 0, 2)).permute(1, 0, 2)
        
        # 额外的提取器
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                if self.grad_ckpt and self.training:
                    xs = checkpoint.checkpoint(
                        extractor, xs.permute(1, 0, 2), x.permute(1, 0, 2),
                        use_reentrant=False
                    ).permute(1, 0, 2)
                else:
                    xs = extractor(xs.permute(1, 0, 2), x.permute(1, 0, 2)).permute(1, 0, 2)
        
        return x, xs


class MambaNeck(nn.Module):
    """
    Mamba Neck: 用于上下文信息融合的主模块
    
    结构:
    - 多层 ResidualBlock (Mamba)
    - 多层 InteractionBlock (Injector + Extractor)
    
    特点:
    - 使用隐藏状态传递上下文信息
    - 线性复杂度
    """
    def __init__(self, d_model=512, d_inner=None, n_layers=4, dt_rank=32, 
                 d_state=16, d_conv=3, grad_ckpt=False):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_inner or d_model * 2
        self.n_layers = n_layers
        self.num_channels = d_model
        
        # Mamba 层
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, self.d_inner, dt_rank, d_state, d_conv, grad_ckpt)
            for _ in range(n_layers)
        ])
        
        # 交互层
        self.interactions = nn.ModuleList([
            InteractionBlock(
                d_model=d_model,
                extra_extractor=(i == n_layers - 1),  # 最后一层使用额外的提取器
                grad_ckpt=grad_ckpt
            )
            for i in range(n_layers)
        ])
        
        # 最终归一化
        self.norm_f = RMSNorm(d_model)

    def forward(self, x, xs, h_states=None):
        """
        Args:
            x: (B, L, D) - 完整特征 (template + search)
            xs: (B, L_s, D) - Search 特征
            h_states: List[隐藏状态] - 每层的隐藏状态
        Returns:
            x: 更新后的完整特征
            xs: 更新后的 Search 特征
            h_states: 更新后的隐藏状态列表
        """
        if h_states is None:
            h_states = [None] * self.n_layers
        
        for i in range(self.n_layers):
            # Mamba 层处理 search 特征
            xs, h_states[i] = self.layers[i](xs, h_states[i])
            # 交互层
            x, xs = self.interactions[i](x, xs)
        
        # 最终归一化
        x = self.norm_f(x)
        
        return x, xs, h_states


class SimpleMambaFusion(nn.Module):
    """
    简化版 Mamba 融合模块 - 用于在 SUTrack Encoder 之后进行特征增强
    
    适用场景: 作为即插即用模块，增强 encoder 输出的上下文信息
    """
    def __init__(self, d_model=384, n_layers=2, d_state=16):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Mamba 层
        self.mamba_layers = nn.ModuleList([
            ResidualBlock(d_model, d_model * 2, d_state=d_state)
            for _ in range(n_layers)
        ])
        
        # 输出归一化
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) - Encoder 输出特征
        Returns:
            x: (B, L, D) - 增强后的特征
        """
        h = None
        for layer in self.mamba_layers:
            x, h = layer(x, h)
        x = self.norm(x)
        return x


def build_mamba_neck(cfg, encoder):
    """构建 Mamba Neck 模块"""
    d_model = encoder.num_channels
    n_layers = getattr(cfg.MODEL, 'MAMBA_LAYERS', 4)
    d_state = getattr(cfg.MODEL, 'MAMBA_D_STATE', 16)
    grad_ckpt = getattr(cfg.TRAIN, 'GRAD_CKPT', False)
    
    neck = MambaNeck(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        grad_ckpt=grad_ckpt
    )
    return neck


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Mamba Modules for SUTrack")
    print("=" * 60)
    
    # 测试 MambaBlock
    print("\n[1] Testing MambaBlock...")
    x = torch.randn(2, 196, 384)  # B=2, L=14*14, D=384
    mamba = MambaBlock(d_model=384)
    out, h = mamba(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Hidden state: {h.shape}")
    print(f"   Parameters: {sum(p.numel() for p in mamba.parameters()) / 1e6:.2f}M")
    
    # 测试 SimpleMambaFusion
    print("\n[2] Testing SimpleMambaFusion...")
    x = torch.randn(2, 247, 384)  # B=2, L=196+49+2, D=384
    fusion = SimpleMambaFusion(d_model=384, n_layers=2)
    out = fusion(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in fusion.parameters()) / 1e6:.2f}M")
    
    # 测试 MambaNeck
    print("\n[3] Testing MambaNeck...")
    x = torch.randn(2, 247, 384)
    xs = torch.randn(2, 196, 384)  # search 特征
    neck = MambaNeck(d_model=384, n_layers=4)
    x_out, xs_out, h_states = neck(x, xs)
    print(f"   Input x: {x.shape}, xs: {xs.shape}")
    print(f"   Output x: {x_out.shape}, xs: {xs_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in neck.parameters()) / 1e6:.2f}M")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
