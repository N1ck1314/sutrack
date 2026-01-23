"""
SMFA: Self-Modulation Feature Aggregation
Reference: SMFANet - A Lightweight Self-Modulation Feature Aggregation Network (ECCV 2024)
GitHub: https://github.com/Zheng-MJ/SMFANet

核心模块:
1. EASA (Efficient Approximation of Self-Attention) - 高效自注意力近似
2. LDE (Local Detail Estimation) - 局部细节估计  
3. SMFA - 自调制特征聚合 (组合EASA和LDE)
4. PCFN (Partial Convolution-based Feed-Forward Network) - 部分卷积前馈网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EASA(nn.Module):
    """
    Efficient Approximation of Self-Attention (EASA)
    
    通过线性近似自注意力机制，降低计算复杂度
    使用 Q @ K^T ≈ (Q @ D) @ (K @ D)^T 的近似策略
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 降维矩阵用于近似 (可选，用于进一步降低复杂度)
        self.use_approximation = True
        if self.use_approximation:
            # 降维到 sqrt(head_dim)
            self.approx_dim = max(int(math.sqrt(self.head_dim)), 4)
            self.proj_q = nn.Linear(self.head_dim, self.approx_dim, bias=False)
            self.proj_k = nn.Linear(self.head_dim, self.approx_dim, bias=False)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        if self.use_approximation:
            # 线性近似: (B, H, N, D) @ (B, H, D, d) = (B, H, N, d)
            q_approx = self.proj_q(q)  # [B, num_heads, N, approx_dim]
            k_approx = self.proj_k(k)  # [B, num_heads, N, approx_dim]
            
            # 近似注意力: Q' @ K'^T
            attn = (q_approx * self.scale) @ k_approx.transpose(-2, -1)  # [B, num_heads, N, N]
        else:
            # 标准注意力
            attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Attention @ V
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class LDE(nn.Module):
    """
    Local Detail Estimation (LDE)
    
    使用深度卷积捕获局部细节信息
    结合多尺度卷积核提取不同感受野的特征
    """
    def __init__(self, dim, kernel_sizes=[3, 5], dilation_rates=[1, 2]):
        super().__init__()
        self.dim = dim
        
        # 多尺度深度卷积分支
        self.branches = nn.ModuleList()
        for i, (k, d) in enumerate(zip(kernel_sizes, dilation_rates)):
            padding = (k + (k - 1) * (d - 1)) // 2
            branch = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=k, padding=padding, 
                         dilation=d, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
            self.branches.append(branch)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * len(kernel_sizes), dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Gate机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        # 多尺度特征提取
        branch_outs = []
        for branch in self.branches:
            branch_outs.append(branch(x))
        
        # 拼接所有分支
        multi_scale = torch.cat(branch_outs, dim=1)  # [B, C*num_branches, H, W]
        
        # 融合
        fused = self.fusion(multi_scale)  # [B, C, H, W]
        
        # 门控调制
        gate_weight = self.gate(fused)  # [B, C, 1, 1]
        out = fused * gate_weight
        
        return out


class SMFA(nn.Module):
    """
    Self-Modulation Feature Aggregation (SMFA)
    
    组合EASA(非局部)和LDE(局部)实现自调制特征聚合
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 lde_kernel_sizes=[3, 5], lde_dilation_rates=[1, 2]):
        super().__init__()
        self.dim = dim
        
        # 非局部分支: EASA
        self.easa = EASA(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        
        # 局部分支: LDE  
        self.lde = LDE(dim, lde_kernel_sizes, lde_dilation_rates)
        
        # 自调制门控
        self.modulation_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 空间特征图
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 转换为序列格式用于EASA
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 非局部分支 (EASA)
        global_feat = self.easa(x_seq)  # [B, H*W, C]
        global_feat = global_feat.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        # 局部分支 (LDE)
        local_feat = self.lde(x)  # [B, C, H, W]
        
        # 自调制: 根据全局和局部特征动态调整权重
        combined = torch.cat([global_feat, local_feat], dim=1)  # [B, 2C, H, W]
        modulation = self.modulation_gate(combined)  # [B, C, 1, 1]
        
        # 加权融合
        aggregated = modulation * global_feat + (1 - modulation) * local_feat
        
        # 输出投影
        out = self.proj(aggregated)
        
        # 残差连接
        out = out + x
        
        return out


class PCFN(nn.Module):
    """
    Partial Convolution-based Feed-Forward Network (PCFN)
    
    使用部分卷积(Partial Conv)的前馈网络
    减少参数量和计算量，同时保持性能
    """
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        # 部分卷积: 只处理部分通道
        self.partial_ratio = 0.5  # 处理50%的通道
        self.partial_dim = int(dim * self.partial_ratio)
        self.remain_dim = dim - self.partial_dim
        
        # Partial branch: 用卷积处理
        self.fc1_partial = nn.Conv2d(self.partial_dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2_partial = nn.Conv2d(hidden_dim, self.partial_dim, 1)
        
        # Remain branch: 直接传递
        # 不需要额外操作，直接identity
        
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        # 分割通道
        x_partial, x_remain = torch.split(x, [self.partial_dim, self.remain_dim], dim=1)
        
        # Partial branch
        x_partial = self.fc1_partial(x_partial)
        x_partial = self.act(x_partial)
        x_partial = self.drop(x_partial)
        x_partial = self.fc2_partial(x_partial)
        x_partial = self.drop(x_partial)
        
        # 拼接
        out = torch.cat([x_partial, x_remain], dim=1)
        
        # 残差连接
        out = out + x
        
        return out


class SMFABlock(nn.Module):
    """
    SMFA Block: 组合SMFA和PCFN的完整模块
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=False,
                 attn_drop=0., proj_drop=0., drop_path=0.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.smfa = SMFA(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        
        self.norm2 = nn.LayerNorm(dim)
        self.pcfn = PCFN(dim, mlp_ratio, proj_drop)
        
        # Drop path (stochastic depth)
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 或 (B, N, C)
        Returns:
            out: same shape as input
        """
        # 检测输入格式
        if x.dim() == 3:
            # (B, N, C) 序列格式
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(1, 2).reshape(B, C, H, W)
            is_sequence = True
        else:
            is_sequence = False
        
        # SMFA分支
        B, C, H, W = x.shape
        x_norm1 = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B,C,H,W)
        x = x + self.drop_path(self.smfa(x_norm1))
        
        # PCFN分支
        x_norm2 = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.drop_path(self.pcfn(x_norm2))
        
        # 转换回序列格式(如果需要)
        if is_sequence:
            x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        
        return x


if __name__ == '__main__':
    # 测试EASA
    print("Testing EASA...")
    easa = EASA(dim=384, num_heads=6)
    x_seq = torch.randn(2, 196, 384)  # (B, N, C)
    out_easa = easa(x_seq)
    print(f'EASA Input: {x_seq.shape}, Output: {out_easa.shape}')
    
    # 测试LDE
    print("\nTesting LDE...")
    lde = LDE(dim=384)
    x_spatial = torch.randn(2, 384, 14, 14)  # (B, C, H, W)
    out_lde = lde(x_spatial)
    print(f'LDE Input: {x_spatial.shape}, Output: {out_lde.shape}')
    
    # 测试SMFA
    print("\nTesting SMFA...")
    smfa = SMFA(dim=384, num_heads=6)
    out_smfa = smfa(x_spatial)
    print(f'SMFA Input: {x_spatial.shape}, Output: {out_smfa.shape}')
    
    # 测试PCFN
    print("\nTesting PCFN...")
    pcfn = PCFN(dim=384, mlp_ratio=4.)
    out_pcfn = pcfn(x_spatial)
    print(f'PCFN Input: {x_spatial.shape}, Output: {out_pcfn.shape}')
    
    # 测试完整SMFABlock
    print("\nTesting SMFABlock...")
    smfa_block = SMFABlock(dim=384, num_heads=6, mlp_ratio=4.)
    out_block = smfa_block(x_spatial)
    print(f'SMFABlock Input: {x_spatial.shape}, Output: {out_block.shape}')
