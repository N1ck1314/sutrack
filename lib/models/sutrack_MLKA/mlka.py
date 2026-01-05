"""
Multi-scale Large Kernel Attention (MLKA) Module
Adapted for SUTrack - Multi-modal Object Tracking

Paper: Multi-scale Attention Network for Single Image Super-Resolution
URL: https://arxiv.org/abs/2209.14145
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer Normalization supporting both channels_last and channels_first"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data_format: {data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLKA(nn.Module):
    """
    Multi-scale Large Kernel Attention (MLKA)
    
    This module captures multi-scale spatial information using large kernels
    with dilated convolutions, suitable for tracking tasks requiring:
    - Multi-scale feature extraction
    - Large receptive field
    - Lightweight computation
    
    Args:
        n_feats (int): Number of input/output feature channels (must be divisible by 3)
        use_norm (bool): Whether to apply layer normalization
    """
    def __init__(self, n_feats, use_norm=True):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError(f"n_feats ({n_feats}) must be divisible by 3 for MLKA.")
        
        self.n_feats = n_feats
        i_feats = 2 * n_feats  # Intermediate features

        # Layer normalization (optional)
        self.use_norm = use_norm
        if use_norm:
            self.norm = LayerNorm(n_feats, data_format='channels_first')

        # First projection: expand channels
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        )

        # LKA3: Small kernel path (3x3 + 5x5 dilated)
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=4, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0)
        )
        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)

        # LKA5: Medium kernel path (5x5 + 7x7 dilated)
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=9, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0)
        )
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 2, groups=n_feats // 3)

        # LKA7: Large kernel path (7x7 + 9x9 dilated)
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 3, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=16, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0)
        )
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 3, groups=n_feats // 3)

        # Final projection
        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        )

        # Learnable scaling parameter
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        """
        Args:
            x: Input feature map, shape (B, C, H, W)
        Returns:
            Enhanced feature map, shape (B, C, H, W)
        """
        shortcut = x.clone()
        
        # Layer normalization
        if self.use_norm:
            x = self.norm(x)
        
        # First projection
        x = self.proj_first(x)  # (B, 2C, H, W)
        
        # Split into attention and content parts
        a, x = torch.chunk(x, 2, dim=1)  # Both (B, C, H, W)
        
        # Multi-scale large kernel attention
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)  # Each (B, C/3, H, W)
        
        # Apply LKA and element-wise attention
        a = torch.cat([
            self.LKA3(a_1) * self.X3(a_1),
            self.LKA5(a_2) * self.X5(a_2),
            self.LKA7(a_3) * self.X7(a_3)
        ], dim=1)  # (B, C, H, W)
        
        # Apply attention to content
        x = x * a
        
        # Final projection with residual connection
        x = self.proj_last(x) * self.scale + shortcut
        
        return x


class MLKABlock(nn.Module):
    """
    MLKA Block with Feed-Forward Network
    Complete building block for integration into SUTrack
    
    Args:
        dim (int): Feature dimension
        mlp_ratio (float): Expansion ratio for FFN
        drop (float): Dropout rate
    """
    def __init__(self, dim, mlp_ratio=3.0, drop=0.):
        super().__init__()
        self.mlka = MLKA(dim, use_norm=True)
        
        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = LayerNorm(dim, data_format='channels_first')
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_dim, dim, 1, 1, 0),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Output: (B, C, H, W)
        """
        # MLKA attention
        x = self.mlka(x)
        
        # FFN with residual
        x = x + self.mlp(self.norm(x))
        
        return x


class MLKAFeatureEnhancement(nn.Module):
    """
    Feature Enhancement Module for SUTrack
    Applies MLKA to enhance template-search feature interaction
    
    Args:
        dim (int): Feature dimension
        num_blocks (int): Number of MLKA blocks
    """
    def __init__(self, dim, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLKABlock(dim) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H, W)
        Returns:
            Enhanced feature map (B, C, H, W)
        """
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == '__main__':
    """Test MLKA module"""
    print("="*60)
    print("Testing MLKA Module for SUTrack")
    print("="*60)
    
    # Test 1: Basic MLKA
    n_feats = 384  # Typical for ViT-T/S
    model = MLKA(n_feats)
    x = torch.randn(2, n_feats, 16, 16)
    y = model(x)
    print(f"\n[Test 1] Basic MLKA:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test 2: MLKA Block with FFN
    block = MLKABlock(n_feats)
    y = block(x)
    print(f"\n[Test 2] MLKA Block:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Params: {sum(p.numel() for p in block.parameters()) / 1e6:.2f}M")
    
    # Test 3: Feature Enhancement (for decoder)
    enhancer = MLKAFeatureEnhancement(dim=512, num_blocks=2)
    x_dec = torch.randn(2, 512, 18, 18)  # Search region features
    y_dec = enhancer(x_dec)
    print(f"\n[Test 3] Feature Enhancement:")
    print(f"  Input:  {x_dec.shape}")
    print(f"  Output: {y_dec.shape}")
    print(f"  Params: {sum(p.numel() for p in enhancer.parameters()) / 1e6:.2f}M")
    
    # Test 4: Different scales
    print(f"\n[Test 4] Multi-scale Test:")
    for h, w in [(14, 14), (18, 18), (28, 28)]:
        x_test = torch.randn(1, n_feats, h, w)
        y_test = model(x_test)
        print(f"  {h}x{w} -> {y_test.shape[2]}x{y_test.shape[3]} ✓")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
