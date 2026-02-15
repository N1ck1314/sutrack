"""
Feature Enhancement Modules for SUTrack Active
These modules can be integrated to improve model performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionModule(nn.Module):
    """
    Cross-attention module to enhance template-search feature interaction.
    This allows the search region to attend to template features more effectively.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, search_feat, template_feat):
        """
        Args:
            search_feat: (B, N_search, C) search region features
            template_feat: (B, N_template, C) template features
        Returns:
            enhanced_search: (B, N_search, C) enhanced search features
        """
        B, N_s, C = search_feat.shape
        _, N_t, _ = template_feat.shape
        
        # Self-attention for search
        q = self.q_proj(search_feat).view(B, N_s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(template_feat).view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(template_feat).view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N_s, C)
        out = self.out_proj(out)
        
        # Residual connection
        enhanced_search = self.norm(search_feat + self.dropout(out))
        return enhanced_search


class FeatureFusionModule(nn.Module):
    """
    Feature fusion module to combine multi-modal features.
    Supports adaptive weighting of different modalities.
    """
    def __init__(self, dim, num_modalities=2, use_adaptive_weight=True):
        super().__init__()
        self.num_modalities = num_modalities
        self.use_adaptive_weight = use_adaptive_weight
        
        if use_adaptive_weight:
            self.weight_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, num_modalities),
                nn.Softmax(dim=-1)
            )
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * num_modalities, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, *features):
        """
        Args:
            features: list of (B, N, C) feature tensors
        Returns:
            fused_feat: (B, N, C) fused features
        """
        if len(features) == 1:
            return features[0]
        
        # Stack features
        stacked = torch.cat(features, dim=-1)  # (B, N, C*num_modalities)
        
        if self.use_adaptive_weight:
            # Compute adaptive weights
            # Use mean pooling to get global representation
            global_feat = stacked.mean(dim=1)  # (B, C*num_modalities)
            weights = self.weight_net(global_feat)  # (B, num_modalities)
            
            # Apply weights to individual features
            weighted_features = []
            for i, feat in enumerate(features):
                weighted_features.append(feat * weights[:, i:i+1].unsqueeze(1))
            stacked = torch.cat(weighted_features, dim=-1)
        
        fused_feat = self.fusion_proj(stacked)
        return fused_feat


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion module.
    Combines features from different scales for better representation.
    """
    def __init__(self, dims, out_dim):
        """
        Args:
            dims: list of feature dimensions from different scales
            out_dim: output feature dimension
        """
        super().__init__()
        self.num_scales = len(dims)
        self.projs = nn.ModuleList([
            nn.Linear(d, out_dim) for d in dims
        ])
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * self.num_scales, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
        
    def forward(self, *features):
        """
        Args:
            features: list of (B, N_i, C_i) features from different scales
        Returns:
            fused: (B, N, C) fused features
        """
        # Project all features to same dimension
        projected = []
        for i, feat in enumerate(features):
            proj_feat = self.projs[i](feat)
            # Resize to same spatial size (use interpolation if needed)
            if proj_feat.shape[1] != features[0].shape[1]:
                proj_feat = F.interpolate(
                    proj_feat.transpose(1, 2),
                    size=features[0].shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            projected.append(proj_feat)
        
        # Concatenate and fuse
        stacked = torch.cat(projected, dim=-1)
        fused = self.fusion(stacked)
        return fused


class TaskAdaptiveModule(nn.Module):
    """
    Task-adaptive module that adjusts features based on task type.
    """
    def __init__(self, dim, num_tasks=5):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_embeddings = nn.Embedding(num_tasks, dim)
        self.adaptive_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, features, task_index):
        """
        Args:
            features: (B, N, C) input features
            task_index: (B,) task indices
        Returns:
            adapted_features: (B, N, C) task-adapted features
        """
        B, N, C = features.shape
        task_emb = self.task_embeddings(task_index)  # (B, C)
        task_emb = task_emb.unsqueeze(1).expand(B, N, C)  # (B, N, C)
        
        combined = torch.cat([features, task_emb], dim=-1)  # (B, N, 2C)
        adapted = self.adaptive_proj(combined)
        
        # Residual connection
        return features + adapted


class ChannelAttention(nn.Module):
    """
    Channel attention module for feature refinement.
    """
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            x * attention: (B, C, H, W) refined features
        """
        B, C, H, W = x.shape
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(B, C))
        # Combine
        attention = (avg_out + max_out).view(B, C, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial attention module for feature refinement.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            x * attention: (B, C, H, W) refined features
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel and spatial attention.
    """
    def __init__(self, dim, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(dim, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            refined: (B, C, H, W) refined features
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

