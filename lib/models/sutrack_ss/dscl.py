"""
Decoupled Spatio-Temporal Consistency Learning (DSCL) Module
Based on SSTrack: Decoupled Spatio-Temporal Consistency Learning for Self-Supervised Tracking
AAAI 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialConsistencyModule(nn.Module):
    """
    空间一致性模块：全局空间定位
    通过空间注意力机制学习目标在不同视角下的空间一致性
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        
        # 空间位置编码
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) - 输入特征
            mask: (B, N) - 可选的掩码
        Returns:
            (B, N, C) - 空间一致性增强的特征
        """
        B, N, C = x.shape
        
        # 添加空间位置编码
        x = x + self.spatial_pos_embed
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # 空间注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        if mask is not None:
            # 应用掩码
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # 聚合
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class TemporalConsistencyModule(nn.Module):
    """
    时间一致性模块：局部时间关联
    学习时间维度上的目标一致性，模拟目标运动变化
    """
    def __init__(self, dim: int, num_frames: int = 2, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # 时间注意力
        self.temporal_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.temporal_proj = nn.Linear(dim, dim)
        
        # 时间位置编码
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames, dim) * 0.02)
        
        # 运动建模 - 可学习的运动偏移
        self.motion_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2)  # x, y偏移
        )
        
    def forward(self, x: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, N, C) - 时序特征，T为时间帧数
            temporal_mask: (B, T) - 时间掩码
        Returns:
            x_enhanced: (B, T, N, C) - 时间一致性增强的特征
            motion_offset: (B, T, 2) - 预测的运动偏移
        """
        B, T, N, C = x.shape
        
        # 池化空间维度得到时间特征
        x_temporal = x.mean(dim=2)  # (B, T, C)
        
        # 添加时间位置编码
        if T <= self.temporal_pos_embed.shape[1]:
            x_temporal = x_temporal + self.temporal_pos_embed[:, :T]
        
        # 时间注意力
        qkv = self.temporal_qkv(x_temporal).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, T, head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if temporal_mask is not None:
            mask = temporal_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        x_temporal = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x_temporal = self.temporal_proj(x_temporal)
        
        # 预测运动偏移
        motion_offset = self.motion_predictor(x_temporal)  # (B, T, 2)
        
        # 将时间特征广播回空间维度
        x_temporal = x_temporal.unsqueeze(2).expand(-1, -1, N, -1)  # (B, T, N, C)
        
        # 融合原始特征和时间特征
        x_enhanced = x + 0.1 * x_temporal  # 残差连接，小系数保持原始特征
        
        return x_enhanced, motion_offset


class DecoupledSTConsistency(nn.Module):
    """
    解耦时空一致性模块 (DSCL)
    将空间一致性和时间一致性解耦学习
    """
    def __init__(
        self,
        dim: int,
        num_frames: int = 2,
        spatial_heads: int = 8,
        temporal_heads: int = 4,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        
        # 空间一致性分支
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_module = SpatialConsistencyModule(dim, num_heads=spatial_heads)
        
        # 时间一致性分支
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_module = TemporalConsistencyModule(dim, num_frames=num_frames, num_heads=temporal_heads)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(drop_path),
            nn.Linear(dim, dim)
        )
        
        # 自适应权重
        self.spatial_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.temporal_weight = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(
        self,
        x: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, T, N, C) 或 (B, N, C) - 输入特征
            spatial_mask: (B, N) - 空间掩码
            temporal_mask: (B, T) - 时间掩码
        Returns:
            out: (B, T, N, C) 或 (B, N, C) - 增强后的特征
            aux_dict: 辅助信息
        """
        if x.dim() == 3:
            # 单帧输入，只应用空间一致性
            B, N, C = x.shape
            x_norm = self.spatial_norm(x)
            x_spatial = self.spatial_module(x_norm, spatial_mask)
            out = x + x_spatial
            return out, {'spatial_attn': None, 'motion_offset': None}
        
        B, T, N, C = x.shape
        
        # 空间一致性 (在每帧内独立计算)
        x_spatial_list = []
        for t in range(T):
            x_t = x[:, t]  # (B, N, C)
            x_t_norm = self.spatial_norm(x_t)
            x_t_spatial = self.spatial_module(x_t_norm, spatial_mask)
            x_spatial_list.append(x_t_spatial)
        x_spatial = torch.stack(x_spatial_list, dim=1)  # (B, T, N, C)
        
        # 时间一致性
        x_temporal_norm = self.temporal_norm(x.reshape(B * T, N, C)).reshape(B, T, N, C)
        x_temporal, motion_offset = self.temporal_module(x_temporal_norm, temporal_mask)
        
        # 自适应融合
        w_s = torch.sigmoid(self.spatial_weight)
        w_t = torch.sigmoid(self.temporal_weight)
        w_sum = w_s + w_t
        w_s, w_t = w_s / w_sum, w_t / w_sum
        
        # 拼接并融合
        x_fused = torch.cat([x_spatial, x_temporal], dim=-1)  # (B, T, N, C*2)
        x_fused = self.fusion(x_fused)
        
        # 残差连接
        out = x + x_fused
        
        aux_dict = {
            'spatial_weight': w_s.item(),
            'temporal_weight': w_t.item(),
            'motion_offset': motion_offset
        }
        
        return out, aux_dict


class InstanceContrastiveLoss(nn.Module):
    """
    实例对比损失
    从多视角建立实例级对应关系
    """
    def __init__(self, temperature: float = 0.5, use_cosine: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_cosine = use_cosine
        
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            feat1: (B, N, C) - 视角1的特征
            feat2: (B, N, C) - 视角2的特征
            mask: (B, N) - 有效区域的掩码
        Returns:
            loss: 标量
        """
        B, N, C = feat1.shape
        
        # 池化特征
        if mask is not None:
            # 使用掩码进行加权池化
            mask = mask.unsqueeze(-1).float()
            feat1_pooled = (feat1 * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # (B, C)
            feat2_pooled = (feat2 * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # (B, C)
        else:
            feat1_pooled = feat1.mean(dim=1)  # (B, C)
            feat2_pooled = feat2.mean(dim=1)  # (B, C)
        
        # 归一化
        if self.use_cosine:
            feat1_pooled = F.normalize(feat1_pooled, dim=-1)
            feat2_pooled = F.normalize(feat2_pooled, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(feat1_pooled, feat2_pooled.T) / self.temperature  # (B, B)
        
        # 正样本对角线
        labels = torch.arange(B, device=feat1.device)
        
        # InfoNCE损失
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    时间一致性损失
    约束相邻帧之间的特征一致性
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, feat_t: torch.Tensor, feat_t1: torch.Tensor, pred_offset: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_t: (B, N, C) - 时刻t的特征
            feat_t1: (B, N, C) - 时刻t+1的特征
            pred_offset: (B, 2) - 预测的运动偏移
        Returns:
            loss: 标量
        """
        B, N, C = feat_t.shape
        
        # 池化特征
        feat_t_pooled = feat_t.mean(dim=1)  # (B, C)
        feat_t1_pooled = feat_t1.mean(dim=1)  # (B, C)
        
        # 计算特征相似度
        feat_t_norm = F.normalize(feat_t_pooled, dim=-1)
        feat_t1_norm = F.normalize(feat_t1_pooled, dim=-1)
        
        # 余弦相似度
        cosine_sim = (feat_t_norm * feat_t1_norm).sum(dim=-1)  # (B,)
        
        # 希望相似度尽可能高
        loss = 1.0 - cosine_sim.mean()
        
        return loss


class SSTrackLoss(nn.Module):
    """
    SSTrack完整的自监督损失函数
    结合实例对比损失和时间一致性损失
    """
    def __init__(
        self,
        temperature: float = 0.5,
        contrastive_weight: float = 1.0,
        temporal_weight: float = 0.5,
        use_cosine: bool = True
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.temporal_weight = temporal_weight
        
        self.contrastive_loss = InstanceContrastiveLoss(temperature, use_cosine)
        self.temporal_loss = TemporalConsistencyLoss()
        
    def forward(
        self,
        view1_feat: torch.Tensor,
        view2_feat: torch.Tensor,
        temporal_feats: Optional[torch.Tensor] = None,
        view1_mask: Optional[torch.Tensor] = None,
        view2_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            view1_feat: (B, N, C) - 视角1的特征
            view2_feat: (B, N, C) - 视角2的特征
            temporal_feats: (B, T, N, C) - 时序特征
            view1_mask: (B, N) - 视角1的掩码
            view2_mask: (B, N) - 视角2的掩码
        Returns:
            total_loss: 总损失
            loss_dict: 各损失的详细信息
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 实例对比损失
        if self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss(view1_feat, view2_feat, view1_mask)
            total_loss += self.contrastive_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss.item()
        
        # 时间一致性损失
        if self.temporal_weight > 0 and temporal_feats is not None and temporal_feats.shape[1] > 1:
            T = temporal_feats.shape[1]
            temporal_loss_total = 0.0
            for t in range(T - 1):
                temporal_loss = self.temporal_loss(
                    temporal_feats[:, t],
                    temporal_feats[:, t + 1],
                    None
                )
                temporal_loss_total += temporal_loss
            temporal_loss_total = temporal_loss_total / (T - 1)
            total_loss += self.temporal_weight * temporal_loss_total
            loss_dict['temporal_loss'] = temporal_loss_total.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
