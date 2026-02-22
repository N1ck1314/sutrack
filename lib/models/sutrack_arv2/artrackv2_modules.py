"""
ARTrackV2核心模块实现
包括：
1. Appearance Prompts - 外观演化token
2. Oriented Masking - 定向注意力掩码
3. Confidence Token - 置信度预测
4. Appearance Reconstruction - MAE式外观重建
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math


class AppearancePrompts(nn.Module):
    """
    外观演化提示模块
    作为可学习的动态模板，参与注意力匹配并重建目标外观
    """
    def __init__(self, dim=768, num_prompts=4, drop_path=0.0):
        super().__init__()
        self.num_prompts = num_prompts
        self.dim = dim
        
        # 可学习的外观token
        self.appearance_tokens = nn.Parameter(torch.zeros(1, num_prompts, dim))
        nn.init.trunc_normal_(self.appearance_tokens, std=0.02)
        
        # 外观更新模块 - 用于演化外观表示
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, search_features, prev_appearance=None):
        """
        Args:
            search_features: [B, N, C] - 搜索区域特征
            prev_appearance: [B, num_prompts, C] - 上一帧的外观token（用于跨帧演化）
        Returns:
            appearance_tokens: [B, num_prompts, C] - 更新后的外观token
        """
        B = search_features.shape[0]
        
        if prev_appearance is None:
            # 第一帧：使用初始化的外观token
            appearance = self.appearance_tokens.expand(B, -1, -1)
        else:
            # 后续帧：基于上一帧演化
            appearance = prev_appearance
        
        # 外观演化更新
        appearance_updated = appearance + self.drop_path(self.update_mlp(appearance))
        
        return appearance_updated


class AppearanceReconstruction(nn.Module):
    """
    外观重建模块 - 借鉴MAE思路
    通过masking和重建防止外观token过拟合
    """
    def __init__(self, dim=768, num_prompts=4, mask_ratio=0.5):
        super().__init__()
        self.num_prompts = num_prompts
        self.mask_ratio = mask_ratio
        
        # 重建解码器
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def forward(self, appearance_tokens, target_features):
        """
        Args:
            appearance_tokens: [B, num_prompts, C] - 外观token
            target_features: [B, N, C] - 目标区域特征（重建目标）
        Returns:
            reconstruction_loss: 重建损失
        """
        B, N, C = target_features.shape
        
        if not self.training:
            return torch.tensor(0.0, device=appearance_tokens.device)
        
        # 随机mask部分appearance tokens
        num_masked = int(self.num_prompts * self.mask_ratio)
        noise = torch.rand(B, self.num_prompts, device=appearance_tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 分离masked和visible tokens
        ids_keep = ids_shuffle[:, :self.num_prompts - num_masked]
        ids_masked = ids_shuffle[:, self.num_prompts - num_masked:]
        
        # 应用mask
        appearance_masked = appearance_tokens.clone()
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        appearance_masked = torch.gather(
            appearance_masked, dim=1, 
            index=ids_keep.unsqueeze(-1).expand(-1, -1, C)
        )
        appearance_masked = torch.cat([appearance_masked, mask_tokens], dim=1)
        appearance_masked = torch.gather(
            appearance_masked, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, C)
        )
        
        # 重建
        reconstructed = self.decoder(appearance_masked)
        
        # 计算重建损失（与目标特征的平均池化对齐）
        target_pooled = target_features.mean(dim=1, keepdim=True).expand(-1, self.num_prompts, -1)
        
        # 只在masked位置计算损失
        mask_indicator = torch.zeros(B, self.num_prompts, device=appearance_tokens.device)
        mask_indicator.scatter_(1, ids_masked, 1.0)
        
        loss = F.mse_loss(reconstructed, target_pooled, reduction='none')
        loss = (loss.mean(dim=-1) * mask_indicator).sum() / (mask_indicator.sum() + 1e-8)
        
        return loss


class ConfidenceToken(nn.Module):
    """
    置信度Token - 预测IoU并抑制低质量外观演化
    """
    def __init__(self, dim=768):
        super().__init__()
        
        # 可学习的置信度token
        self.confidence_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.confidence_token, std=0.02)
        
        # IoU预测头
        self.iou_predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, N, C] - 编码后的特征（包含confidence token）
        Returns:
            confidence_score: [B, 1] - 预测的置信度分数
        """
        B = features.shape[0]
        confidence_score = self.iou_predictor(features[:, 0])  # 假设第一个token是confidence token
        return confidence_score
    
    def compute_iou_loss(self, pred_iou, gt_bbox, pred_bbox):
        """
        计算IoU回归损失
        Args:
            pred_iou: [B, 1] - 预测的IoU
            gt_bbox: [B, 4] - ground truth bbox
            pred_bbox: [B, 4] - 预测的bbox
        Returns:
            iou_loss: IoU回归损失
        """
        # 计算真实IoU
        x1 = torch.max(gt_bbox[:, 0], pred_bbox[:, 0])
        y1 = torch.max(gt_bbox[:, 1], pred_bbox[:, 1])
        x2 = torch.min(gt_bbox[:, 2], pred_bbox[:, 2])
        y2 = torch.min(gt_bbox[:, 3], pred_bbox[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area_gt = (gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1])
        area_pred = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])
        union = area_gt + area_pred - intersection
        
        gt_iou = intersection / (union + 1e-8)
        
        # L1 loss
        iou_loss = F.l1_loss(pred_iou.squeeze(-1), gt_iou)
        return iou_loss


class OrientedMasking:
    """
    定向注意力掩码 - 限制appearance tokens的注意力路径
    防止外观token从trajectory token抄近路
    """
    @staticmethod
    def create_attention_mask(
        batch_size,
        num_confidence_tokens,
        num_trajectory_tokens, 
        num_appearance_tokens,
        num_search_tokens,
        device
    ):
        """
        创建定向注意力掩码
        
        Token顺序假设：[confidence, trajectory, appearance, search]
        
        掩码规则：
        - confidence token: 可以看所有token
        - trajectory token: 可以看所有token
        - appearance token: 只能看search和confidence，不能看trajectory（防信息泄漏）
        - search token: 可以看所有token
        
        Args:
            batch_size: batch大小
            num_confidence_tokens: 置信度token数量
            num_trajectory_tokens: 轨迹token数量
            num_appearance_tokens: 外观token数量
            num_search_tokens: 搜索区域token数量
            device: 设备
            
        Returns:
            attention_mask: [N, N] - 注意力掩码 (0表示允许，-inf表示禁止)
                           PyTorch会自动广播到所有batch和heads
        """
        total_tokens = (num_confidence_tokens + num_trajectory_tokens + 
                       num_appearance_tokens + num_search_tokens)
        
        # 初始化全0掩码（默认允许所有注意力）
        # 注意：PyTorch TransformerEncoderLayer期望2D mask [N, N]，会自动广播
        mask = torch.zeros(total_tokens, total_tokens, device=device)
        
        # 定义token范围
        conf_start = 0
        conf_end = num_confidence_tokens
        traj_start = conf_end
        traj_end = traj_start + num_trajectory_tokens
        app_start = traj_end
        app_end = app_start + num_appearance_tokens
        search_start = app_end
        search_end = search_start + num_search_tokens
        
        # 限制appearance tokens的注意力：不能看trajectory tokens
        mask[app_start:app_end, traj_start:traj_end] = float('-inf')
        
        return mask


class PureEncoderDecoder(nn.Module):
    """
    Pure Encoder架构 - 取消帧内自回归，并行处理所有token
    结合轨迹提示、外观提示和置信度token
    """
    def __init__(
        self,
        dim=768,
        num_trajectory_tokens=4,  # x1, y1, x2, y2
        num_appearance_tokens=4,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path=0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_trajectory_tokens = num_trajectory_tokens
        self.num_appearance_tokens = num_appearance_tokens
        
        # 轨迹token（位置信息）
        self.trajectory_tokens = nn.Parameter(torch.zeros(1, num_trajectory_tokens, dim))
        nn.init.trunc_normal_(self.trajectory_tokens, std=0.02)
        
        # 外观演化模块
        self.appearance_prompts = AppearancePrompts(
            dim=dim,
            num_prompts=num_appearance_tokens,
            drop_path=drop_path
        )
        
        # 置信度模块
        self.confidence_module = ConfidenceToken(dim=dim)
        
        # 外观重建模块
        self.appearance_reconstruction = AppearanceReconstruction(
            dim=dim,
            num_prompts=num_appearance_tokens,
            mask_ratio=0.5
        )
        
        # Transformer encoder层（并行处理）
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        # 坐标回归头（并行输出4个坐标，归一化到[0,1]）
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 4),  # x1, y1, x2, y2
            nn.Sigmoid()  # 归一化到[0,1]范围，与标准decoder一致
        )
        
    def forward(
        self,
        search_features,
        prev_trajectory=None,
        prev_appearance=None,
        target_features=None
    ):
        """
        Args:
            search_features: [B, N, C] - 搜索区域特征
            prev_trajectory: [B, 4, C] - 上一帧的轨迹token
            prev_appearance: [B, num_appearance, C] - 上一帧的外观token
            target_features: [B, M, C] - 目标区域特征（用于外观重建）
            
        Returns:
            bbox: [B, 4] - 预测的边界框
            confidence: [B, 1] - 置信度分数
            aux_dict: 辅助信息字典
        """
        B = search_features.shape[0]
        
        # 1. 准备轨迹token
        if prev_trajectory is None:
            trajectory = self.trajectory_tokens.expand(B, -1, -1)
        else:
            trajectory = prev_trajectory
        
        # 2. 外观演化
        appearance = self.appearance_prompts(search_features, prev_appearance)
        
        # 3. 置信度token
        confidence_token = self.confidence_module.confidence_token.expand(B, -1, -1)
        
        # 4. 拼接所有token：[confidence, trajectory, appearance, search]
        all_tokens = torch.cat([
            confidence_token,  # [B, 1, C]
            trajectory,        # [B, 4, C]
            appearance,        # [B, num_appearance, C]
            search_features    # [B, N, C]
        ], dim=1)
        
        # 5. 创建oriented masking
        attention_mask = OrientedMasking.create_attention_mask(
            batch_size=B,
            num_confidence_tokens=1,
            num_trajectory_tokens=self.num_trajectory_tokens,
            num_appearance_tokens=self.num_appearance_tokens,
            num_search_tokens=search_features.shape[1],
            device=search_features.device
        )
        
        # 6. Encoder并行处理（pure encoder架构）
        encoded = self.encoder_layer(all_tokens, src_mask=attention_mask)
        
        # 7. 提取各部分token
        confidence_encoded = encoded[:, 0:1]
        trajectory_encoded = encoded[:, 1:1+self.num_trajectory_tokens]
        appearance_encoded = encoded[:, 1+self.num_trajectory_tokens:1+self.num_trajectory_tokens+self.num_appearance_tokens]
        
        # 8. 预测bbox（从轨迹token并行生成）
        bbox = self.bbox_head(trajectory_encoded.mean(dim=1))  # [B, 4]
        
        # 9. 预测置信度
        confidence = self.confidence_module(confidence_encoded)
        
        # 10. 外观重建损失（训练时）
        aux_dict = {}
        if self.training and target_features is not None:
            aux_dict['appearance_recon_loss'] = self.appearance_reconstruction(
                appearance_encoded, target_features
            )
        
        # 保存当前token用于下一帧
        aux_dict['trajectory_token'] = trajectory_encoded.detach()
        aux_dict['appearance_token'] = appearance_encoded.detach()
        
        return bbox, confidence, aux_dict
