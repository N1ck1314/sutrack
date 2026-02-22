"""
Selective Depth Integration Modules for SUTrack
选择性深度集成 - 借鉴SGLA的层跳过机制，实现深度特征的智能选择使用

核心思想:
1. 深度需求预测 - 预测每一层是否需要深度信息
2. 选择性融合 - 只在需要时使用深度特征
3. 计算效率优化 - 减少不必要的深度处理

改进点:
- 基于SGLA的层自适应思想
- 添加深度特征的重要性评估
- 支持训练时软跳过和推理时硬跳过
- 添加统计信息收集用于分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DepthNeedPredictor(nn.Module):
    """
    深度需求预测器
    根据RGB特征预测每一层是否需要深度信息
    
    优化点:
    - 使用全局和局部特征结合
    - 添加温度参数控制决策锐度
    - 支持多层独立预测
    """
    def __init__(self, dim, num_layers, reduction=4, dropout=0.1, init_bias=1.0, temperature=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.temperature = temperature
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 局部特征提取（用于捕获细节）
        self.local_attn = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, 1),
            nn.Sigmoid()
        )
        
        # 预测网络
        hidden_dim = max(dim // reduction, 64)
        self.predictor = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),  # *2 因为结合全局和局部特征
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_layers)
        )
        
        self._init_weights(init_bias)
    
    def _init_weights(self, init_bias):
        """初始化权重，使初始时倾向于使用深度"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 最后一层偏置设为正值
        nn.init.constant_(self.predictor[-1].bias, init_bias)
    
    def forward(self, rgb_feat):
        """
        Args:
            rgb_feat: [B, N, C] RGB特征
        Returns:
            probs: [B, num_layers] 各层对深度的需求概率
        """
        B, N, C = rgb_feat.shape
        
        # 全局特征: [B, N, C] -> [B, C]
        global_feat = self.global_pool(rgb_feat.transpose(1, 2)).squeeze(-1)
        
        # 局部特征加权: [B, N, C] -> [B, N, 1] -> [B, C]
        local_weights = self.local_attn(rgb_feat)  # [B, N, 1]
        local_feat = (rgb_feat * local_weights).sum(dim=1)  # [B, C]
        
        # 组合特征
        combined_feat = torch.cat([global_feat, local_feat], dim=-1)  # [B, 2C]
        
        # 预测各层需求
        logits = self.predictor(combined_feat)  # [B, num_layers]
        probs = torch.sigmoid(logits / self.temperature)
        
        return probs


class DepthEnhancer(nn.Module):
    """
    深度特征增强模块
    对深度特征进行处理和增强
    """
    def __init__(self, dim, enhance_type='mlp'):
        super().__init__()
        self.enhance_type = enhance_type
        
        if enhance_type == 'mlp':
            self.enhancer = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
        elif enhance_type == 'residual':
            self.enhancer = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
        else:
            raise ValueError(f"Unknown enhance_type: {enhance_type}")
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, depth_feat):
        """
        Args:
            depth_feat: [B, N, C]
        Returns:
            enhanced_depth: [B, N, C]
        """
        if self.enhance_type == 'residual':
            return depth_feat + self.enhancer(depth_feat)
        else:
            return self.enhancer(depth_feat)


class SelectiveDepthIntegration(nn.Module):
    """
    选择性深度集成 - 核心模块
    只在需要时使用深度信息，结合SGLA的层跳过思想
    
    优化点:
    - 训练时软跳过（可微分）
    - 推理时硬跳过（提升速度）
    - 支持Gumbel-Softmax采样
    - 统计深度使用率
    """
    def __init__(self, dim, num_layers, reduction=4, dropout=0.1, 
                 threshold=0.5, use_gumbel=False, gumbel_tau=1.0, 
                 soft_skip=True, enhance_type='mlp'):
        super().__init__()
        self.num_layers = num_layers
        self.threshold = threshold
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau
        self.soft_skip = soft_skip
        
        # 深度需求预测器
        self.depth_predictor = DepthNeedPredictor(
            dim=dim,
            num_layers=num_layers,
            reduction=reduction,
            dropout=dropout
        )
        
        # 每一层的深度增强模块
        self.depth_enhancers = nn.ModuleList([
            DepthEnhancer(dim, enhance_type) 
            for _ in range(num_layers)
        ])
        
        # 统计信息
        self.register_buffer('depth_usage_count', torch.zeros(num_layers))
        self.register_buffer('total_forwards', torch.zeros(1))
    
    def forward(self, rgb_feat, depth_feat, layer_idx):
        """
        Args:
            rgb_feat: [B, N, C] RGB特征
            depth_feat: [B, N, C] 或 None 深度特征
            layer_idx: int 当前层索引
        Returns:
            fused_feat: [B, N, C] 融合后的特征
            layer_prob: [B, 1] 该层使用深度的概率（用于损失计算）
        """
        # 如果没有深度特征，直接返回RGB
        if depth_feat is None:
            return rgb_feat, torch.zeros(rgb_feat.size(0), 1, device=rgb_feat.device)
        
        # 预测各层的深度需求
        depth_need_probs = self.depth_predictor(rgb_feat)  # [B, num_layers]
        layer_prob = depth_need_probs[:, layer_idx:layer_idx+1]  # [B, 1]
        
        self.total_forwards += 1
        
        if self.training:
            # 训练模式: 软跳过或Gumbel采样
            if self.use_gumbel:
                # Gumbel-Softmax: 可微分的离散采样
                logits = torch.stack([layer_prob, 1 - layer_prob], dim=-1)  # [B, 1, 2]
                gumbel = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)
                mask = gumbel[..., 0]  # [B, 1]
            else:
                # 伯努利采样
                mask = (torch.rand_like(layer_prob) < layer_prob).float()
            
            # 增强深度特征
            enhanced_depth = self.depth_enhancers[layer_idx](depth_feat)
            
            if self.soft_skip:
                # 软跳过: 加权融合
                fused = rgb_feat + mask.unsqueeze(1) * enhanced_depth
            else:
                # 硬跳过: 全或无
                mask_expanded = mask.unsqueeze(1).expand_as(rgb_feat)
                fused = torch.where(mask_expanded > 0.5, 
                                   rgb_feat + enhanced_depth, 
                                   rgb_feat)
        else:
            # 推理模式: 硬跳过
            # 使用batch平均概率决策
            avg_prob = layer_prob.mean().item()
            
            if avg_prob > self.threshold:
                # 使用深度
                enhanced_depth = self.depth_enhancers[layer_idx](depth_feat)
                fused = rgb_feat + enhanced_depth
                self.depth_usage_count[layer_idx] += 1
            else:
                # 跳过深度处理
                fused = rgb_feat
        
        return fused, layer_prob
    
    def get_depth_usage_stats(self):
        """获取深度使用统计信息"""
        if self.total_forwards == 0:
            return None
        
        usage_rate = (self.depth_usage_count / self.total_forwards).cpu().numpy()
        return {
            'usage_rate_per_layer': usage_rate,
            'avg_usage_rate': usage_rate.mean(),
            'total_forwards': self.total_forwards.item()
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.depth_usage_count.zero_()
        self.total_forwards.zero_()


class DepthSelectionLoss(nn.Module):
    """
    深度选择损失
    鼓励模型合理使用深度信息
    
    包含两部分:
    1. 稀疏性损失 - 鼓励减少深度使用
    2. 一致性损失 - 鼓励相邻层的决策一致性
    """
    def __init__(self, sparsity_weight=0.01, consistency_weight=0.01):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.consistency_weight = consistency_weight
    
    def forward(self, layer_probs: List[torch.Tensor]):
        """
        Args:
            layer_probs: List of [B, 1] 各层使用深度的概率
        Returns:
            loss: 标量损失
        """
        if len(layer_probs) == 0:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 拼接所有层的概率 [B, num_layers]
        all_probs = torch.cat(layer_probs, dim=1)
        
        # 1. 稀疏性损失: 鼓励降低使用率（但不能太低）
        # 使用L1范数鼓励稀疏性
        sparsity_loss = all_probs.mean()
        
        # 2. 一致性损失: 相邻层决策应该相似
        if all_probs.size(1) > 1:
            consistency_loss = 0.0
            for i in range(all_probs.size(1) - 1):
                diff = torch.abs(all_probs[:, i] - all_probs[:, i+1])
                consistency_loss += diff.mean()
            consistency_loss /= (all_probs.size(1) - 1)
        else:
            consistency_loss = 0.0
        
        # 总损失
        total_loss = (self.sparsity_weight * sparsity_loss + 
                     self.consistency_weight * consistency_loss)
        
        return total_loss


def build_selective_depth_module(dim, num_layers, reduction=4, dropout=0.1,
                                 threshold=0.5, use_gumbel=False,
                                 sparsity_weight=0.01, consistency_weight=0.01):
    """
    构建选择性深度集成模块的工厂函数
    
    Args:
        dim: 特征维度
        num_layers: Transformer层数
        reduction: 预测器中的降维比例
        dropout: Dropout比例
        threshold: 推理时的阈值
        use_gumbel: 是否使用Gumbel-Softmax
        sparsity_weight: 稀疏性损失权重
        consistency_weight: 一致性损失权重
    
    Returns:
        selective_module: 选择性深度集成模块
        selection_loss: 深度选择损失函数
    """
    selective_module = SelectiveDepthIntegration(
        dim=dim,
        num_layers=num_layers,
        reduction=reduction,
        dropout=dropout,
        threshold=threshold,
        use_gumbel=use_gumbel
    )
    
    selection_loss = DepthSelectionLoss(
        sparsity_weight=sparsity_weight,
        consistency_weight=consistency_weight
    )
    
    return selective_module, selection_loss
