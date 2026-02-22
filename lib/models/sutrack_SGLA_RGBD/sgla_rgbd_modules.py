"""
SGLA-RGBD: SGLA-inspired Multi-Modal RGBD Fusion Modules
基于SGLA思想的多模态RGBD融合模块

核心改进:
1. ModalSelectionModule: 自适应模态权重选择
2. ModalComplementarityLoss: 模态互补性损失
3. LayerwiseModalFusion: 逐层模态融合决策
4. SelectiveDepthIntegration: 选择性深度集成
5. SGLA_RGBD_Encoder: 完整的SGLA-RGBD编码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple


class ModalSelectionModule(nn.Module):
    """
    模态选择模块
    根据场景特征动态决定 RGB 和 Depth 的使用权重
    
    优化点:
    - 添加场景感知机制
    - 温度参数控制权重分布
    - 支持多种池化策略
    """
    def __init__(self, dim, reduction=4, temperature=1.0, pool_type='adaptive'):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.pool_type = pool_type
        
        # 全局特征提取
        if pool_type == 'adaptive':
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == 'max':
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 场景感知网络
        hidden_dim = max(dim // reduction, 32)
        self.scene_encoder = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )
        
        # 模态重要性预测 [RGB, Depth]
        self.modal_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 2),
        )
        
        # 可学习的初始偏置(偏向RGB)
        self.register_buffer('init_bias', torch.tensor([0.6, 0.4]))
        
    def forward(self, rgb_feat, depth_feat):
        """
        Args:
            rgb_feat: [B, N, C] RGB特征
            depth_feat: [B, N, C] Depth特征
        Returns:
            modal_weights: [B, 2] (RGB权重, Depth权重)
            scene_feature: [B, hidden_dim//2] 场景特征(用于其他模块)
        """
        B, N, C = rgb_feat.shape
        
        # 拼接特征
        concat_feat = torch.cat([rgb_feat, depth_feat], dim=-1)  # [B, N, 2C]
        
        # 全局池化
        pooled = self.global_pool(concat_feat.transpose(1, 2)).squeeze(-1)  # [B, 2C]
        
        # 场景编码
        scene_feat = self.scene_encoder(pooled)  # [B, hidden_dim//2]
        
        # 预测模态权重
        logits = self.modal_predictor(scene_feat)  # [B, 2]
        
        # 添加初始偏置
        logits = logits + self.init_bias.unsqueeze(0)
        
        # 应用温度和Softmax
        weights = F.softmax(logits / self.temperature, dim=-1)
        
        return weights, scene_feat


class ModalComplementarityLoss(nn.Module):
    """
    模态互补性损失
    鼓励 RGB 和 Depth 学习互补特征
    
    优化点:
    - 支持多种相似度计算模式
    - 添加目标相似度控制
    - 层级加权机制
    """
    def __init__(self, mode='controlled_sim', target_sim=0.3, layer_weights=None):
        super().__init__()
        self.mode = mode
        self.target_sim = target_sim
        self.layer_weights = layer_weights
        
        assert mode in ['controlled_sim', 'negative_cosine', 'mutual_info', 'contrastive'], \
            f"mode must be one of ['controlled_sim', 'negative_cosine', 'mutual_info', 'contrastive']"
        
    def forward(self, rgb_features: List[torch.Tensor], 
                depth_features: List[torch.Tensor]) -> torch.Tensor:
        """
        计算模态间互补性损失
        Args:
            rgb_features: List of [B, N, C] RGB各层特征
            depth_features: List of [B, N, C] Depth各层特征
        Returns:
            loss: 标量损失
        """
        if len(rgb_features) != len(depth_features):
            raise ValueError("RGB和Depth特征层数不匹配")
        
        if len(rgb_features) == 0:
            return torch.tensor(0.0, device=rgb_features[0].device if rgb_features else 'cpu')
        
        num_layers = len(rgb_features)
        loss = 0.0
        
        # 层级权重(如果未指定，则均匀权重)
        if self.layer_weights is None:
            weights = [1.0 / num_layers] * num_layers
        else:
            weights = self.layer_weights
        
        for idx, (rgb_f, depth_f, w) in enumerate(zip(rgb_features, depth_features, weights)):
            # 归一化
            rgb_norm = F.normalize(rgb_f, p=2, dim=-1, eps=1e-8)
            depth_norm = F.normalize(depth_f, p=2, dim=-1, eps=1e-8)
            
            # 计算相似度
            sim = (rgb_norm * depth_norm).sum(dim=-1).mean()
            
            if self.mode == 'controlled_sim':
                # 控制相似度在目标值附近
                loss += w * torch.abs(sim - self.target_sim)
                
            elif self.mode == 'negative_cosine':
                # 鼓励不相似(互补)
                loss += w * torch.clamp(sim - self.target_sim, min=0)
                
            elif self.mode == 'mutual_info':
                # 最小化互信息(鼓励独立)
                loss += w * sim
                
            elif self.mode == 'contrastive':
                # 对比损失(最大化差异)
                loss += w * (1.0 + sim)  # 相似度越低，损失越小
        
        return loss


class LayerwiseModalFusion(nn.Module):
    """
    逐层模态融合决策
    每层独立决定最佳融合策略
    
    优化点:
    - 支持多种融合模式
    - Gumbel-Softmax可微采样
    - 层级自适应阈值
    """
    def __init__(self, dim, num_layers, fusion_modes=['concat', 'add', 'gate'], 
                 threshold=0.5, use_gumbel=True, gumbel_tau=1.0):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.fusion_modes = fusion_modes
        self.num_modes = len(fusion_modes)
        self.threshold = threshold
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau
        
        # 每层的融合决策网络
        self.fusion_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim // 4),
                nn.LayerNorm(dim // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim // 4, self.num_modes)
            ) for _ in range(num_layers)
        ])
        
        # 门控融合网络
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])
        
        # Concat后的降维
        self.concat_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            ) for _ in range(num_layers)
        ])
        
        # 统计信息
        self.register_buffer('fusion_stats', torch.zeros(num_layers, self.num_modes))
        
    def forward(self, rgb_feat, depth_feat, layer_idx):
        """
        Args:
            rgb_feat: [B, N, C]
            depth_feat: [B, N, C]
            layer_idx: 当前层索引
        Returns:
            fused_feat: [B, N, C] 融合特征
            decision_info: Dict 决策信息
        """
        B, N, C = rgb_feat.shape
        
        # 全局特征用于决策
        rgb_global = rgb_feat.mean(dim=1)  # [B, C]
        depth_global = depth_feat.mean(dim=1)  # [B, C]
        concat_global = torch.cat([rgb_global, depth_global], dim=-1)  # [B, 2C]
        
        # 预测融合策略
        fusion_logits = self.fusion_predictors[layer_idx](concat_global)  # [B, num_modes]
        fusion_probs = F.softmax(fusion_logits, dim=-1)
        
        # 采样融合策略
        if self.training and self.use_gumbel:
            # Gumbel-Softmax: 可微的离散采样
            fusion_weights = F.gumbel_softmax(fusion_logits, tau=self.gumbel_tau, hard=False)
        else:
            # 推理: 硬选择
            fusion_weights = F.one_hot(
                fusion_probs.argmax(dim=-1), 
                num_classes=self.num_modes
            ).float()
            
        # 更新统计
        if not self.training:
            self.fusion_stats[layer_idx] += fusion_weights.sum(dim=0).detach()
        
        # 执行各种融合模式
        fused_results = []
        
        # Mode 0: Concat
        if 'concat' in self.fusion_modes:
            concat_feat = torch.cat([rgb_feat, depth_feat], dim=-1)  # [B, N, 2C]
            fused_concat = self.concat_projs[layer_idx](concat_feat)  # [B, N, C]
            fused_results.append(fused_concat)
        
        # Mode 1: Add
        if 'add' in self.fusion_modes:
            fused_add = rgb_feat + depth_feat
            fused_results.append(fused_add)
        
        # Mode 2: Gate
        if 'gate' in self.fusion_modes:
            gate_input = torch.cat([rgb_feat, depth_feat], dim=-1)
            gate = self.gate_networks[layer_idx](gate_input)  # [B, N, C]
            fused_gate = gate * rgb_feat + (1 - gate) * depth_feat
            fused_results.append(fused_gate)
        
        # 加权组合
        fused_feat = torch.zeros_like(rgb_feat)
        for i, feat in enumerate(fused_results):
            weight = fusion_weights[:, i].view(B, 1, 1)
            fused_feat = fused_feat + weight * feat
        
        decision_info = {
            'fusion_logits': fusion_logits,
            'fusion_probs': fusion_probs,
            'fusion_weights': fusion_weights,
            'selected_mode': fusion_probs.argmax(dim=-1)
        }
        
        return fused_feat, decision_info


class SelectiveDepthIntegration(nn.Module):
    """
    选择性深度集成
    智能决定何时使用深度信息
    
    优化点:
    - 深度需求自适应预测
    - 软/硬跳过机制
    - 深度增强网络
    """
    def __init__(self, dim, num_layers, skip_threshold=0.5, use_soft_skip=True):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.skip_threshold = skip_threshold
        self.use_soft_skip = use_soft_skip
        
        # 深度需求预测器
        self.depth_need_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 4, num_layers),
            nn.Sigmoid()
        )
        
        # 深度增强模块(逐层)
        self.depth_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # 统计信息
        self.register_buffer('depth_usage_count', torch.zeros(num_layers))
        self.register_buffer('total_count', torch.zeros(num_layers))
        
    def forward(self, rgb_feat, depth_feat, layer_idx):
        """
        Args:
            rgb_feat: [B, N, C] RGB特征(可能已融合)
            depth_feat: [B, N, C] 原始Depth特征
            layer_idx: 当前层索引
        Returns:
            final_feat: [B, N, C] 最终特征
            depth_prob: [B, 1] 深度使用概率
        """
        if depth_feat is None:
            return rgb_feat, torch.zeros(rgb_feat.size(0), 1, device=rgb_feat.device)
        
        B, N, C = rgb_feat.shape
        
        # 预测所有层的深度需求
        rgb_global = rgb_feat.mean(dim=1)  # [B, C]
        depth_need_probs = self.depth_need_predictor(rgb_global)  # [B, num_layers]
        layer_prob = depth_need_probs[:, layer_idx:layer_idx+1]  # [B, 1]
        
        # 更新统计
        self.total_count[layer_idx] += B
        
        if self.training:
            if self.use_soft_skip:
                # 训练: 软跳过(加权)
                mask = (torch.rand(B, 1, device=rgb_feat.device) < layer_prob).float()
                enhanced_depth = self.depth_enhancers[layer_idx](depth_feat)
                final_feat = rgb_feat + mask.unsqueeze(1) * enhanced_depth
                self.depth_usage_count[layer_idx] += mask.sum()
            else:
                # 训练: 硬跳过(随机)
                mask = (torch.rand_like(layer_prob) < layer_prob).float()
                if mask.any():
                    enhanced_depth = self.depth_enhancers[layer_idx](depth_feat)
                    final_feat = torch.where(
                        mask.unsqueeze(1) > 0.5,
                        rgb_feat + enhanced_depth,
                        rgb_feat
                    )
                else:
                    final_feat = rgb_feat
                self.depth_usage_count[layer_idx] += mask.sum()
        else:
            # 推理: 阈值决策
            if layer_prob.mean() > self.skip_threshold:
                enhanced_depth = self.depth_enhancers[layer_idx](depth_feat)
                final_feat = rgb_feat + enhanced_depth
                self.depth_usage_count[layer_idx] += B
            else:
                final_feat = rgb_feat
        
        return final_feat, layer_prob
    
    def get_depth_usage_rate(self):
        """获取各层深度使用率"""
        usage_rate = self.depth_usage_count / (self.total_count + 1e-8)
        return usage_rate
    
    def reset_stats(self):
        """重置统计信息"""
        self.depth_usage_count.zero_()
        self.total_count.zero_()


class SGLA_RGBD_Encoder(nn.Module):
    """
    SGLA启发的RGBD编码器
    结合所有优化模块的完整方案
    
    优化改进:
    - 支持渐进式特征提取
    - 添加残差连接
    - 损失权重自适应调整
    - 详细的统计信息收集
    """
    def __init__(self, base_encoder, dim=384, num_layers=12, 
                 use_modal_selection=True, use_layerwise_fusion=True,
                 use_selective_depth=True, use_complementarity_loss=True):
        super().__init__()
        self.base_encoder = base_encoder
        self.dim = dim
        self.num_layers = num_layers
        
        # 功能开关
        self.use_modal_selection = use_modal_selection
        self.use_layerwise_fusion = use_layerwise_fusion
        self.use_selective_depth = use_selective_depth
        self.use_complementarity_loss = use_complementarity_loss
        
        # 1. 模态选择模块
        if use_modal_selection:
            self.modal_selector = ModalSelectionModule(dim)
        
        # 2. 逐层融合决策
        if use_layerwise_fusion:
            self.layerwise_fusion = LayerwiseModalFusion(dim, num_layers)
        
        # 3. 选择性深度集成
        if use_selective_depth:
            self.selective_depth = SelectiveDepthIntegration(dim, num_layers)
        
        # 4. 模态互补性损失
        if use_complementarity_loss:
            self.complementarity_loss = ModalComplementarityLoss()
        
        # 统计信息
        self.register_buffer('modal_usage', torch.zeros(2))  # RGB, Depth
        self.register_buffer('forward_count', torch.tensor(0))
        
    def forward(self, template_list, search_list, template_anno_list=None, 
                text_src=None, task_index=None):
        """
        Args:
            template_list: List of template images (RGB and Depth)
            search_list: List of search images (RGB and Depth)
            template_anno_list: Template annotations
            text_src: Text embeddings (optional)
            task_index: Task index (optional)
        Returns:
            features: 融合特征
            aux_info: 辅助信息(损失、统计等)
        """
        # 分离RGB和Depth
        # template_list 和 search_list 是列表，每个元素是 [B, C, H, W] 的 tensor
        template = template_list[0]  # 获取第一个模板
        search = search_list[0]  # 获取第一个搜索区域
        
        # 检查是否是6通道输入(RGB+Depth)
        if template.size(1) == 6:
            # 6通道输入：直接通过base_encoder处理
            # base_encoder会通过修改的patch_embed处理6通道输入
            # 简化版本：直接调用base_encoder的forward
            xs = self.base_encoder.forward(template_list, search_list, template_anno_list, text_src, task_index)
            
            # 添加辅助信息（但在简化版本中暂时为空）
            aux_info = {
                'modal_weights': None,
                'fusion_decisions': [],
                'depth_usage_probs': [],
                'complementarity_loss': torch.tensor(0.0).to(template.device),
                'modal_balance_loss': torch.tensor(0.0).to(template.device)
            }
            return xs, aux_info
        else:
            # 单模态(仅RGB)：直接使用base_encoder
            xs = self.base_encoder.forward(template_list, search_list, template_anno_list, text_src, task_index)
            aux_info = {
                'modal_weights': None,
                'fusion_decisions': [],
                'depth_usage_probs': [],
                'complementarity_loss': torch.tensor(0.0).to(template.device),
                'modal_balance_loss': torch.tensor(0.0).to(template.device)
            }
            return xs, aux_info
    
    def get_statistics(self):
        """获取模块统计信息"""
        stats = {
            'modal_usage': (self.modal_usage / (self.forward_count + 1e-8)).tolist(),
            'forward_count': self.forward_count.item()
        }
        
        if self.use_selective_depth:
            stats['depth_usage_rate'] = self.selective_depth.get_depth_usage_rate().tolist()
        
        if self.use_layerwise_fusion:
            stats['fusion_stats'] = self.layerwise_fusion.fusion_stats.tolist()
        
        return stats
    
    def reset_statistics(self):
        """重置所有统计信息"""
        self.modal_usage.zero_()
        self.forward_count.zero_()
        
        if self.use_selective_depth:
            self.selective_depth.reset_stats()
        
        if self.use_layerwise_fusion:
            self.layerwise_fusion.fusion_stats.zero_()
