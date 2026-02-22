"""
SGLA: Similarity-Guided Layer-Adaptive modules for SUTrack
Reference: SGLATrack - Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking (CVPR 2025)
GitHub: https://github.com/GXNU-ZhongLab/SGLATrack

优化改进点:
1. SelectionModule: 增强的层选择模块 (添加Dropout、可配置初始化)
2. SimilarityLoss: 改进的相似度损失 (支持更多模式、数值稳定性)
3. LayerAdaptiveWrapper: 优化的自适应包装器 (可配置阈值、Gumbel-Softmax采样)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class SelectionModule(nn.Module):
    """
    增强的层选择模块
    根据输入特征动态预测各层的开启/关闭概率
    
    优化点:
    - 添加 Dropout 防止过拟合
    - 支持自定义初始化偏置
    - 添加温度参数控制概率分布锐度
    """
    def __init__(self, dim, num_layers, reduction=4, dropout=0.1, init_bias=1.0, temperature=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.temperature = temperature
        
        # 使用更高效的全局池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP with Dropout
        hidden_dim = max(dim // reduction, 32)  # 确保隐藏层维度不会太小
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_layers)
        )
        
        # 可配置的初始化策略
        self._init_weights(init_bias)

    def _init_weights(self, init_bias):
        """初始化权重，使得初始时大部分层是开启的"""
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 最后一层的偏置初始化为正值，使初始概率偏高
        nn.init.constant_(self.fc[-1].bias, init_bias)

    def forward(self, x):
        """
        Args:
            x: [B, N, C] 输入特征
        Returns:
            probs: [B, num_layers] 各层的启用概率
        """
        B, N, C = x.shape
        
        # 全局池化: [B, N, C] -> [B, C, N] -> [B, C, 1] -> [B, C]
        x_pooled = self.avg_pool(x.transpose(1, 2)).squeeze(-1)
        
        # MLP预测: [B, C] -> [B, num_layers]
        logits = self.fc(x_pooled)
        
        # 应用温度缩放和Sigmoid
        probs = torch.sigmoid(logits / self.temperature)
        
        return probs

class SimilarityLoss(nn.Module):
    """
    改进的层间相似度损失
    引导模型减少层间冗余，使得相邻层学习到不同的特征
    
    优化点:
    - 支持多种相似度计算模式
    - 添加数值稳定性保护
    - 支持非相邻层的相似度计算
    - 添加对比损失模式
    """
    def __init__(self, mode='cosine', eps=1e-8, adjacent_only=True):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.adjacent_only = adjacent_only
        
        assert mode in ['cosine', 'mse', 'kl', 'contrastive'], 
            f"mode must be one of ['cosine', 'mse', 'kl', 'contrastive'], got {mode}"

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        计算层间特征相似度
        Args:
            features: List of [B, N, C] 各层的输出特征
        Returns:
            loss: 标量损失
        """
        if len(features) < 2:
            return torch.tensor(0.0, device=features[0].device, dtype=features[0].dtype)
        
        # 只计算相邻层
        if self.adjacent_only:
            pairs = [(i, i+1) for i in range(len(features) - 1)]
        else:
            # 计算所有层对
            pairs = [(i, j) for i in range(len(features)) for j in range(i+1, len(features))]
        
        loss = 0.0
        for i, j in pairs:
            f1, f2 = features[i], features[j]
            
            if self.mode == 'cosine':
                # 余弦相似度 (数值稳定版本)
                f1_norm = F.normalize(f1, p=2, dim=-1, eps=self.eps)
                f2_norm = F.normalize(f2, p=2, dim=-1, eps=self.eps)
                sim = (f1_norm * f2_norm).sum(dim=-1).mean()
                loss += sim
                
            elif self.mode == 'mse':
                # 均方误差
                loss += F.mse_loss(f1, f2, reduction='mean')
                
            elif self.mode == 'kl':
                # KL散度 (需要先转为概率分布)
                f1_prob = F.softmax(f1, dim=-1)
                f2_prob = F.softmax(f2, dim=-1)
                loss += F.kl_div(f1_prob.log(), f2_prob, reduction='batchmean')
                
            elif self.mode == 'contrastive':
                # 对比损失: 最大化层间差异
                f1_flat = f1.flatten(1)  # [B, N*C]
                f2_flat = f2.flatten(1)
                # 计算余弦相似度然后取负(鼓励不相似)
                sim = F.cosine_similarity(f1_flat, f2_flat, dim=1, eps=self.eps)
                loss += sim.mean()
        
        # 归一化
        return loss / len(pairs)

class LayerAdaptiveWrapper(nn.Module):
    """
    优化的Transformer Block层自适应包装器
    
    优化点:
    - 支持Gumbel-Softmax采样 (可微分的离散采样)
    - 可配置的推理阈值
    - 支持软跳过 (加权组合) 和硬跳过
    - 添加统计信息收集
    """
    def __init__(self, block, threshold=0.5, use_gumbel=False, gumbel_tau=1.0, soft_skip=True):
        super().__init__()
        self.block = block
        self.threshold = threshold
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau
        self.soft_skip = soft_skip
        
        # 统计信息 (用于分析)
        self.register_buffer('num_skips', torch.zeros(1))
        self.register_buffer('num_forwards', torch.zeros(1))

    def forward(self, x, prob: Optional[torch.Tensor] = None, **kwargs):
        """
        Args:
            x: 输入特征 [B, N, C]
            prob: 启用概率 [B] (由SelectionModule提供)
        Returns:
            输出特征 [B, N, C]
        """
        if prob is None:
            return self.block(x, **kwargs)
        
        # 统计前向传播次数
        self.num_forwards += 1
        
        if self.training:
            # 训练模式: 随机采样或Gumbel-Softmax
            if self.use_gumbel:
                # Gumbel-Softmax: 可微分的离散采样
                # 构造logits: [prob, 1-prob]
                logits = torch.stack([prob, 1 - prob], dim=-1)  # [B, 2]
                gumbel = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False)
                mask = gumbel[:, 0]  # 取第一个维度 (执行概率)
            else:
                # 伯努利采样
                mask = (torch.rand(prob.shape, device=prob.device) < prob).float()
            
            # 执行block
            out = self.block(x, **kwargs)
            
            if self.soft_skip:
                # 软跳过: 加权组合
                return x + mask.view(-1, 1, 1) * (out - x)
            else:
                # 硬跳过: 要么全执行要么全跳过
                mask_expanded = mask.view(-1, 1, 1).expand_as(x)
                return torch.where(mask_expanded > 0.5, out, x)
        else:
            # 推理模式: 基于阈值的确定性决策
            # 使用batch平均概率或每个样本独立决策
            if prob.dim() == 0:  # 标量
                execute = prob.item() > self.threshold
            else:
                execute = prob.mean().item() > self.threshold
            
            if execute:
                return self.block(x, **kwargs)
            else:
                self.num_skips += 1
                return x
    
    def get_skip_rate(self):
        """获取层跳过率 (仅推理时有效)"""
        if self.num_forwards == 0:
            return 0.0
        return (self.num_skips / self.num_forwards).item()
    
    def reset_stats(self):
        """重置统计信息"""
        self.num_skips.zero_()
        self.num_forwards.zero_()

def build_sgla_module(dim, num_layers, reduction=4, dropout=0.1, 
                      loss_mode='cosine', threshold=0.5, use_gumbel=False):
    """
    构建SGLA模块的工厂函数
    
    Args:
        dim: 特征维度
        num_layers: Transformer层数
        reduction: SelectionModule中的降维比例
        dropout: Dropout比例
        loss_mode: 相似度损失模式 ['cosine', 'mse', 'kl', 'contrastive']
        threshold: LayerAdaptiveWrapper的推理阈值
        use_gumbel: 是否使用Gumbel-Softmax采样
    
    Returns:
        selection_module: 层选择模块
        similarity_loss: 相似度损失函数
    """
    selection_module = SelectionModule(
        dim=dim, 
        num_layers=num_layers, 
        reduction=reduction,
        dropout=dropout
    )
    
    similarity_loss = SimilarityLoss(mode=loss_mode)
    
    return selection_module, similarity_loss
