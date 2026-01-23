"""
SGLA: Similarity-Guided Layer-Adaptive modules for SUTrack
Reference: SGLATrack - Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking (CVPR 2025)
GitHub: https://github.com/GXNU-ZhongLab/SGLATrack

核心改进点:
1. SelectionModule: 用于动态选择层
2. SimilarityLoss: 用于引导层间差异化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectionModule(nn.Module):
    """
    层选择模块
    根据输入特征动态预测各层的开启/关闭概率
    """
    def __init__(self, dim, num_layers, reduction=4):
        super().__init__()
        self.num_layers = num_layers
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, num_layers),
            nn.Sigmoid()
        )
        
        # 初始化权重，使得初始时大部分层是开启的
        nn.init.constant_(self.fc[-2].bias, 1.0)

    def forward(self, x):
        """
        Args:
            x: [B, N, C] 输入特征
        Returns:
            probs: [B, num_layers] 各层的启用概率
        """
        # x shape: [B, N, C]
        x_pooled = self.avg_pool(x.transpose(1, 2)).squeeze(-1) # [B, C]
        probs = self.fc(x_pooled) # [B, num_layers]
        return probs

class SimilarityLoss(nn.Module):
    """
    层间相似度损失
    引导模型减少层间冗余，使得相邻层学习到不同的特征
    """
    def __init__(self, mode='cosine'):
        super().__init__()
        self.mode = mode

    def forward(self, features):
        """
        计算相邻层特征之间的相似度
        Args:
            features: List of [B, N, C] 各层的输出特征
        Returns:
            loss: 标量损失
        """
        if len(features) < 2:
            return torch.tensor(0.0).to(features[0].device)
            
        loss = 0
        for i in range(len(features) - 1):
            f1 = features[i]
            f2 = features[i+1]
            
            if self.mode == 'cosine':
                f1_norm = F.normalize(f1, dim=-1)
                f2_norm = F.normalize(f2, dim=-1)
                sim = (f1_norm * f2_norm).sum(dim=-1).mean()
                loss += sim
            elif self.mode == 'mse':
                loss += F.mse_loss(f1, f2)
                
        return loss / (len(features) - 1)

class LayerAdaptiveWrapper(nn.Module):
    """
    包装Transformer Block以支持层自适应
    """
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, prob=None, **kwargs):
        """
        Args:
            x: 输入特征
            prob: 启用概率 (由SelectionModule提供)
        """
        if prob is None:
            return self.block(x, **kwargs)
            
        # 训练时使用随机采样或加权
        # 推理时根据阈值判断
        if self.training:
            # 随机采样 (Stochastic Depth like)
            mask = (torch.rand(prob.shape, device=prob.device) < prob).float()
            # 如果被mask，则直接跳过 (保持残差连接)
            out = self.block(x, **kwargs)
            return x + mask.view(-1, 1, 1) * (out - x)
        else:
            # 推理时使用阈值 (0.5)
            if prob.mean() > 0.5:
                return self.block(x, **kwargs)
            else:
                return x

def build_sgla_module(dim, num_layers):    
    return SelectionModule(dim, num_layers), SimilarityLoss()
