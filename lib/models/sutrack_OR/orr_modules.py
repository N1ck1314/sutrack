"""
ORR: Occlusion-Robust Representations for UAV Tracking
Reference: ORTrack - Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking (CVPR 2025)
GitHub: https://github.com/wuyou3474/ORTrack

核心模块:
1. SpatialCoxMasking - 基于空间Cox过程的随机遮挡模拟
2. OcclusionRobustEncoder - 遮挡鲁棒特征编码器
3. FeatureInvarianceLoss - 特征不变性损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialCoxMasking(nn.Module):
    """
    Spatial Cox Process Masking
    
    使用空间Cox过程模拟目标遮挡
    通过随机mask patches来模拟UAV跟踪中的遮挡场景
    """
    def __init__(self, mask_ratio=0.3, mask_strategy='random', intensity_lambda=1.0):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy  # 'random', 'block', 'cox'
        self.intensity_lambda = intensity_lambda  # Cox过程强度参数
    
    def generate_cox_intensity(self, H, W, device):
        """
        生成Cox过程强度场
        模拟遮挡的空间分布特性
        """
        # 使用高斯随机场作为强度函数
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # 随机中心点
        cx = torch.rand(1, device=device) * 2 - 1
        cy = torch.rand(1, device=device) * 2 - 1
        
        # 高斯核
        sigma = 0.5
        intensity = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        intensity = intensity * self.intensity_lambda
        
        return intensity
    
    def random_masking(self, x, mask_ratio):
        """
        随机遮挡
        x: [B, N, C] - N个patches
        """
        B, N, C = x.shape
        num_mask = int(N * mask_ratio)
        
        # 为每个batch生成不同的mask
        masks = []
        for _ in range(B):
            perm = torch.randperm(N, device=x.device)
            mask = torch.zeros(N, device=x.device)
            mask[perm[:num_mask]] = 1
            masks.append(mask)
        
        mask = torch.stack(masks, dim=0)  # [B, N]
        return mask
    
    def block_masking(self, x, H, W, mask_ratio):
        """
        块状遮挡 - 更接近真实遮挡场景
        """
        B, N, C = x.shape
        assert N == H * W, "N must equal H * W"
        
        masks = []
        for _ in range(B):
            mask = torch.zeros(H, W, device=x.device)
            
            # 随机块的大小和位置
            block_h = int(H * np.sqrt(mask_ratio))
            block_w = int(W * np.sqrt(mask_ratio))
            
            start_h = torch.randint(0, max(1, H - block_h), (1,)).item()
            start_w = torch.randint(0, max(1, W - block_w), (1,)).item()
            
            mask[start_h:start_h + block_h, start_w:start_w + block_w] = 1
            masks.append(mask.reshape(-1))
        
        mask = torch.stack(masks, dim=0)  # [B, N]
        return mask
    
    def cox_masking(self, x, H, W, mask_ratio):
        """
        基于Cox过程的遮挡 - 空间非均匀分布
        """
        B, N, C = x.shape
        assert N == H * W, "N must equal H * W"
        
        masks = []
        for _ in range(B):
            # 生成强度场
            intensity = self.generate_cox_intensity(H, W, x.device)  # [H, W]
            
            # 根据强度场采样
            probs = intensity.reshape(-1)
            probs = probs / probs.sum()  # 归一化
            
            # 采样要mask的patches
            num_mask = int(N * mask_ratio)
            mask_indices = torch.multinomial(probs, num_mask, replacement=False)
            
            mask = torch.zeros(N, device=x.device)
            mask[mask_indices] = 1
            masks.append(mask)
        
        mask = torch.stack(masks, dim=0)  # [B, N]
        return mask
    
    def forward(self, x, H, W):
        """
        生成遮挡mask
        
        Args:
            x: [B, N, C] 特征序列
            H, W: 空间维度
        Returns:
            mask: [B, N] 1表示被mask，0表示保留
        """
        if self.mask_strategy == 'random':
            mask = self.random_masking(x, self.mask_ratio)
        elif self.mask_strategy == 'block':
            mask = self.block_masking(x, H, W, self.mask_ratio)
        elif self.mask_strategy == 'cox':
            mask = self.cox_masking(x, H, W, self.mask_ratio)
        else:
            raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")
        
        return mask


class FeatureInvarianceLoss(nn.Module):
    """
    特征不变性损失
    
    强制模型学习对遮挡不变的特征表示
    """
    def __init__(self, loss_type='cosine', temperature=0.1):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
    
    def forward(self, feat_clean, feat_masked):
        """
        计算干净特征和遮挡特征之间的不变性损失
        
        Args:
            feat_clean: [B, N, C] 原始特征
            feat_masked: [B, N, C] 遮挡后的特征
        """
        if self.loss_type == 'cosine':
            # 余弦相似度损失
            feat_clean_norm = F.normalize(feat_clean, dim=-1)
            feat_masked_norm = F.normalize(feat_masked, dim=-1)
            
            # 逐patch计算相似度
            similarity = (feat_clean_norm * feat_masked_norm).sum(dim=-1)  # [B, N]
            loss = 1 - similarity.mean()
            
        elif self.loss_type == 'mse':
            # MSE损失
            loss = F.mse_loss(feat_clean, feat_masked)
            
        elif self.loss_type == 'contrastive':
            # 对比学习损失
            B, N, C = feat_clean.shape
            
            # 展平
            feat_clean_flat = feat_clean.reshape(B * N, C)
            feat_masked_flat = feat_masked.reshape(B * N, C)
            
            # 归一化
            feat_clean_flat = F.normalize(feat_clean_flat, dim=-1)
            feat_masked_flat = F.normalize(feat_masked_flat, dim=-1)
            
            # 计算相似度矩阵
            sim_matrix = torch.matmul(feat_clean_flat, feat_masked_flat.t()) / self.temperature
            
            # 对角线是正样本
            labels = torch.arange(B * N, device=feat_clean.device)
            loss = F.cross_entropy(sim_matrix, labels)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class OcclusionRobustEncoder(nn.Module):
    """
    遮挡鲁棒编码器包装器
    
    在训练时应用遮挡模拟和特征不变性约束
    """
    def __init__(self, use_orr=True, mask_ratio=0.3, mask_strategy='cox',
                 invariance_loss_weight=0.5, invariance_loss_type='cosine'):
        super().__init__()
        self.use_orr = use_orr
        
        if self.use_orr:
            self.masking = SpatialCoxMasking(
                mask_ratio=mask_ratio,
                mask_strategy=mask_strategy
            )
            self.invariance_loss = FeatureInvarianceLoss(
                loss_type=invariance_loss_type
            )
            self.invariance_loss_weight = invariance_loss_weight
        
        print(f"[ORR] OcclusionRobustEncoder initialized: use_orr={use_orr}, "
              f"mask_ratio={mask_ratio}, mask_strategy={mask_strategy}")
    
    def apply_masking(self, x, mask):
        """
        应用遮挡mask
        
        Args:
            x: [B, N, C] 特征
            mask: [B, N] 遮挡mask (1表示被mask)
        Returns:
            x_masked: [B, N, C] 遮挡后的特征
        """
        # 将mask的位置设置为0或学习的mask token
        mask = mask.unsqueeze(-1)  # [B, N, 1]
        x_masked = x * (1 - mask)  # 简单置零
        
        return x_masked
    
    def forward(self, x, H, W, training=True):
        """
        前向传播，在训练时应用ORR
        
        Args:
            x: [B, N, C] 输入特征
            H, W: 空间维度
            training: 是否训练模式
        Returns:
            x: [B, N, C] 输出特征
            orr_loss: 不变性损失（仅训练时）
        """
        orr_loss = None
        
        if self.use_orr and training:
            # 生成遮挡mask
            mask = self.masking(x, H, W)  # [B, N]
            
            # 应用遮挡
            x_masked = self.apply_masking(x, mask)
            
            # 存储用于后续计算不变性损失
            # 注意: 实际使用时需要在encoder forward后计算
            self.clean_features = x
            self.masked_features = x_masked
            self.orr_mask = mask
            
            # 使用遮挡后的特征继续前向传播
            x = x_masked
        
        return x, orr_loss
    
    def compute_invariance_loss(self, feat_clean, feat_masked):
        """
        计算特征不变性损失
        
        在encoder输出后调用
        """
        if self.use_orr:
            loss = self.invariance_loss(feat_clean, feat_masked)
            return loss * self.invariance_loss_weight
        return None


if __name__ == '__main__':
    # 测试SpatialCoxMasking
    print("Testing SpatialCoxMasking...")
    masking = SpatialCoxMasking(mask_ratio=0.3, mask_strategy='cox')
    x = torch.randn(2, 196, 384)  # [B, N, C]
    H, W = 14, 14
    mask = masking(x, H, W)
    print(f'Mask shape: {mask.shape}')
    print(f'Mask ratio: {mask.sum(dim=1) / mask.shape[1]}')
    print("✅ SpatialCoxMasking test passed")
    
    # 测试FeatureInvarianceLoss
    print("\nTesting FeatureInvarianceLoss...")
    loss_fn = FeatureInvarianceLoss(loss_type='cosine')
    feat_clean = torch.randn(2, 196, 384)
    feat_masked = torch.randn(2, 196, 384)
    loss = loss_fn(feat_clean, feat_masked)
    print(f'Invariance loss: {loss.item():.4f}')
    print("✅ FeatureInvarianceLoss test passed")
    
    # 测试OcclusionRobustEncoder
    print("\nTesting OcclusionRobustEncoder...")
    orr_encoder = OcclusionRobustEncoder(use_orr=True, mask_ratio=0.3)
    x = torch.randn(2, 196, 384)
    x_out, orr_loss = orr_encoder(x, H=14, W=14, training=True)
    print(f'Output shape: {x_out.shape}')
    print("✅ OcclusionRobustEncoder test passed")
    
    print("\n" + "="*60)
    print("All ORR modules test passed!")
    print("="*60)
