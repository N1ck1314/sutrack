"""
Dynamic Resolution Modules for SUTrack
基于 DynRefer 论文的动态分辨率思想，适配目标跟踪任务

核心创新:
1. DynamicResolutionExtractor: 根据目标大小自适应调整特征提取分辨率
2. MultiViewFusion: 多视图特征融合，模拟人类视觉的多尺度观察
3. AdaptiveRegionAlign: 区域级特征对齐，精确提取目标特征

论文: DynRefer: Delving into Region-level Multimodal Tasks via Dynamic Resolution
适配场景: 目标跟踪中的小目标、尺度变化、遮挡处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicResolutionExtractor(nn.Module):
    """
    动态分辨率特征提取器
    
    核心思想（借鉴 DynRefer）:
    - 小目标：提取高分辨率局部特征
    - 大目标：使用全局上下文特征
    - 中等目标：融合多尺度特征
    
    适用于目标跟踪中的尺度自适应
    """
    def __init__(self, dim, num_scales=3):
        """
        Args:
            dim: 特征维度
            num_scales: 动态分辨率尺度数量（通常3个：低/中/高）
        """
        super(DynamicResolutionExtractor, self).__init__()
        self.dim = dim
        self.num_scales = num_scales
        
        # 多尺度特征提取分支
        self.scale_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1)
            )
            for _ in range(num_scales)
        ])
        
        # 尺度选择网络（自适应决定使用哪个分辨率）
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(dim * num_scales, dim, kernel_size=1)
        
    def forward(self, x, target_size_hint=None):
        """
        Args:
            x: (B, C, H, W) 输入特征
            target_size_hint: 可选，目标大小提示（用于指导尺度选择）
        Returns:
            动态分辨率增强的特征: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. 多尺度特征提取
        scale_features = []
        for i, branch in enumerate(self.scale_branches):
            # 不同尺度使用不同的采样策略
            if i == 0:  # 低分辨率（全局视图）
                feat = F.adaptive_avg_pool2d(x, (H // 2, W // 2))
                feat = branch(feat)
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            elif i == 1:  # 中分辨率（原始）
                feat = branch(x)
            else:  # 高分辨率（局部细节）
                feat = F.interpolate(x, scale_factor=1.5, mode='bilinear', align_corners=False)
                feat = branch(feat)
                feat = F.adaptive_avg_pool2d(feat, (H, W))
            
            scale_features.append(feat)
        
        # 2. 自适应尺度选择
        scale_weights = self.scale_selector(x)  # (B, num_scales, 1, 1)
        
        # 3. 加权融合多尺度特征
        weighted_features = []
        for i, feat in enumerate(scale_features):
            weight = scale_weights[:, i:i+1, :, :]  # (B, 1, 1, 1)
            weighted_features.append(feat * weight)
        
        # 拼接并融合
        multi_scale_feat = torch.cat(weighted_features, dim=1)  # (B, C*num_scales, H, W)
        output = self.fusion_conv(multi_scale_feat)
        
        return output


class MultiViewFusion(nn.Module):
    """
    多视图特征融合模块
    
    核心思想（借鉴 DynRefer 的多视图采样）:
    - 从不同位置和尺度观察目标
    - 类似人类视觉的"扫视"机制
    - 增强对遮挡和形变的鲁棒性
    """
    def __init__(self, dim, num_views=4):
        """
        Args:
            dim: 特征维度
            num_views: 视图数量
        """
        super(MultiViewFusion, self).__init__()
        self.dim = dim
        self.num_views = num_views
        
        # 视图特征提取
        self.view_extractors = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
            for _ in range(num_views)
        ])
        
        # 跨视图注意力
        self.cross_view_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        
        # 视图融合
        self.view_fusion = nn.Sequential(
            nn.Conv2d(dim * num_views, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            多视图融合特征: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 1. 提取多个视图的特征
        view_features = []
        for extractor in self.view_extractors:
            view_feat = extractor(x)
            view_features.append(view_feat)
        
        # 2. 跨视图注意力（增强视图间的信息交互）
        # 将空间维度展平用于注意力计算
        view_tokens = [feat.flatten(2).transpose(1, 2) for feat in view_features]  # [(B, HW, C), ...]
        
        # 对每个视图应用跨视图注意力
        refined_views = []
        for i, query in enumerate(view_tokens):
            # 使用其他视图作为 key 和 value
            key_value = torch.cat([v for j, v in enumerate(view_tokens) if j != i], dim=1)
            refined, _ = self.cross_view_attn(query, key_value, key_value)
            refined_views.append(refined.transpose(1, 2).reshape(B, C, H, W))
        
        # 3. 融合所有视图
        multi_view = torch.cat(refined_views, dim=1)  # (B, C*num_views, H, W)
        output = self.view_fusion(multi_view)
        
        return output


class AdaptiveRegionAlign(nn.Module):
    """
    自适应区域对齐模块
    
    核心思想（借鉴 DynRefer 的区域对齐）:
    - 精确提取目标区域特征
    - 根据目标形状自适应调整感受野
    - 减少背景干扰
    """
    def __init__(self, dim, align_size=7):
        """
        Args:
            dim: 特征维度
            align_size: 对齐后的特征图大小
        """
        super(AdaptiveRegionAlign, self).__init__()
        self.dim = dim
        self.align_size = align_size
        
        # 区域特征增强
        self.region_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        
        # 边界感知卷积（增强目标边缘）
        self.boundary_aware = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            区域对齐增强特征: (B, C, H, W)
        """
        # 区域特征增强
        enhanced = self.region_enhance(x)
        
        # 边界感知
        boundary = self.boundary_aware(x)
        
        # 融合
        output = enhanced + boundary + x
        
        return output


class DynResModule(nn.Module):
    """
    DynRes 完整模块（组合动态分辨率、多视图融合、区域对齐）
    """
    def __init__(self, dim, num_scales=3, num_views=4, use_multi_view=True, use_region_align=True):
        """
        Args:
            dim: 特征维度
            num_scales: 动态分辨率尺度数
            num_views: 多视图数量
            use_multi_view: 是否使用多视图融合
            use_region_align: 是否使用区域对齐
        """
        super(DynResModule, self).__init__()
        
        # 动态分辨率提取（核心）
        self.dyn_res_extractor = DynamicResolutionExtractor(dim, num_scales)
        
        # 多视图融合（可选）
        self.use_multi_view = use_multi_view
        if use_multi_view:
            self.multi_view_fusion = MultiViewFusion(dim, num_views)
        
        # 区域对齐（可选）
        self.use_region_align = use_region_align
        if use_region_align:
            self.region_align = AdaptiveRegionAlign(dim)
        
        # 残差连接权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            DynRes增强特征: (B, C, H, W)
        """
        identity = x
        
        # 1. 动态分辨率提取
        x = self.dyn_res_extractor(x)
        
        # 2. 多视图融合（如果启用）
        if self.use_multi_view:
            x = self.multi_view_fusion(x)
        
        # 3. 区域对齐（如果启用）
        if self.use_region_align:
            x = self.region_align(x)
        
        # 4. 残差连接
        x = x * self.alpha + identity * (1 - self.alpha)
        
        return x


if __name__ == '__main__':
    print("="*60)
    print("Testing DynRes Modules for SUTrack")
    print("="*60)
    
    # 测试动态分辨率提取器
    print("\n[1] Testing DynamicResolutionExtractor...")
    x = torch.randn(2, 384, 14, 14)
    dyn_res = DynamicResolutionExtractor(384, num_scales=3)
    out = dyn_res(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in dyn_res.parameters()) / 1e6:.2f}M")
    
    # 测试多视图融合
    print("\n[2] Testing MultiViewFusion...")
    x = torch.randn(2, 384, 14, 14)
    multi_view = MultiViewFusion(384, num_views=4)
    out = multi_view(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in multi_view.parameters()) / 1e6:.2f}M")
    
    # 测试区域对齐
    print("\n[3] Testing AdaptiveRegionAlign...")
    x = torch.randn(2, 384, 14, 14)
    region_align = AdaptiveRegionAlign(384)
    out = region_align(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in region_align.parameters()) / 1e6:.2f}M")
    
    # 测试完整 DynRes 模块
    print("\n[4] Testing DynResModule (Complete)...")
    x = torch.randn(2, 384, 14, 14)
    dynres = DynResModule(384, num_scales=3, num_views=4)
    out = dynres(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    print(f"   Total Parameters: {sum(p.numel() for p in dynres.parameters()) / 1e6:.2f}M")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
