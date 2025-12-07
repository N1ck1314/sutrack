"""
Cross-Modal Attention (CMA) Mechanism
跨模态注意力机制

基于SalM²论文的跨模态注意力融合模块，用于融合视觉特征与语义信息。
适用于多模态信息融合场景，如驾驶场景理解、视觉问答等。

参考论文: SalM²: Saliency-Aware Language-Guided Driving with Mamba
核心功能:
1. 跨模态注意力融合：通过注意力机制统一语义特征与图像特征的维度
2. 语义引导的注意力预测：利用语义信息引导视觉注意力分配
3. 轻量化设计：仅引入一个可学习参数gamma
"""

import torch
from torch import nn


class CMA(nn.Module):
    """
    Cross-Modal Attention Mechanism（跨模态注意力机制）
    
    通过通道注意力方式统一语义特征与图像特征的维度，实现跨模态信息融合。
    
    Args:
        无需额外参数，自动适配输入特征维度
    
    Forward:
        img_feat: [B, C, H, W] 图像特征（来自Bottom-up分支）
        text_feat: [B, C, H, W] 语义特征（来自Top-down分支）
    
    Returns:
        output: [B, C, H, W] 融合后的特征
    """
    def __init__(self):
        super().__init__()
        # 定义一个可学习的参数gamma，初始值为0
        # gamma控制语义信息对视觉特征的影响权重
        self.gamma = nn.Parameter(torch.zeros(1))
        # 定义一个Softmax层，用于在最后一个维度上进行softmax操作
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_feat, text_feat):
        """
        跨模态注意力融合前向传播
        
        Args:
            img_feat: [B, C, H, W] 视觉特征张量
            text_feat: [B, C, H, W] 语义特征张量
            
        Returns:
            output: [B, C, H, W] 融合后的特征
        """
        # 获取图像特征张量的形状，B为批次大小，C为通道数，H为高度，W为宽度
        B, C, H, W = img_feat.shape

        # 将图像特征张量进行变形，将后两个维度展平，方便后续矩阵乘法操作
        # q: [B, C, H*W]
        q = img_feat.view(B, C, -1)
        
        # 将文本特征张量进行变形，展平后交换最后两个维度，以便后续与q进行矩阵乘法
        # k: [B, H*W, C]
        k = text_feat.view(B, C, -1).permute(0, 2, 1)
        
        # 计算图像特征和文本特征的注意力映射，通过矩阵乘法得到
        # attention_map: [B, C, C] 表示每个通道之间的相似度
        attention_map = torch.bmm(q, k)
        
        # 对注意力映射进行softmax操作，使其值在0到1之间且总和为1
        attention_map = self.softmax(attention_map)

        # 将文本特征张量进行变形，展平以便后续与注意力映射进行矩阵乘法
        # v: [B, C, H*W]
        v = text_feat.view(B, C, -1)
        
        # 通过注意力映射对文本特征进行加权，得到注意力信息
        # attention_info: [B, C, H*W]
        attention_info = torch.bmm(attention_map, v)
        
        # 将注意力信息的形状恢复为与输入图像特征相同的形状
        attention_info = attention_info.view(B, C, H, W)
        
        # 将注意力信息与图像特征进行加权融合，得到最终输出
        # gamma从0开始学习，逐步调整语义信息的贡献度
        output = self.gamma * attention_info + img_feat
        
        return output


class MultiModalFusionWithCMA(nn.Module):
    """
    基于CMA的多模态融合模块
    用于SUTrack中融合不同模态的特征（RGB + Depth/Thermal/Event等）
    
    Args:
        dim: 特征维度
        fusion_mode: 融合模式，'cma' 或 'simple'
    """
    def __init__(self, dim, fusion_mode='cma'):
        super().__init__()
        self.dim = dim
        self.fusion_mode = fusion_mode
        
        if fusion_mode == 'cma':
            # 使用CMA模块进行跨模态融合
            self.cma = CMA()
        elif fusion_mode == 'simple':
            # 简单的卷积融合作为对比
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
    
    def forward(self, feat1, feat2):
        """
        多模态特征融合
        
        Args:
            feat1: [B, C, H, W] 或 [B, N, C] 第一个模态特征（如RGB）
            feat2: [B, C, H, W] 或 [B, N, C] 第二个模态特征（如Depth/Thermal）
            
        Returns:
            fused: 融合后的特征
        """
        # 处理输入维度：如果是token序列格式[B, N, C]，转换为[B, C, H, W]
        if feat1.dim() == 3:  # [B, N, C]
            B, N, C = feat1.shape
            H = W = int(N ** 0.5)
            feat1 = feat1.permute(0, 2, 1).reshape(B, C, H, W)
            feat2 = feat2.permute(0, 2, 1).reshape(B, C, H, W)
            reshape_back = True
        else:  # [B, C, H, W]
            reshape_back = False
        
        if self.fusion_mode == 'cma':
            # 使用CMA进行跨模态注意力融合
            fused = self.cma(feat1, feat2)
        elif self.fusion_mode == 'simple':
            # 简单拼接后卷积融合
            concat = torch.cat([feat1, feat2], dim=1)
            fused = self.fusion_conv(concat)
        
        # 如果需要，恢复为token序列格式
        if reshape_back:
            B, C, H, W = fused.shape
            fused = fused.reshape(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        
        return fused


class DualBranchCMA(nn.Module):
    """
    双分支CMA模块：Bottom-up + Top-down
    
    适用于需要同时处理低层视觉特征和高层语义特征的场景
    
    Args:
        visual_dim: 视觉特征维度
        semantic_dim: 语义特征维度（可与视觉特征不同）
    """
    def __init__(self, visual_dim, semantic_dim=None):
        super().__init__()
        if semantic_dim is None:
            semantic_dim = visual_dim
        
        self.visual_dim = visual_dim
        self.semantic_dim = semantic_dim
        
        # 如果维度不同，使用投影层对齐
        if visual_dim != semantic_dim:
            self.semantic_proj = nn.Conv2d(semantic_dim, visual_dim, kernel_size=1)
        else:
            self.semantic_proj = nn.Identity()
        
        # CMA模块
        self.cma = CMA()
        
        # 可选的后处理层
        self.post_conv = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, visual_feat, semantic_feat):
        """
        双分支融合
        
        Args:
            visual_feat: [B, C1, H, W] Bottom-up视觉特征
            semantic_feat: [B, C2, H, W] Top-down语义特征
            
        Returns:
            output: [B, C1, H, W] 融合后的特征
        """
        # 对齐语义特征维度
        semantic_feat = self.semantic_proj(semantic_feat)
        
        # CMA融合
        fused = self.cma(visual_feat, semantic_feat)
        
        # 后处理
        output = self.post_conv(fused)
        
        return output


if __name__ == "__main__":
    print("=" * 60)
    print("测试CMA模块")
    print("=" * 60)
    
    # 测试1: 基础CMA
    print("\n测试1: 基础CMA模块")
    img_feat = torch.randn(2, 32, 50, 50)
    text_feat = torch.randn(2, 32, 50, 50)
    model = CMA()
    output = model(img_feat, text_feat)
    print(f'图像特征输入: {img_feat.size()}')
    print(f'文本特征输入: {text_feat.size()}')
    print(f'融合输出: {output.size()}')
    print(f'Gamma参数值: {model.gamma.item():.4f}')
    
    # 测试2: MultiModalFusionWithCMA
    print("\n测试2: 多模态融合模块")
    fusion_cma = MultiModalFusionWithCMA(dim=512, fusion_mode='cma')
    feat1 = torch.randn(2, 512, 14, 14)  # RGB特征
    feat2 = torch.randn(2, 512, 14, 14)  # Depth特征
    fused = fusion_cma(feat1, feat2)
    print(f'模态1特征: {feat1.size()}')
    print(f'模态2特征: {feat2.size()}')
    print(f'融合输出: {fused.size()}')
    
    # 测试3: Token序列格式
    print("\n测试3: Token序列格式输入")
    token1 = torch.randn(2, 196, 512)  # [B, N, C]
    token2 = torch.randn(2, 196, 512)
    fused_token = fusion_cma(token1, token2)
    print(f'Token1输入: {token1.size()}')
    print(f'Token2输入: {token2.size()}')
    print(f'融合输出: {fused_token.size()}')
    
    # 测试4: 双分支CMA
    print("\n测试4: 双分支CMA模块")
    dual_cma = DualBranchCMA(visual_dim=256, semantic_dim=512)
    visual = torch.randn(2, 256, 28, 28)
    semantic = torch.randn(2, 512, 28, 28)
    output = dual_cma(visual, semantic)
    print(f'视觉特征: {visual.size()}')
    print(f'语义特征: {semantic.size()}')
    print(f'融合输出: {output.size()}')
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
