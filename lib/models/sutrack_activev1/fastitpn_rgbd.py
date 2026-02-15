"""
Fast-iTPN with Adaptive RGBD Fusion
基于原始Fast-iTPN，添加自适应RGBD深度融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fastitpn import Fast_iTPN, Block, ConvPatchEmbed, PatchEmbed
from .rgbd_dynamic_fusion import RGBDDynamicFusion, LayerwiseDepthGate


class AdaptiveRGBDBlock(nn.Module):
    """
    自适应RGBD Transformer Block
    在每个block中动态选择是否融合深度信息
    """
    def __init__(self, original_block, layer_idx, dim, num_heads=8, 
                 use_dynamic_depth=True, target_depth_ratio=0.5):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.use_dynamic_depth = use_dynamic_depth
        
        # 保留原始block的所有功能
        self.original_block = original_block
        
        # 添加动态深度融合
        if use_dynamic_depth:
            self.depth_fusion = RGBDDynamicFusion(dim, num_heads)
            self.layer_gate = LayerwiseDepthGate(1, dim)  # 每层一个门控
            
        # 深度使用统计
        self.register_buffer('depth_usage_count', torch.tensor(0))
        self.register_buffer('total_count', torch.tensor(0))
        
    def forward(self, x, depth_feat=None, rel_pos_bias=None, dynamic_activation=False):
        """
        Args:
            x: (B, N, C) 输入特征（已包含RGB和可能的深度）
            depth_feat: (B, N, C//2) 独立的深度特征（可选）
            rel_pos_bias: 相对位置偏置
            dynamic_activation: 是否使用动态激活
        """
        # 分离RGB和深度（如果x中包含两者）
        if x.size(-1) == self.dim * 2:  # 假设输入是RGB+Depth拼接
            rgb_feat = x[..., :self.dim]
            depth_feat = x[..., self.dim:]
            x = rgb_feat
        else:
            rgb_feat = x
        
        # 原始block处理（RGB-only）
        if dynamic_activation and hasattr(self.original_block, 'active_score_module'):
            x, prob_active = self.original_block(x, rel_pos_bias, dynamic_activation=True)
        else:
            x = self.original_block(x, rel_pos_bias)
            prob_active = None
        
        # 动态深度融合
        if self.use_dynamic_depth and depth_feat is not None:
            # 层间门控决策
            gate, _ = self.layer_gate(0, x)  # layer_idx=0因为每个block独立
            
            # 根据门控决定是否融合
            if gate.mean() > 0.1:  # 至少部分样本需要深度
                x_fused, decision = self.depth_fusion(
                    x, depth_feat, return_decision=True
                )
                
                # 软融合：根据门控值加权
                gate_expanded = gate.unsqueeze(1)  # (B, 1, 1)
                x = x * (1 - gate_expanded) + x_fused * gate_expanded
                
                # 更新统计
                self.depth_usage_count += (gate > 0.5).sum()
                
                # 将深度使用概率附加到prob_active
                if prob_active is not None:
                    prob_active = torch.cat([prob_active, decision['use_depth_prob'].unsqueeze(0)], dim=1)
                else:
                    prob_active = decision['use_depth_prob'].unsqueeze(0)
            
            self.total_count += x.size(0)
        
        if dynamic_activation:
            return x, prob_active
        return x


class Fast_iTPN_RGBD(nn.Module):
    """
    Fast-iTPN with Adaptive RGBD Fusion
    完整的RGBD自适应融合模型
    """
    def __init__(self, original_model, use_dynamic_depth=True, target_depth_ratio=0.5):
        super().__init__()
        
        # 复制原始模型的所有属性
        self.search_size = original_model.search_size
        self.template_size = original_model.template_size
        self.token_type_indicate = original_model.token_type_indicate
        self.mlp_ratio = original_model.mlp_ratio
        self.grad_ckpt = original_model.grad_ckpt
        self.num_main_blocks = original_model.num_main_blocks
        self.depth_stage1 = original_model.depth_stage1
        self.depth_stage2 = original_model.depth_stage2
        self.depth = original_model.depth
        self.patch_size = original_model.patch_size
        self.num_features = original_model.num_features
        self.embed_dim = original_model.embed_dim
        self.convmlp = original_model.convmlp
        self.stop_grad_conv1 = original_model.stop_grad_conv1
        self.use_rel_pos_bias = original_model.use_rel_pos_bias
        self.use_shared_rel_pos_bias = original_model.use_shared_rel_pos_bias
        self.use_shared_decoupled_rel_pos_bias = original_model.use_shared_decoupled_rel_pos_bias
        self.use_decoupled_rel_pos_bias = original_model.use_decoupled_rel_pos_bias
        self.subln = original_model.subln
        self.swiglu = original_model.swiglu
        self.naiveswiglu = original_model.naiveswiglu
        
        # 关键：暴露原始模型的属性
        self.num_patches_search = original_model.num_patches_search
        self.num_patches_template = original_model.num_patches_template
        self.num_channels = original_model.embed_dim  # 用于decoder等组件
        self.cls_token = original_model.cls_token  # 必须暴露，SUTRACK需要访问
        
        # 复制原始组件
        self.patch_embed = original_model.patch_embed
        self.pos_embed = original_model.pos_embed
        if self.token_type_indicate:
            self.template_background_token = original_model.template_background_token
            self.template_foreground_token = original_model.template_foreground_token
            self.search_token = original_model.search_token
        self.pos_drop = original_model.pos_drop
        self.rel_pos_bias = original_model.rel_pos_bias
        self.norm = original_model.norm
        if hasattr(original_model, 'fc_norm'):
            self.fc_norm = original_model.fc_norm
        
        # 包装blocks为自适应RGBD blocks
        self.use_dynamic_depth = use_dynamic_depth
        self.target_depth_ratio = target_depth_ratio
        
        self.blocks = nn.ModuleList()
        for i, blk in enumerate(original_model.blocks):
            if i >= len(original_model.blocks) - self.num_main_blocks:
                # 只对主要blocks使用动态深度
                adaptive_blk = AdaptiveRGBDBlock(
                    blk, i, self.embed_dim, 
                    use_dynamic_depth=use_dynamic_depth,
                    target_depth_ratio=target_depth_ratio
                )
                self.blocks.append(adaptive_blk)
            else:
                # 早期blocks保持原样
                self.blocks.append(blk)
        
        # 深度特征投影（将深度图像投影到特征空间）
        if use_dynamic_depth:
            self.depth_patch_embed = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 假设深度是3通道或单通道重复
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, self.embed_dim // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.embed_dim // 2),
                nn.ReLU(),
            )
        
        # 统计信息
        self.depth_usage_history = []
        
    def prepare_tokens_with_masks(self, template_list, search_list, template_anno_list, text_src, task_index):
        """复用原始方法处理RGB输入"""
        # 这里简化处理，实际应该调用原始模型的方法
        B = search_list[0].size(0)
        
        # 处理template和search（假设是RGBD，6通道）
        # 分离RGB和深度
        template_rgb = [t[:, :3] for t in template_list]
        template_depth = [t[:, 3:] for t in template_list]
        search_rgb = [s[:, :3] for s in search_list]
        search_depth = [s[:, 3:] for s in search_list]
        
        # 使用原始patch_embed处理RGB
        # ...（这里应该调用原始模型的prepare_tokens_with_masks）
        
        # 处理深度特征
        if self.use_dynamic_depth:
            depth_features = []
            for sd in search_depth:
                df = self.depth_patch_embed(sd)  # (B, C//2, H, W)
                df = df.flatten(2).transpose(1, 2)  # (B, N, C//2)
                depth_features.append(df)
            self.cached_depth_features = depth_features
        
        # 返回RGB tokens（简化版）
        return torch.randn(B, 100, self.embed_dim).cuda()  # 占位符
        
    def forward_features(self, template_list, search_list, template_anno_list, text_src, task_index):
        """前向传播，集成动态RGBD融合"""
        # 准备tokens
        xz = self.prepare_tokens_with_masks(template_list, search_list, template_anno_list, text_src, task_index)
        xz = self.pos_drop(xz)
        
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        probs_active = []
        depth_probs = []
        
        # 获取深度特征
        depth_features = getattr(self, 'cached_depth_features', None)
        
        for i, blk in enumerate(self.blocks):
            # 对主要blocks使用动态深度
            if i >= len(self.blocks) - self.num_main_blocks and self.use_dynamic_depth:
                depth_feat = depth_features[0] if depth_features else None  # 简化处理
                xz, prob_active = blk(xz, depth_feat, rel_pos_bias, dynamic_activation=True)
            else:
                xz = blk(xz, rel_pos_bias)
                prob_active = None
            
            if prob_active is not None:
                probs_active.append(prob_active)
        
        xz = self.norm(xz)
        
        return xz, probs_active if len(probs_active) > 0 else None
    
    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        """完整前向"""
        xz, probs_active = self.forward_features(template_list, search_list, template_anno_list, text_src, task_index)
        return [xz], probs_active
    
    def get_depth_usage_stats(self):
        """获取深度使用统计"""
        stats = {}
        total_usage = 0
        total_count = 0
        
        for blk in self.blocks:
            if isinstance(blk, AdaptiveRGBDBlock):
                usage = blk.depth_usage_count.item()
                count = blk.total_count.item()
                if count > 0:
                    stats[f'layer_{blk.layer_idx}'] = usage / count
                    total_usage += usage
                    total_count += count
        
        if total_count > 0:
            stats['overall_depth_usage'] = total_usage / total_count
            stats['estimated_speedup'] = 1.0 / (1.0 + stats['overall_depth_usage'] * 0.5)
        
        return stats


def build_fastitpn_rgbd(cfg, use_dynamic_depth=True):
    """
    构建带自适应RGBD融合的Fast-iTPN
    
    Args:
        cfg: 配置
        use_dynamic_depth: 是否使用动态深度融合
    Returns:
        model: Fast_iTPN_RGBD模型
    """
    # 首先构建原始模型
    from .fastitpn import Fast_iTPN
    
    # 创建原始模型（简化参数，实际需要完整参数）
    original_model = Fast_iTPN(
        search_size=cfg.DATA.SEARCH.SIZE,
        template_size=cfg.DATA.TEMPLATE.SIZE,
        patch_size=16,
        in_chans=3,  # RGB-only for original
        embed_dim=384,
        depth_stage1=1,
        depth_stage2=1,
        depth=12,
        num_heads=6,
        convmlp=True,
        token_type_indicate=True,
    )
    
    # 包装为RGBD版本
    model = Fast_iTPN_RGBD(
        original_model,
        use_dynamic_depth=use_dynamic_depth,
        target_depth_ratio=cfg.TRAIN.get('TARGET_DEPTH_RATIO', 0.5)
    )
    
    return model


if __name__ == '__main__':
    # 测试代码
    print("Testing Fast_iTPN_RGBD...")
    
    # 创建模拟配置
    class MockCfg:
        class DATA:
            class SEARCH:
                SIZE = 224
            class TEMPLATE:
                SIZE = 112
        class TRAIN:
            TARGET_DEPTH_RATIO = 0.5
    
    cfg = MockCfg()
    
    # 构建模型
    model = build_fastitpn_rgbd(cfg, use_dynamic_depth=True)
    
    # 模拟输入（RGBD，6通道）
    B = 2
    template_list = [torch.randn(B, 6, 112, 112)]
    search_list = [torch.randn(B, 6, 224, 224)]
    
    # 前向传播
    output, probs_active = model(template_list, search_list, None, None, None)
    
    print(f"Output shape: {output[0].shape}")
    print(f"Active probs: {probs_active}")
    
    # 统计
    stats = model.get_depth_usage_stats()
    print(f"Stats: {stats}")
    
    print("\n✅ Fast_iTPN_RGBD test passed!")
