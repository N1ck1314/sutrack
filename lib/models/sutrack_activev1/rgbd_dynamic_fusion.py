"""
RGBD 动态融合模块 - 简化版
直接集成到 Transformer Block 中，实现层间深度选择
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBDDynamicFusion(nn.Module):
    """
    动态RGBD融合模块
    根据RGB特征内容，动态决定是否融合深度信息
    """
    def __init__(self, dim, num_heads=8, depth_dim_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.depth_dim = int(dim * depth_dim_ratio)
        
        # 深度使用决策器（轻量级）
        self.depth_decider = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 2)  # 二分类：使用/不使用深度
        )
        
        # 深度特征投影
        self.depth_proj = nn.Linear(self.depth_dim, dim)
        
        # 融合层（仅在需要时使用）
        self.fusion_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.fusion_norm = nn.LayerNorm(dim)
        
        # 温度参数（用于Gumbel-Softmax）
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, rgb_feat, depth_feat=None, return_decision=False):
        """
        Args:
            rgb_feat: (B, N, C) RGB特征
            depth_feat: (B, N, C//2) 深度特征（可选）
            return_decision: 是否返回决策信息
        Returns:
            output: (B, N, C) 融合后的特征
            decision: dict 决策信息（可选）
        """
        B, N, C = rgb_feat.shape
        
        # 基于RGB特征决定是否使用深度
        global_rgb = rgb_feat.mean(dim=1)  # (B, C)
        decision_logits = self.depth_decider(global_rgb)  # (B, 2)
        
        # Gumbel-Softmax 实现可微分的硬选择
        if self.training:
            # 训练时使用软选择
            probs = F.gumbel_softmax(decision_logits, tau=self.temperature, hard=False)
            use_depth_prob = probs[:, 1:2]  # (B, 1)
        else:
            # 推理时硬选择
            use_depth = decision_logits.argmax(dim=-1)  # (B,)
            use_depth_prob = use_depth.float().unsqueeze(1)  # (B, 1)
        
        # 基础RGB处理（总是执行）
        output = rgb_feat
        
        # 条件深度融合
        if depth_feat is not None and use_depth_prob.mean() > 0.01:
            # 投影深度特征
            depth_proj = self.depth_proj(depth_feat)  # (B, N, C)
            
            # 融合RGB和深度
            residual = output
            output_norm = self.fusion_norm(output)
            
            # Concatenate for cross-attention
            combined = torch.cat([output_norm, depth_proj], dim=1)  # (B, 2N, C)
            
            # Self-attention with depth
            attn_out, _ = self.fusion_attn(output_norm, combined, combined)
            
            # 应用深度使用概率作为门控
            output = residual + use_depth_prob.unsqueeze(1) * attn_out
        
        if return_decision:
            decision = {
                'use_depth_prob': use_depth_prob.mean().item(),
                'temperature': self.temperature.item(),
                'logits': decision_logits
            }
            return output, decision
        
        return output


class LayerwiseDepthGate(nn.Module):
    """
    层间深度门控
    为每一层学习一个门控，决定该层是否使用深度
    """
    def __init__(self, num_layers=12, dim=384):
        super().__init__()
        self.num_layers = num_layers
        
        # 可学习的层门控（0-1之间）
        self.layer_gates = nn.Parameter(torch.ones(num_layers) * 0.5)
        
        # 层特征编码器
        self.layer_encoder = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.GELU()
        )
        
        # 层间依赖（简单RNN）
        self.layer_rnn = nn.GRUCell(dim // 4, dim // 4)
        
    def forward(self, layer_idx, layer_feat, prev_hidden=None):
        """
        Args:
            layer_idx: 当前层索引
            layer_feat: (B, N, C) 当前层特征
            prev_hidden: 前一层隐藏状态 (B, C//4)
        Returns:
            gate: (B, 1) 门控值
            new_hidden: 新的隐藏状态
        """
        B = layer_feat.size(0)
        
        # 编码当前层特征
        layer_state = self.layer_encoder(layer_feat.mean(dim=1))  # (B, C//4)
        
        # RNN更新
        if prev_hidden is not None:
            new_hidden = self.layer_rnn(layer_state, prev_hidden)
        else:
            new_hidden = layer_state
        
        # 基础门控（可学习参数）
        base_gate = torch.sigmoid(self.layer_gates[layer_idx])
        
        # 动态调整
        adjustment = torch.tanh(new_hidden.mean(dim=-1, keepdim=True)) * 0.1
        gate = torch.clamp(base_gate + adjustment, 0, 1)
        
        return gate, new_hidden


class FastRGBDBlock(nn.Module):
    """
    快速RGBD Transformer Block
    在每个block中动态选择是否使用深度
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., depth_ratio=0.5):
        super().__init__()
        self.dim = dim
        
        # 标准RGB处理
        self.rgb_norm1 = nn.LayerNorm(dim)
        self.rgb_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.rgb_norm2 = nn.LayerNorm(dim)
        self.rgb_mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
        # 动态深度融合
        self.depth_fusion = RGBDDynamicFusion(dim, num_heads, depth_ratio)
        
        # 层门控
        self.layer_gate = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, depth_feat=None, return_stats=False):
        """
        Args:
            x: (B, N, C) RGB特征
            depth_feat: (B, N, C//2) 深度特征（可选）
            return_stats: 返回统计信息
        Returns:
            x: (B, N, C) 输出特征
            stats: 统计信息（可选）
        """
        # 标准RGB self-attention
        residual = x
        x = self.rgb_norm1(x)
        attn_out, _ = self.rgb_attn(x, x, x)
        x = residual + attn_out
        
        # MLP
        residual = x
        x = x + self.rgb_mlp(self.rgb_norm2(x))
        
        # 动态深度融合
        if depth_feat is not None:
            x, decision = self.depth_fusion(x, depth_feat, return_decision=True)
            
            if return_stats:
                stats = {
                    'depth_prob': decision['use_depth_prob'],
                    'layer_gate': torch.sigmoid(self.layer_gate).item()
                }
                return x, stats
        
        return x


# 训练辅助函数
def compute_depth_efficiency_loss(depth_probs, target_ratio=0.5):
    """
    计算深度使用效率损失
    鼓励模型在保持精度的前提下，减少深度使用
    
    Args:
        depth_probs: list of 深度使用概率
        target_ratio: 目标深度使用比例
    Returns:
        loss: 效率损失
    """
    if len(depth_probs) == 0:
        return torch.tensor(0.0)
    
    # 平均深度使用比例
    avg_prob = sum(depth_probs) / len(depth_probs)
    
    # L2损失：鼓励接近目标比例
    efficiency_loss = F.mse_loss(
        torch.tensor(avg_prob),
        torch.tensor(target_ratio)
    )
    
    return efficiency_loss


# 统计和监控类
class DepthUsageMonitor:
    """
    监控深度使用情况
    """
    def __init__(self, num_layers=12):
        self.num_layers = num_layers
        self.reset()
        
    def reset(self):
        self.layer_usage = [[] for _ in range(self.num_layers)]
        self.total_samples = 0
        
    def update(self, layer_idx, use_depth_prob):
        self.layer_usage[layer_idx].append(use_depth_prob)
        
    def get_stats(self):
        stats = {}
        for i, usage in enumerate(self.layer_usage):
            if len(usage) > 0:
                stats[f'layer_{i}_depth_usage'] = sum(usage) / len(usage)
        
        # 整体统计
        all_usage = [u for layer in self.layer_usage for u in layer]
        if len(all_usage) > 0:
            stats['avg_depth_usage'] = sum(all_usage) / len(all_usage)
            stats['estimated_speedup'] = 1.0 / (1.0 + stats['avg_depth_usage'] * 0.5)
        
        return stats


if __name__ == '__main__':
    # 测试
    B, N, C = 2, 196, 384
    D = C // 2
    
    print("Testing RGBD Dynamic Fusion...")
    
    # 测试动态融合
    fusion = RGBDDynamicFusion(C)
    rgb = torch.randn(B, N, C)
    depth = torch.randn(B, N, D)
    
    output, decision = fusion(rgb, depth, return_decision=True)
    print(f"Output shape: {output.shape}")
    print(f"Depth usage prob: {decision['use_depth_prob']:.3f}")
    
    # 测试完整block
    print("\nTesting Fast RGBD Block...")
    block = FastRGBDBlock(C)
    output, stats = block(rgb, depth, return_stats=True)
    print(f"Output shape: {output.shape}")
    print(f"Stats: {stats}")
    
    # 估算加速比
    speedup = 1.0 / (1.0 + stats['depth_prob'] * 0.5)
    print(f"\nEstimated speedup: {speedup:.2f}x")
