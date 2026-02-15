"""
自适应深度融合模块 (Adaptive Depth Fusion)
让模型在每一层动态选择是否使用深度信息，加速RGBD推理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthGateController(nn.Module):
    """
    深度门控控制器：决定当前层是否使用深度信息
    基于RGB特征的内容自适应决策
    """
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = dim
        
        # RGB特征分析器：评估当前场景是否需要深度信息
        self.rgb_analyzer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        
        # 深度信息需求评估器
        self.depth_need_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 场景复杂度评估（用于辅助决策）
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_feat):
        """
        Args:
            rgb_feat: (B, N, C) RGB特征
        Returns:
            depth_gate: (B, 1) 深度使用门控 (0-1)
            complexity: (B, 1) 场景复杂度
            need_depth: (B, 1) 深度需求度
        """
        # 全局平均池化获取整体特征
        global_feat = rgb_feat.mean(dim=1)  # (B, C)
        
        # 分析RGB特征
        analyzed = self.rgb_analyzer(global_feat)  # (B, hidden_dim//2)
        
        # 评估深度需求
        need_depth = self.depth_need_estimator(analyzed)  # (B, 1)
        
        # 评估场景复杂度
        complexity = self.complexity_estimator(analyzed)  # (B, 1)
        
        # 综合门控：复杂场景或需要深度的场景使用深度
        # 门控值 = 0.3 * complexity + 0.7 * need_depth
        depth_gate = 0.3 * complexity + 0.7 * need_depth
        
        return depth_gate, complexity, need_depth


class AdaptiveDepthFusion(nn.Module):
    """
    自适应深度融合模块
    根据场景动态决定是否融合深度信息
    """
    def __init__(self, dim, num_heads=8, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        # 深度门控控制器
        self.depth_gate = DepthGateController(dim)
        
        # RGB处理分支
        self.rgb_norm = nn.LayerNorm(dim)
        self.rgb_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.rgb_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
        
        # RGBD融合分支（当需要使用深度时）
        self.rgbd_norm = nn.LayerNorm(dim)
        self.rgbd_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.rgbd_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
        
        # 深度特征投影（将深度特征投影到RGB特征空间）
        self.depth_proj = nn.Sequential(
            nn.Linear(dim // 2, dim),  # 假设深度特征维度是RGB的一半
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # 输出融合
        self.output_fusion = nn.Linear(dim * 2, dim)
        
    def forward(self, rgb_feat, depth_feat=None, return_gate=False):
        """
        Args:
            rgb_feat: (B, N_rgb, C) RGB特征
            depth_feat: (B, N_depth, C//2) 深度特征（可选）
            return_gate: 是否返回门控值
        Returns:
            output: (B, N_rgb, C) 融合后的特征
            gate_value: (B, 1) 门控值（如果return_gate=True）
        """
        B = rgb_feat.size(0)
        
        # 评估是否需要深度信息
        gate, complexity, need_depth = self.depth_gate(rgb_feat)
        
        # RGB-only 分支（总是计算）
        rgb_residual = rgb_feat
        rgb_feat_norm = self.rgb_norm(rgb_feat)
        rgb_attn_out, _ = self.rgb_attn(rgb_feat_norm, rgb_feat_norm, rgb_feat_norm)
        rgb_feat = rgb_residual + rgb_attn_out
        rgb_feat = rgb_feat + self.rgb_mlp(self.rgb_norm(rgb_feat))
        
        # RGBD 融合分支（条件计算）
        if depth_feat is not None and gate.mean() > 0.1:  # 至少部分样本需要深度
            # 投影深度特征
            depth_proj = self.depth_proj(depth_feat)  # (B, N_depth, C)
            
            # 软门控：根据gate值加权融合
            rgbd_residual = rgb_feat
            rgbd_feat_norm = self.rgbd_norm(rgb_feat)
            
            # Cross-attention: RGB查询，RGBD键值
            rgbd_concat = torch.cat([rgbd_feat_norm, depth_proj], dim=1)
            rgbd_attn_out, _ = self.rgbd_cross_attn(
                rgbd_feat_norm, rgbd_concat, rgbd_concat
            )
            
            # 应用门控
            gate_expanded = gate.unsqueeze(1)  # (B, 1, 1)
            rgbd_feat = rgbd_residual + gate_expanded * rgbd_attn_out
            rgbd_feat = rgbd_feat + gate_expanded * self.rgbd_mlp(self.rgbd_norm(rgbd_feat))
            
            # 融合RGB-only和RGBD结果
            combined = torch.cat([rgb_feat, rgbd_feat], dim=-1)
            output = self.output_fusion(combined)
        else:
            # 不使用深度，直接返回RGB结果
            output = rgb_feat
        
        if return_gate:
            return output, gate.mean()
        return output


class LayerwiseDepthSelector(nn.Module):
    """
    层间深度选择器
    为每一层学习是否使用深度信息，形成层间策略
    """
    def __init__(self, num_layers=12, dim=384):
        super().__init__()
        self.num_layers = num_layers
        
        # 每层的学习门控（可学习的参数）
        self.layer_gates = nn.Parameter(torch.ones(num_layers) * 0.5)
        
        # 层间依赖建模（LSTM）
        self.layer_lstm = nn.LSTM(
            input_size=dim // 4,
            hidden_size=dim // 4,
            num_layers=1,
            batch_first=True
        )
        
        # 层状态编码器
        self.layer_state_encoder = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.GELU()
        )
        
    def forward(self, layer_idx, layer_feat, prev_state=None):
        """
        Args:
            layer_idx: 当前层索引
            layer_feat: (B, N, C) 当前层特征
            prev_state: LSTM前一层状态
        Returns:
            use_depth: bool 是否使用深度
            new_state: LSTM新状态
        """
        B = layer_feat.size(0)
        
        # 编码层状态
        state = self.layer_state_encoder(layer_feat.mean(dim=1))  # (B, C//4)
        
        # LSTM处理层间依赖
        if prev_state is not None:
            lstm_out, new_state = self.layer_lstm(
                state.unsqueeze(1), prev_state
            )
            lstm_out = lstm_out.squeeze(1)  # (B, C//4)
        else:
            lstm_out = state
            new_state = None
        
        # 结合学习门控和LSTM输出
        learned_gate = torch.sigmoid(self.layer_gates[layer_idx])
        
        # 动态调整：根据LSTM输出微调门控
        adjustment = torch.tanh(lstm_out.mean(dim=-1, keepdim=True)) * 0.2
        final_gate = torch.clamp(learned_gate + adjustment, 0, 1)
        
        # 硬决策（推理时）或软门控（训练时）
        if self.training:
            use_depth = final_gate > 0.5
        else:
            use_depth = final_gate > 0.5
            
        return use_depth, final_gate, new_state


class EfficientRGBDTransformer(nn.Module):
    """
    高效的RGBD Transformer
    整合上述所有模块，实现自适应深度融合
    """
    def __init__(self, dim=384, num_layers=12, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # 层间深度选择器
        self.layer_selector = LayerwiseDepthSelector(num_layers, dim)
        
        # 自适应深度融合层
        self.fusion_layers = nn.ModuleList([
            AdaptiveDepthFusion(dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # 统计信息
        self.register_buffer('depth_usage_count', torch.zeros(num_layers))
        self.register_buffer('total_forward_count', torch.zeros(1))
        
    def forward(self, rgb_features, depth_features=None, return_stats=False):
        """
        Args:
            rgb_features: list of (B, N, C) RGB特征（每层一个）
            depth_features: list of (B, N, C//2) 深度特征（可选）
            return_stats: 是否返回使用统计
        Returns:
            outputs: 处理后的特征
            stats: 使用统计（如果return_stats=True）
        """
        B = rgb_features[0].size(0)
        outputs = []
        prev_state = None
        layer_gates = []
        
        for layer_idx in range(self.num_layers):
            rgb_feat = rgb_features[layer_idx]
            depth_feat = depth_features[layer_idx] if depth_features else None
            
            # 层间决策：是否使用深度
            use_depth, gate, prev_state = self.layer_selector(
                layer_idx, rgb_feat, prev_state
            )
            layer_gates.append(gate.mean().item())
            
            # 自适应融合
            if use_depth.any() and depth_feat is not None:
                output, gate_val = self.fusion_layers[layer_idx](
                    rgb_feat, depth_feat, return_gate=True
                )
                self.depth_usage_count[layer_idx] += use_depth.sum().item()
            else:
                output = self.fusion_layers[layer_idx](rgb_feat, None)
            
            outputs.append(output)
        
        self.total_forward_count += B
        
        if return_stats:
            stats = {
                'layer_gates': layer_gates,
                'depth_usage_ratio': (self.depth_usage_count / self.total_forward_count).tolist(),
                'avg_gate': sum(layer_gates) / len(layer_gates)
            }
            return outputs, stats
        
        return outputs


# 辅助函数：计算理论加速比
def calculate_speedup(depth_usage_ratio, depth_computation_cost=1.5):
    """
    计算理论加速比
    Args:
        depth_usage_ratio: 使用深度的比例 (0-1)
        depth_computation_cost: 深度融合的计算成本倍数
    Returns:
        speedup: 加速比 (>1表示加速)
    """
    base_cost = 1.0
    rgbd_cost = depth_usage_ratio * depth_computation_cost
    rgb_only_cost = (1 - depth_usage_ratio) * 1.0
    
    total_cost = rgbd_cost + rgb_only_cost
    speedup = base_cost / total_cost
    
    return speedup


if __name__ == '__main__':
    # 测试模块
    B, N, C = 2, 196, 384
    
    # 测试 DepthGateController
    gate_controller = DepthGateController(C)
    rgb_feat = torch.randn(B, N, C)
    gate, complexity, need = gate_controller(rgb_feat)
    print(f"Depth gate: {gate.squeeze()}")
    print(f"Complexity: {complexity.squeeze()}")
    print(f"Need depth: {need.squeeze()}")
    
    # 测试 AdaptiveDepthFusion
    fusion = AdaptiveDepthFusion(C)
    depth_feat = torch.randn(B, N, C // 2)
    output = fusion(rgb_feat, depth_feat)
    print(f"Fusion output shape: {output.shape}")
    
    # 计算加速比
    usage_ratio = gate.mean().item()
    speedup = calculate_speedup(usage_ratio)
    print(f"\nEstimated speedup: {speedup:.2f}x")
