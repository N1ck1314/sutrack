"""
测试脚本 - SGLA模块
测试 Similarity-Guided Layer-Adaptive (SGLA) 的核心组件
"""
import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from lib.models.sutrack_SGLA.sgla_modules import SelectionModule, SimilarityLoss, LayerAdaptiveWrapper

def test_sgla_modules():
    print("=" * 60)
    print("测试 1: SelectionModule")
    print("=" * 60)
    
    B, N, C = 2, 196, 384
    num_layers = 12
    x = torch.randn(B, N, C)
    
    selector = SelectionModule(C, num_layers)
    probs = selector(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出概率形状: {probs.shape}")
    print(f"概率范围: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    
    assert probs.shape == (B, num_layers)
    assert (probs >= 0).all() and (probs <= 1).all()
    print("✅ SelectionModule 测试通过")

    print("\n" + "=" * 60)
    print("测试 2: SimilarityLoss")
    print("=" * 60)
    
    loss_fn = SimilarityLoss(mode='cosine')
    
    # 模拟相似特征
    f1 = torch.randn(B, N, C)
    f2 = f1 + torch.randn(B, N, C) * 0.1
    features_similar = [f1, f2]
    loss_similar = loss_fn(features_similar)
    
    # 模拟不相似特征
    f3 = torch.randn(B, N, C)
    features_dissimilar = [f1, f3]
    loss_dissimilar = loss_fn(features_dissimilar)
    
    print(f"相似特征 Loss: {loss_similar.item():.4f}")
    print(f"不相似特征 Loss: {loss_dissimilar.item():.4f}")
    
    # 相似特征的余弦相似度损失应该更高 (趋近于1)
    # 因为 SimilarityLoss 计算的是均值相似度，而不是 1-sim
    # SGLATrack 论文中，相似度越高表示冗余越大，通常希望最小化这个损失或者用于指导
    assert loss_similar > loss_dissimilar
    print("✅ SimilarityLoss 测试通过")

    print("\n" + "=" * 60)
    print("测试 3: LayerAdaptiveWrapper")
    print("=" * 60)
    
    class MockBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(C, C)
        def forward(self, x, **kwargs):
            return x + self.linear(x)
            
    block = MockBlock()
    wrapper = LayerAdaptiveWrapper(block)
    
    # 测试推理模式 (prob > 0.5)
    wrapper.eval()
    prob_on = torch.tensor([0.9])
    out_on = wrapper(x, prob=prob_on)
    print("推理模式 (prob=0.9): Block 已执行")
    assert not torch.allclose(x, out_on)
    
    # 测试推理模式 (prob < 0.5)
    prob_off = torch.tensor([0.1])
    out_off = wrapper(x, prob=prob_off)
    print("推理模式 (prob=0.1): Block 已跳过")
    assert torch.allclose(x, out_off)
    
    # 测试训练模式 (随机性)
    wrapper.train()
    print("训练模式: 启用随机采样")
    out_train = wrapper(x, prob=torch.tensor([0.5]))
    print(f"训练模式输出形状: {out_train.shape}")
    assert out_train.shape == x.shape
    
    print("✅ LayerAdaptiveWrapper 测试通过")

if __name__ == '__main__':
    test_sgla_modules()
    print("\n" + "=" * 60)
    print("🎉 所有 SGLA 模块测试通过！")
    print("=" * 60)
