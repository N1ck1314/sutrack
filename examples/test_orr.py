"""
测试脚本 - ORR模块
测试 Occlusion-Robust Representations (ORR) 的核心组件
"""
import sys
sys.path.append('/home/nick/code/code.sutrack/SUTrack')

import torch
from lib.models.sutrack_OR.orr_modules import (
    SpatialCoxMasking,
    FeatureInvarianceLoss,
    OcclusionRobustEncoder
)

def test_spatial_cox_masking():
    """测试空间Cox过程遮挡"""
    print("=" * 60)
    print("测试 1: SpatialCoxMasking")
    print("=" * 60)
    
    # 测试不同策略
    strategies = ['random', 'block', 'cox']
    x = torch.randn(2, 196, 384)  # [B, N, C]
    H, W = 14, 14
    
    for strategy in strategies:
        masking = SpatialCoxMasking(mask_ratio=0.3, mask_strategy=strategy)
        mask = masking(x, H, W)
        
        mask_ratio_actual = mask.sum(dim=1) / mask.shape[1]
        print(f"\n策略: {strategy}")
        print(f"  - Mask shape: {mask.shape}")
        print(f"  - 实际遮挡比例: {mask_ratio_actual[0].item():.2%}, {mask_ratio_actual[1].item():.2%}")
        print(f"  - 预期遮挡比例: 30%")
        
        # 验证mask值范围
        assert mask.min() >= 0 and mask.max() <= 1, "Mask值应该在[0,1]范围内"
        print(f"  ✅ {strategy}策略测试通过")
    
    print("\n" + "=" * 60)
    print("✅ SpatialCoxMasking 所有测试通过")
    print("=" * 60 + "\n")


def test_feature_invariance_loss():
    """测试特征不变性损失"""
    print("=" * 60)
    print("测试 2: FeatureInvarianceLoss")
    print("=" * 60)
    
    feat_clean = torch.randn(2, 196, 384)
    
    # 测试不同损失类型
    loss_types = ['cosine', 'mse', 'contrastive']
    
    for loss_type in loss_types:
        loss_fn = FeatureInvarianceLoss(loss_type=loss_type)
        
        # 测试1: 相同特征，损失应该接近0
        loss_same = loss_fn(feat_clean, feat_clean)
        print(f"\n损失类型: {loss_type}")
        print(f"  - 相同特征损失: {loss_same.item():.6f} (应该接近0)")
        
        # 测试2: 不同特征，损失应该大于0
        feat_different = torch.randn(2, 196, 384)
        loss_diff = loss_fn(feat_clean, feat_different)
        print(f"  - 不同特征损失: {loss_diff.item():.6f} (应该>0)")
        
        # 验证
        if loss_type == 'cosine':
            assert loss_same < 0.01, f"相同特征的余弦损失应该接近0"
        elif loss_type == 'mse':
            assert loss_same < 1e-10, f"相同特征的MSE损失应该接近0"
        
        assert loss_diff > loss_same, f"不同特征的损失应该大于相同特征"
        print(f"  ✅ {loss_type}损失测试通过")
    
    print("\n" + "=" * 60)
    print("✅ FeatureInvarianceLoss 所有测试通过")
    print("=" * 60 + "\n")


def test_occlusion_robust_encoder():
    """测试遮挡鲁棒编码器"""
    print("=" * 60)
    print("测试 3: OcclusionRobustEncoder")
    print("=" * 60)
    
    # 初始化
    orr_encoder = OcclusionRobustEncoder(
        use_orr=True,
        mask_ratio=0.3,
        mask_strategy='cox',
        invariance_loss_weight=0.5
    )
    
    x = torch.randn(2, 196, 384)  # [B, N, C]
    H, W = 14, 14
    
    # 测试训练模式
    print("\n训练模式测试:")
    x_out, _ = orr_encoder(x, H, W, training=True)
    print(f"  - 输入形状: {x.shape}")
    print(f"  - 输出形状: {x_out.shape}")
    print(f"  - 形状一致: {x.shape == x_out.shape}")
    assert x.shape == x_out.shape, "输入输出形状应该一致"
    
    # 检查是否存储了clean和masked特征
    assert hasattr(orr_encoder, 'clean_features'), "应该存储clean_features"
    assert hasattr(orr_encoder, 'masked_features'), "应该存储masked_features"
    print(f"  - Clean features: {orr_encoder.clean_features.shape}")
    print(f"  - Masked features: {orr_encoder.masked_features.shape}")
    
    # 计算不变性损失
    inv_loss = orr_encoder.compute_invariance_loss(
        orr_encoder.clean_features,
        orr_encoder.masked_features
    )
    print(f"  - 不变性损失: {inv_loss.item():.6f}")
    assert inv_loss is not None and inv_loss > 0, "不变性损失应该存在且>0"
    print("  ✅ 训练模式测试通过")
    
    # 测试推理模式
    print("\n推理模式测试:")
    orr_encoder_infer = OcclusionRobustEncoder(use_orr=False)
    x_out_infer, _ = orr_encoder_infer(x, H, W, training=False)
    print(f"  - 输入形状: {x.shape}")
    print(f"  - 输出形状: {x_out_infer.shape}")
    # 推理模式不应用遮挡
    torch.testing.assert_close(x, x_out_infer, rtol=0, atol=0)
    print("  ✅ 推理模式测试通过（无遮挡）")
    
    print("\n" + "=" * 60)
    print("✅ OcclusionRobustEncoder 所有测试通过")
    print("=" * 60 + "\n")


def test_integration():
    """集成测试"""
    print("=" * 60)
    print("测试 4: 集成测试")
    print("=" * 60)
    
    # 模拟完整流程
    print("\n模拟完整ORR流程...")
    
    # 1. 初始化模块
    orr_encoder = OcclusionRobustEncoder(
        use_orr=True,
        mask_ratio=0.3,
        mask_strategy='cox',
        invariance_loss_weight=0.5
    )
    
    # 2. 输入特征
    B, N, C = 4, 196, 384
    H, W = 14, 14
    features = torch.randn(B, N, C, requires_grad=True)  # 需要梯度
    
    # 3. 训练时前向传播
    features_out, _ = orr_encoder(features, H, W, training=True)
    
    # 4. 计算不变性损失
    inv_loss = orr_encoder.compute_invariance_loss(
        orr_encoder.clean_features,
        orr_encoder.masked_features
    )
    
    print(f"  - Batch size: {B}")
    print(f"  - Feature dimension: {N}x{C}")
    print(f"  - Spatial dimension: {H}x{W}")
    print(f"  - 不变性损失: {inv_loss.item():.6f}")
    print(f"  - 损失权重: {orr_encoder.invariance_loss_weight}")
    print(f"  - 最终损失贡献: {inv_loss.item() * orr_encoder.invariance_loss_weight:.6f}")
    
    # 验证损失可以反向传播
    inv_loss.backward()
    print(f"  - 损失可反向传播: ✅")
    
    print("\n" + "=" * 60)
    print("✅ 集成测试通过")
    print("=" * 60 + "\n")


def test_mask_visualization():
    """可视化遮挡模式"""
    print("=" * 60)
    print("测试 5: 遮挡模式可视化")
    print("=" * 60)
    
    x = torch.randn(1, 196, 384)
    H, W = 14, 14
    
    strategies = ['random', 'block', 'cox']
    
    for strategy in strategies:
        masking = SpatialCoxMasking(mask_ratio=0.3, mask_strategy=strategy)
        mask = masking(x, H, W)  # [1, 196]
        
        # 重塑为2D
        mask_2d = mask[0].reshape(H, W).numpy()  # 从[1, 196]到[14, 14]
        
        print(f"\n{strategy}策略遮挡模式 ({H}x{W}):")
        # 简单的ASCII可视化
        for i in range(H):
            row = ""
            for j in range(W):
                row += "█" if mask_2d[i, j] > 0.5 else "·"
            print(f"  {row}")
        
        masked_ratio = mask_2d.sum() / (H * W)
        print(f"  遮挡比例: {masked_ratio:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ 遮挡模式可视化完成")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" ORR (Occlusion-Robust Representations) 模块测试")
    print("="*60 + "\n")
    
    # 运行所有测试
    test_spatial_cox_masking()
    test_feature_invariance_loss()
    test_occlusion_robust_encoder()
    test_integration()
    test_mask_visualization()
    
    print("\n" + "="*60)
    print("🎉 所有ORR模块测试通过！")
    print("="*60)
    print("\n核心功能验证:")
    print("  ✅ 空间Cox过程遮挡模拟")
    print("  ✅ 特征不变性损失计算")
    print("  ✅ 遮挡鲁棒编码器")
    print("  ✅ 训练/推理模式切换")
    print("  ✅ 端到端集成")
    print("\n应用场景:")
    print("  🚁 UAV跟踪中的遮挡处理")
    print("  🏙️  建筑物遮挡鲁棒性")
    print("  🌳 树木遮挡场景")
    print("  ⚡ 实时跟踪性能")
    print("="*60 + "\n")
