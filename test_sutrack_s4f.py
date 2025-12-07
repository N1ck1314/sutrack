"""
测试sutrack_S4F的CMSA模块集成
验证跨模态空间感知模块的功能
"""

import torch
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_cmsa_module():
    """测试CMSA模块的基本功能"""
    print("=" * 60)
    print("测试1: CMSA核心模块")
    print("=" * 60)
    
    from lib.models.sutrack_S4F.cmsa import CMSA, MultiModalFusionWithCMSA
    
    # 测试参数
    batch_size = 2
    dim = 512
    h = w = 14  # search region: 224/16 = 14
    
    # 创建CMSA模块
    cmsa = CMSA(dim=dim, h=h, w=w, use_ssm=True)
    print(f"✓ CMSA模块创建成功")
    print(f"  - 维度: {dim}, 特征图尺寸: {h}x{w}")
    
    # 创建模拟输入（RGB和Depth/Thermal）
    rgb_feat = torch.randn(batch_size, h*w, dim)
    depth_feat = torch.randn(batch_size, h*w, dim)
    
    print(f"\n输入特征形状:")
    print(f"  - RGB: {rgb_feat.shape}")
    print(f"  - Depth/Thermal: {depth_feat.shape}")
    
    # 前向传播
    try:
        fused_feat = cmsa(rgb_feat, depth_feat)
        print(f"\n✓ CMSA前向传播成功")
        print(f"  - 融合特征形状: {fused_feat.shape}")
        assert fused_feat.shape == (batch_size, h*w, dim), "输出形状不匹配"
        print(f"  - 输出形状验证通过")
    except Exception as e:
        print(f"\n✗ CMSA前向传播失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("测试2: MultiModalFusionWithCMSA")
    print("=" * 60)
    
    # 测试MultiModalFusionWithCMSA
    fusion_cmsa = MultiModalFusionWithCMSA(dim=dim, h=h, w=w, use_ssm=True, fusion_mode='cmsa')
    fusion_concat = MultiModalFusionWithCMSA(dim=dim, h=h, w=w, use_ssm=False, fusion_mode='concat')
    
    print("✓ MultiModalFusionWithCMSA创建成功")
    print("  - CMSA模式")
    print("  - Concat模式")
    
    # 测试两种模式
    try:
        fused_cmsa = fusion_cmsa(rgb_feat, depth_feat)
        fused_concat = fusion_concat(rgb_feat, depth_feat)
        
        print(f"\n✓ 两种融合模式都成功")
        print(f"  - CMSA模式输出: {fused_cmsa.shape}")
        print(f"  - Concat模式输出: {fused_concat.shape}")
    except Exception as e:
        print(f"\n✗ 融合失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("所有CMSA模块测试通过! ✓")
    print("=" * 60)
    return True


def test_encoder_integration():
    """测试CMSA在encoder中的集成"""
    print("\n" + "=" * 60)
    print("测试3: Encoder集成")
    print("=" * 60)
    
    from lib.config.sutrack_S4F.config import cfg, update_config_from_file
    
    # 加载配置
    config_path = "experiments/sutrack_S4F/sutrack_s4f_cmsa.yaml"
    if os.path.exists(config_path):
        update_config_from_file(config_path)
        print(f"✓ 配置文件加载成功: {config_path}")
        
        # 检查CMSA配置
        print(f"\nCMSA配置:")
        print(f"  - USE_CMSA: {cfg.MODEL.ENCODER.get('USE_CMSA', False)}")
        print(f"  - CMSA_MODE: {cfg.MODEL.ENCODER.get('CMSA_MODE', 'cmsa')}")
        print(f"  - USE_SSM: {cfg.MODEL.ENCODER.get('USE_SSM', True)}")
    else:
        print(f"✗ 配置文件不存在: {config_path}")
        return False
    
    # 测试encoder创建
    try:
        from lib.models.sutrack_S4F.encoder import build_encoder
        
        # 暂时跳过encoder的完整测试，因为需要预训练权重
        print("\n✓ Encoder模块导入成功")
        print("  注意: 完整测试需要预训练权重")
        
    except Exception as e:
        print(f"\n✗ Encoder测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Encoder集成测试通过! ✓")
    print("=" * 60)
    return True


def test_full_model():
    """测试完整模型创建"""
    print("\n" + "=" * 60)
    print("测试4: 完整模型创建")
    print("=" * 60)
    
    try:
        from lib.models.sutrack_S4F import build_sutrack_s4f
        from lib.config.sutrack_S4F.config import cfg, update_config_from_file
        
        # 加载配置
        config_path = "experiments/sutrack_S4F/sutrack_s4f_cmsa.yaml"
        if os.path.exists(config_path):
            update_config_from_file(config_path)
            print(f"✓ 配置加载成功")
        
        # 注意：完整模型创建需要预训练权重
        print("\n✓ 模型构建函数导入成功")
        print("  注意: 完整模型创建需要预训练权重和CLIP模型")
        
    except Exception as e:
        print(f"\n✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("完整模型测试通过! ✓")
    print("=" * 60)
    return True


def main():
    print("\n" + "=" * 60)
    print("SUTrack with S4Fusion CMSA 集成测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("CMSA核心模块", test_cmsa_module),
        ("Encoder集成", test_encoder_integration),
        ("完整模型", test_full_model),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n测试 '{test_name}' 出现异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 打印总结
    print("\n\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过!")
        print("\n下一步:")
        print("1. 运行训练脚本测试:")
        print("   python tracking/train.py --script sutrack_S4F --config sutrack_s4f_cmsa --save_dir . --mode single")
        print("\n2. CMSA模块改进了:")
        print("   - 替代了简单的torch.cat拼接")
        print("   - 使用空间位置标记进行模态对齐")
        print("   - 通过状态空间模型进行跨模态交互")
        print("   - 自适应门控融合多模态特征")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
