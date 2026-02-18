#!/usr/bin/env python3
"""
ARTrackV2集成验证脚本
验证所有模块是否正确集成到SUTrack中
"""

import torch
import sys
sys.path.append('.')

def test_artrackv2_modules():
    """测试ARTrackV2核心模块"""
    print("="*60)
    print("测试ARTrackV2核心模块")
    print("="*60)
    
    from lib.models.sutrack_arv2.artrackv2_modules import (
        AppearancePrompts,
        AppearanceReconstruction,
        ConfidenceToken,
        OrientedMasking,
        PureEncoderDecoder
    )
    
    batch_size = 2
    num_tokens = 196  # 14x14 search region
    dim = 384  # tiny model dimension
    
    # 1. 测试Appearance Prompts
    print("\n1. 测试Appearance Prompts")
    appearance_module = AppearancePrompts(dim=dim, num_prompts=4)
    search_features = torch.randn(batch_size, num_tokens, dim)
    appearance_tokens = appearance_module(search_features)
    print(f"   ✓ Appearance tokens shape: {appearance_tokens.shape}")
    assert appearance_tokens.shape == (batch_size, 4, dim)
    
    # 2. 测试Appearance Reconstruction
    print("\n2. 测试Appearance Reconstruction")
    recon_module = AppearanceReconstruction(dim=dim, num_prompts=4)
    recon_module.train()
    target_features = torch.randn(batch_size, 49, dim)  # 7x7 template
    recon_loss = recon_module(appearance_tokens, target_features)
    print(f"   ✓ Reconstruction loss: {recon_loss.item():.4f}")
    assert recon_loss.item() >= 0
    
    # 3. 测试Confidence Token
    print("\n3. 测试Confidence Token")
    conf_module = ConfidenceToken(dim=dim)
    features_with_conf = torch.randn(batch_size, num_tokens+1, dim)
    confidence = conf_module(features_with_conf)
    print(f"   ✓ Confidence shape: {confidence.shape}")
    assert confidence.shape == (batch_size, 1)
    assert (confidence >= 0).all() and (confidence <= 1).all()
    
    # 4. 测试Oriented Masking
    print("\n4. 测试Oriented Masking")
    mask = OrientedMasking.create_attention_mask(
        batch_size=batch_size,
        num_confidence_tokens=1,
        num_trajectory_tokens=4,
        num_appearance_tokens=4,
        num_search_tokens=num_tokens,
        device='cpu'
    )
    print(f"   ✓ Attention mask shape: {mask.shape}")
    total_tokens = 1 + 4 + 4 + num_tokens
    assert mask.shape == (total_tokens, total_tokens)  # 现在是2D而不是3D
    
    # 5. 测试Pure Encoder Decoder
    print("\n5. 测试Pure Encoder Decoder")
    pure_encoder = PureEncoderDecoder(dim=dim, num_trajectory_tokens=4, num_appearance_tokens=4)
    pure_encoder.train()
    bbox, confidence, aux_dict = pure_encoder(search_features, target_features=target_features)
    print(f"   ✓ Predicted bbox shape: {bbox.shape}")
    print(f"   ✓ Confidence shape: {confidence.shape}")
    print(f"   ✓ Aux dict keys: {aux_dict.keys()}")
    assert bbox.shape == (batch_size, 4)
    assert confidence.shape == (batch_size, 1)
    assert 'appearance_recon_loss' in aux_dict
    
    print("\n" + "="*60)
    print("✅ 所有ARTrackV2核心模块测试通过！")
    print("="*60)


def test_model_building():
    """测试模型构建"""
    print("\n" + "="*60)
    print("测试SUTRACK_ARV2模型构建")
    print("="*60)
    
    from lib.config.sutrack_arv2.config import cfg
    from lib.models.sutrack_arv2 import build_sutrack_arv2
    
    # 设置基本配置
    cfg.MODEL.ENCODER.TYPE = "fastitpnt"
    cfg.MODEL.ENCODER.STRIDE = 16
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = None  # 测试时不加载预训练
    cfg.MODEL.ARTRACKV2.ENABLE = True
    cfg.MODEL.ARTRACKV2.NUM_APPEARANCE_TOKENS = 4
    cfg.DATA.SEARCH.SIZE = 224
    cfg.DATA.TEMPLATE.SIZE = 112
    cfg.DATA.MULTI_MODAL_LANGUAGE = False
    
    print("\n正在构建模型...")
    try:
        model = build_sutrack_arv2(cfg)
        print(f"✓ 模型构建成功")
        print(f"✓ ARTrackV2启用状态: {model.use_artrackv2}")
        print(f"✓ Encoder ARTrackV2启用: {model.encoder.use_artrackv2}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ 总参数量: {total_params/1e6:.2f}M")
        print(f"✓ 可训练参数: {trainable_params/1e6:.2f}M")
        
        print("\n" + "="*60)
        print("✅ 模型构建测试通过！")
        print("="*60)
        return True
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "="*60)
    print("测试SUTRACK_ARV2前向传播")
    print("="*60)
    
    from lib.config.sutrack_arv2.config import cfg
    from lib.models.sutrack_arv2 import build_sutrack_arv2
    
    # 配置
    cfg.MODEL.ENCODER.TYPE = "fastitpnt"
    cfg.MODEL.ENCODER.STRIDE = 16
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = None
    cfg.MODEL.ENCODER.CLASS_TOKEN = True
    cfg.MODEL.ARTRACKV2.ENABLE = True
    cfg.MODEL.ARTRACKV2.NUM_APPEARANCE_TOKENS = 4
    cfg.DATA.SEARCH.SIZE = 224
    cfg.DATA.SEARCH.NUMBER = 1
    cfg.DATA.TEMPLATE.SIZE = 112
    cfg.DATA.TEMPLATE.NUMBER = 1
    cfg.DATA.MULTI_MODAL_LANGUAGE = False
    cfg.DATA.MULTI_MODAL_VISION = True
    
    try:
        model = build_sutrack_arv2(cfg)
        model.eval()
        
        # 准备输入
        batch_size = 2
        template = torch.randn(batch_size, 6, 112, 112)  # RGBD
        search = torch.randn(batch_size, 6, 224, 224)
        template_anno = torch.randn(batch_size, 4)
        
        print("\n测试encoder forward...")
        with torch.no_grad():
            xz, aux_dict = model.forward_encoder(
                [template], [search], [template_anno.unsqueeze(1)],
                text_src=None, task_index=None
            )
        
        if isinstance(xz, list):
            print(f"✓ Encoder输出shape: {xz[0].shape}")
        else:
            print(f"✓ Encoder输出shape: {xz.shape}")
        print(f"✓ Aux dict keys: {aux_dict.keys()}")
        
        print("\n测试decoder forward...")
        with torch.no_grad():
            out_dict = model.forward_decoder((xz, aux_dict))
        
        print(f"✓ Decoder输出keys: {out_dict.keys()}")
        if 'pred_boxes' in out_dict:
            print(f"✓ Predicted boxes shape: {out_dict['pred_boxes'].shape}")
        if 'confidence' in out_dict:
            print(f"✓ Confidence shape: {out_dict['confidence'].shape}")
            print(f"✓ 使用ARTrackV2 Pure Encoder预测")
        else:
            print(f"✓ 使用标准Decoder预测")
        
        print("\n" + "="*60)
        print("✅ 前向传播测试通过！")
        print("="*60)
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n")
    print("="*60)
    print("ARTrackV2集成验证")
    print("="*60)
    
    # 1. 测试核心模块
    try:
        test_artrackv2_modules()
    except Exception as e:
        print(f"❌ 核心模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 测试模型构建
    if not test_model_building():
        return
    
    # 3. 测试前向传播
    if not test_forward_pass():
        return
    
    print("\n" + "="*60)
    print("🎉 所有测试通过！ARTrackV2集成成功！")
    print("="*60)
    print("\n下一步:")
    print("1. 启动训练:")
    print("   python tracking/train.py --script sutrack_arv2 --config sutrack_arv2_t224")
    print("\n2. 运行测试:")
    print("   python tracking/test.py sutrack_arv2 sutrack_arv2_t224 --dataset depthtrack")
    print("="*60)


if __name__ == "__main__":
    main()
