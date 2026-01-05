"""
SUTrack-MLKA 快速使用示例
演示如何在SUTrack中使用MLKA模块
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def example1_basic_mlka():
    """示例1: 基本MLKA模块使用"""
    print("="*60)
    print("示例1: 基本MLKA模块")
    print("="*60)
    
    try:
        import torch
        from lib.models.sutrack_MLKA.mlka import MLKA
        
        # 创建MLKA模块
        n_feats = 384  # 特征维度（必须是3的倍数）
        mlka = MLKA(n_feats, use_norm=True)
        
        # 创建输入 (Batch, Channels, Height, Width)
        x = torch.randn(2, n_feats, 16, 16)
        
        # 前向传播
        y = mlka(x)
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {y.shape}")
        print(f"参数量: {sum(p.numel() for p in mlka.parameters()) / 1e6:.2f}M")
        print("✅ MLKA模块测试成功\n")
        
    except ImportError as e:
        print(f"⚠️  需要安装PyTorch: {e}")
        print("请运行: pip install torch\n")


def example2_mlka_block():
    """示例2: 完整的MLKA Block (含FFN)"""
    print("="*60)
    print("示例2: MLKA Block (MLKA + FFN)")
    print("="*60)
    
    try:
        import torch
        from lib.models.sutrack_MLKA.mlka import MLKABlock
        
        dim = 512  # ViT-B 的特征维度
        block = MLKABlock(dim, mlp_ratio=3.0)
        
        x = torch.randn(2, dim, 18, 18)
        y = block(x)
        
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {y.shape}")
        print(f"参数量: {sum(p.numel() for p in block.parameters()) / 1e6:.2f}M")
        print("✅ MLKA Block测试成功\n")
        
    except ImportError as e:
        print(f"⚠️  需要安装PyTorch: {e}\n")


def example3_feature_enhancement():
    """示例3: 用于SUTrack decoder的特征增强"""
    print("="*60)
    print("示例3: MLKA Feature Enhancement for Decoder")
    print("="*60)
    
    try:
        import torch
        from lib.models.sutrack_MLKA.mlka import MLKAFeatureEnhancement
        
        # 创建增强模块（可用于decoder前）
        enhancer = MLKAFeatureEnhancement(dim=512, num_blocks=2)
        
        # 模拟搜索区域特征
        search_feat = torch.randn(2, 512, 18, 18)
        enhanced_feat = enhancer(search_feat)
        
        print(f"原始特征: {search_feat.shape}")
        print(f"增强特征: {enhanced_feat.shape}")
        print(f"参数量: {sum(p.numel() for p in enhancer.parameters()) / 1e6:.2f}M")
        print("✅ 特征增强测试成功\n")
        
    except ImportError as e:
        print(f"⚠️  需要安装PyTorch: {e}\n")


def example4_build_model():
    """示例4: 构建完整的SUTrack-MLKA模型"""
    print("="*60)
    print("示例4: 构建SUTrack-MLKA模型")
    print("="*60)
    
    try:
        from lib.config.sutrack_MLKA.config import cfg
        from lib.models.sutrack_MLKA import build_sutrack_mlka
        
        # 配置MLKA参数
        cfg.MODEL.USE_MLKA = True
        cfg.MODEL.MLKA_POSITION = "decoder"  # "encoder", "decoder", "both"
        cfg.MODEL.MLKA_BLOCKS = 1
        
        print("配置参数:")
        print(f"  USE_MLKA: {cfg.MODEL.USE_MLKA}")
        print(f"  MLKA_POSITION: {cfg.MODEL.MLKA_POSITION}")
        print(f"  MLKA_BLOCKS: {cfg.MODEL.MLKA_BLOCKS}")
        print(f"  ENCODER_TYPE: {cfg.MODEL.ENCODER.TYPE}")
        print(f"  DECODER_TYPE: {cfg.MODEL.DECODER.TYPE}")
        
        print("\n💡 提示: 构建完整模型需要预训练权重")
        print("   使用: model = build_sutrack_mlka(cfg)")
        print("✅ 配置加载成功\n")
        
    except Exception as e:
        print(f"⚠️  配置加载失败: {e}\n")


def example5_config_variations():
    """示例5: 不同配置对比"""
    print("="*60)
    print("示例5: MLKA配置对比")
    print("="*60)
    
    configs = [
        ("decoder + 1 block", "decoder", 1, "推荐：平衡性能与效率"),
        ("encoder + 1 block", "encoder", 1, "提升整体特征表达"),
        ("both + 1 block", "both", 1, "最强效果，计算量增加"),
        ("decoder + 2 blocks", "decoder", 2, "更强的定位能力"),
    ]
    
    print("\n配置方案对比:")
    print("-" * 80)
    print(f"{'配置':<20} | {'位置':<10} | {'块数':<5} | {'说明':<30}")
    print("-" * 80)
    
    for name, pos, blocks, desc in configs:
        print(f"{name:<20} | {pos:<10} | {blocks:<5} | {desc:<30}")
    
    print("-" * 80)
    print("\n推荐配置:")
    print("  - 快速原型: decoder + 1 block")
    print("  - 复杂场景: encoder + 1 block")
    print("  - 最佳性能: both + 1 block (资源充足)")
    print("✅ 配置说明完成\n")


def example6_usage_guide():
    """示例6: 使用指南"""
    print("="*60)
    print("示例6: SUTrack-MLKA使用指南")
    print("="*60)
    
    print("\n📝 步骤1: 配置模型")
    print("---")
    print("from lib.config.sutrack_MLKA.config import cfg")
    print("cfg.MODEL.USE_MLKA = True")
    print("cfg.MODEL.MLKA_POSITION = 'decoder'")
    print("cfg.MODEL.MLKA_BLOCKS = 1")
    
    print("\n📝 步骤2: 构建模型")
    print("---")
    print("from lib.models.sutrack_MLKA import build_sutrack_mlka")
    print("model = build_sutrack_mlka(cfg)")
    
    print("\n📝 步骤3: 训练")
    print("---")
    print("python tracking/train.py \\")
    print("    --config experiments/sutrack_MLKA/config.yaml \\")
    print("    --output checkpoints/sutrack_mlka")
    
    print("\n📝 步骤4: 测试")
    print("---")
    print("python tracking/test.py \\")
    print("    --tracker_name sutrack_mlka \\")
    print("    --dataset depthtrack")
    
    print("\n✅ 使用指南完成\n")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("SUTrack-MLKA 使用示例集")
    print("="*60 + "\n")
    
    example1_basic_mlka()
    example2_mlka_block()
    example3_feature_enhancement()
    example4_build_model()
    example5_config_variations()
    example6_usage_guide()
    
    print("="*60)
    print("📚 更多信息请查看:")
    print("  - README: lib/models/sutrack_MLKA/README.md")
    print("  - MLKA论文: https://arxiv.org/abs/2209.14145")
    print("  - SUTrack: lib/models/sutrack/")
    print("="*60)


if __name__ == "__main__":
    main()
