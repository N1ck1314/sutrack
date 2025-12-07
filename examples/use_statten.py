"""
STAtten使用示例
演示如何在SUTrack中使用STAtten注意力机制
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example1_basic_usage():
    """示例1: 基本使用 - 直接使用STAtten注意力模块"""
    print("="*60)
    print("示例1: 基本使用STAtten注意力模块")
    print("="*60)
    
    from lib.models.sutrack_STAtten.statten import STAttenAttention
    
    # 创建STAtten注意力层
    dim = 384  # 特征维度
    num_heads = 6  # 注意力头数
    
    statten = STAttenAttention(
        dim=dim,
        num_heads=num_heads,
        attention_mode="STAtten",  # 时空注意力模式
        use_snn=False  # 不使用脉冲神经网络
    )
    
    # 创建输入 [Batch, Tokens, Channels]
    batch_size = 2
    num_tokens = 196  # 14x14 patches
    x = torch.randn(batch_size, num_tokens, dim)
    
    # 前向传播
    output = statten(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ 基本使用成功\n")


def example2_with_config():
    """示例2: 使用配置文件构建完整模型"""
    print("="*60)
    print("示例2: 通过配置文件使用STAtten")
    print("="*60)
    
    from lib.config.sutrack_STAtten.config import cfg, update_config_from_file
    from lib.models.sutrack_STAtten import build_sutrack_statten
    
    # 加载配置
    config_file = "experiments/sutrack_STAtten/sutrack_statten_t224.yaml"
    update_config_from_file(config_file)
    
    print("配置参数:")
    print(f"  Encoder类型: {cfg.MODEL.ENCODER.TYPE}")
    print(f"  启用STAtten: {cfg.MODEL.ENCODER.USE_STATTEN}")
    print(f"  STAtten模式: {cfg.MODEL.ENCODER.STATTEN_MODE}")
    print(f"  使用SNN: {cfg.MODEL.ENCODER.USE_SNN}")
    
    # 注意：实际构建需要预训练权重，这里仅展示配置
    print("\n💡 提示：实际构建模型需要预训练权重")
    print("   使用: model = build_sutrack_statten(cfg)")
    print("✅ 配置加载成功\n")


def example3_compare_modes():
    """示例3: 比较不同注意力模式"""
    print("="*60)
    print("示例3: 比较STAtten和SDT模式")
    print("="*60)
    
    from lib.models.sutrack_STAtten.statten import STAttenAttention
    
    dim = 384
    num_heads = 6
    x = torch.randn(2, 196, dim)
    
    # STAtten模式（时空注意力）
    statten_mode = STAttenAttention(
        dim=dim,
        num_heads=num_heads,
        attention_mode="STAtten",
        use_snn=False
    )
    
    # SDT模式（脉冲驱动Transformer）
    sdt_mode = STAttenAttention(
        dim=dim,
        num_heads=num_heads,
        attention_mode="SDT",
        use_snn=False
    )
    
    # 标准注意力
    standard_mode = STAttenAttention(
        dim=dim,
        num_heads=num_heads,
        attention_mode="standard",
        use_snn=False
    )
    
    out_statten = statten_mode(x)
    out_sdt = sdt_mode(x)
    out_standard = standard_mode(x)
    
    print(f"输入形状: {x.shape}")
    print(f"STAtten输出: {out_statten.shape}")
    print(f"SDT输出: {out_sdt.shape}")
    print(f"标准注意力输出: {out_standard.shape}")
    print("\n说明:")
    print("  - STAtten: 时空注意力，适合视频序列")
    print("  - SDT: 脉冲驱动，计算量更小")
    print("  - Standard: 标准自注意力")
    print("✅ 模式比较完成\n")


def example4_transformer_block():
    """示例4: 在Transformer Block中使用STAtten"""
    print("="*60)
    print("示例4: Transformer Block集成STAtten")
    print("="*60)
    
    from lib.models.sutrack_STAtten.fastitpn import Block
    
    dim = 384
    
    # 创建带STAtten的Block
    block = Block(
        dim=dim,
        num_heads=6,
        mlp_ratio=3.0,
        use_statten=True,  # 启用STAtten
        statten_mode="STAtten",
        use_snn=False
    )
    
    # 测试
    x = torch.randn(2, 196, dim)
    output = block(x)
    
    print(f"Block输入: {x.shape}")
    print(f"Block输出: {output.shape}")
    print("✅ Transformer Block测试成功\n")


def example5_custom_config():
    """示例5: 自定义配置"""
    print("="*60)
    print("示例5: 自定义STAtten配置")
    print("="*60)
    
    from lib.config.sutrack_STAtten.config import cfg
    
    # 自定义配置
    cfg.MODEL.ENCODER.USE_STATTEN = True
    cfg.MODEL.ENCODER.STATTEN_MODE = "STAtten"
    cfg.MODEL.ENCODER.USE_SNN = False
    cfg.MODEL.ENCODER.TYPE = "fastitpnt"
    
    print("自定义配置:")
    print(f"  USE_STATTEN: {cfg.MODEL.ENCODER.USE_STATTEN}")
    print(f"  STATTEN_MODE: {cfg.MODEL.ENCODER.STATTEN_MODE}")
    print(f"  USE_SNN: {cfg.MODEL.ENCODER.USE_SNN}")
    print(f"  ENCODER_TYPE: {cfg.MODEL.ENCODER.TYPE}")
    
    print("\n💡 配置建议:")
    print("  1. 初次使用建议 USE_SNN=False")
    print("  2. STATTEN_MODE='STAtten' 适合视频跟踪")
    print("  3. 需要高性能可尝试 USE_SNN=True（需安装spikingjelly）")
    print("✅ 自定义配置完成\n")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("STAtten 使用示例集")
    print("="*60 + "\n")
    
    example1_basic_usage()
    example2_with_config()
    example3_compare_modes()
    example4_transformer_block()
    example5_custom_config()
    
    print("="*60)
    print("📚 更多信息请查看:")
    print("  - README: lib/models/sutrack_STAtten/README_STATTEN.md")
    print("  - 论文: https://arxiv.org/pdf/2409.19764")
    print("  - 代码: https://github.com/Intelligent-Computing-Lab-Panda/STAtten")
    print("="*60)


if __name__ == "__main__":
    main()
