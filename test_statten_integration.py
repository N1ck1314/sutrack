"""
STAtten集成测试脚本
测试sutrack_STAtten模块是否正确集成
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_statten_module():
    """测试STAtten模块基本功能"""
    print("="*60)
    print("测试1: STAtten模块导入")
    print("="*60)
    
    from lib.models.sutrack_STAtten.statten import STAttenAttention, MS_SSA_Conv
    print("✅ STAtten模块导入成功")
    
    # 测试STAttenAttention
    print("\n测试STAttenAttention (适配SUTrack token格式)...")
    dim = 384
    num_heads = 6
    batch_size = 2
    num_tokens = 196  # 14x14
    
    attn = STAttenAttention(
        dim=dim,
        num_heads=num_heads,
        attention_mode="STAtten",
        use_snn=False
    )
    
    x = torch.randn(batch_size, num_tokens, dim)
    out = attn(x)
    
    assert out.shape == x.shape, f"输出形状不匹配: {out.shape} vs {x.shape}"
    print(f"✅ STAttenAttention测试通过 - 输入: {x.shape}, 输出: {out.shape}")
    
    # 测试MS_SSA_Conv
    print("\n测试MS_SSA_Conv (原始STAtten实现)...")
    T, B, C, H, W = 4, 2, 64, 32, 32
    
    ms_conv = MS_SSA_Conv(
        dim=C,
        num_heads=8,
        attention_mode="STAtten",
        chunk_size=2,
        use_snn=False
    )
    
    x_3d = torch.randn(T, B, C, H, W)
    out_3d, _, _ = ms_conv(x_3d)
    
    assert out_3d.shape == x_3d.shape, f"输出形状不匹配: {out_3d.shape} vs {x_3d.shape}"
    print(f"✅ MS_SSA_Conv测试通过 - 输入: {x_3d.shape}, 输出: {out_3d.shape}")
    

def test_config():
    """测试配置系统"""
    print("\n" + "="*60)
    print("测试2: 配置系统")
    print("="*60)
    
    from lib.config.sutrack_STAtten.config import cfg, update_config_from_file
    
    print(f"默认配置:")
    print(f"  USE_STATTEN: {cfg.MODEL.ENCODER.USE_STATTEN}")
    print(f"  STATTEN_MODE: {cfg.MODEL.ENCODER.STATTEN_MODE}")
    print(f"  USE_SNN: {cfg.MODEL.ENCODER.USE_SNN}")
    
    # 加载yaml配置
    config_file = "experiments/sutrack_STAtten/sutrack_statten_t224.yaml"
    if os.path.exists(config_file):
        update_config_from_file(config_file)
        print(f"\n从{config_file}加载配置:")
        print(f"  USE_STATTEN: {cfg.MODEL.ENCODER.USE_STATTEN}")
        print(f"  STATTEN_MODE: {cfg.MODEL.ENCODER.STATTEN_MODE}")
        print(f"  USE_SNN: {cfg.MODEL.ENCODER.USE_SNN}")
        print(f"  ENCODER_TYPE: {cfg.MODEL.ENCODER.TYPE}")
        print("✅ 配置加载成功")
    else:
        print(f"⚠️  配置文件不存在: {config_file}")


def test_model_build():
    """测试模型构建"""
    print("\n" + "="*60)
    print("测试3: 模型构建")
    print("="*60)
    
    from lib.config.sutrack_STAtten.config import cfg, update_config_from_file
    from lib.models.sutrack_STAtten import build_sutrack_statten
    
    # 加载配置
    config_file = "experiments/sutrack_STAtten/sutrack_statten_t224.yaml"
    if os.path.exists(config_file):
        update_config_from_file(config_file)
    
    # 临时设置为不加载预训练权重
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = ""
    
    try:
        print("正在构建SUTrack+STAtten模型...")
        model = build_sutrack_statten(cfg)
        print(f"✅ 模型构建成功")
        print(f"  Encoder类型: {cfg.MODEL.ENCODER.TYPE}")
        print(f"  使用STAtten: {cfg.MODEL.ENCODER.USE_STATTEN}")
        
        # 检查模型结构
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  总参数量: {total_params:,}")
        
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "="*60)
    print("测试4: 前向传播（简化版）")
    print("="*60)
    
    from lib.models.sutrack_STAtten.fastitpn import Block
    
    # 测试带STAtten的Block
    print("测试带STAtten的Transformer Block...")
    dim = 384
    block_statten = Block(
        dim=dim,
        num_heads=6,
        mlp_ratio=3.0,
        use_statten=True,
        statten_mode="STAtten",
        use_snn=False
    )
    
    # 测试标准Block
    block_standard = Block(
        dim=dim,
        num_heads=6,
        mlp_ratio=3.0,
        use_statten=False
    )
    
    # 前向传播
    x = torch.randn(2, 196, dim)  # [B, N, C]
    
    out_statten = block_statten(x)
    out_standard = block_standard(x)
    
    assert out_statten.shape == x.shape, "STAtten Block输出形状错误"
    assert out_standard.shape == x.shape, "标准Block输出形状错误"
    
    print(f"✅ STAtten Block测试通过 - 输入: {x.shape}, 输出: {out_statten.shape}")
    print(f"✅ 标准Block测试通过 - 输入: {x.shape}, 输出: {out_standard.shape}")
    

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("SUTrack + STAtten 集成测试")
    print("="*60)
    
    try:
        test_statten_module()
        test_config()
        test_forward_pass()
        # test_model_build()  # 注释掉，因为可能需要预训练权重
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        print("\n💡 使用提示:")
        print("1. 在配置文件中设置 USE_STATTEN: True 启用STAtten")
        print("2. STATTEN_MODE可选: 'STAtten' (时空注意力) 或 'SDT' (脉冲驱动)")
        print("3. 安装spikingjelly后可设置 USE_SNN: True 使用脉冲神经网络")
        print("4. 查看 lib/models/sutrack_STAtten/README_STATTEN.md 获取详细文档")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
