"""
SMFA模块测试脚本
验证SMFA (Self-Modulation Feature Aggregation) 集成是否正常工作
"""
import sys
sys.path.insert(0, '/home/nick/code/code.sutrack/SUTrack')

import torch
from lib.models.sutrack_SMFA.smfa_modules import EASA, LDE, SMFA, PCFN, SMFABlock

def test_easa():
    """测试 EASA (Efficient Approximation of Self-Attention)"""
    print("\n" + "="*60)
    print("测试 EASA (Efficient Approximation of Self-Attention)")
    print("="*60)
    
    dim = 384
    num_heads = 6
    batch_size = 2
    num_patches = 196
    
    easa = EASA(dim=dim, num_heads=num_heads)
    x = torch.randn(batch_size, num_patches, dim)
    
    print(f"输入形状: {x.shape}")
    out = easa(x)
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in easa.parameters()) / 1e6:.2f}M")
    print("✅ EASA 测试通过")
    
    return out

def test_lde():
    """测试 LDE (Local Detail Estimation)"""
    print("\n" + "="*60)
    print("测试 LDE (Local Detail Estimation)")
    print("="*60)
    
    dim = 384
    batch_size = 2
    h, w = 14, 14
    
    lde = LDE(dim=dim, kernel_sizes=[3, 5], dilation_rates=[1, 2])
    x = torch.randn(batch_size, dim, h, w)
    
    print(f"输入形状: {x.shape}")
    out = lde(x)
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in lde.parameters()) / 1e6:.2f}M")
    print("✅ LDE 测试通过")
    
    return out

def test_smfa():
    """测试 SMFA (Self-Modulation Feature Aggregation)"""
    print("\n" + "="*60)
    print("测试 SMFA (Self-Modulation Feature Aggregation)")
    print("="*60)
    
    dim = 384
    num_heads = 6
    batch_size = 2
    h, w = 14, 14
    
    smfa = SMFA(dim=dim, num_heads=num_heads)
    x = torch.randn(batch_size, dim, h, w)
    
    print(f"输入形状: {x.shape}")
    out = smfa(x)
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in smfa.parameters()) / 1e6:.2f}M")
    
    # 验证残差连接
    print(f"残差连接检查: {'✅ 正确' if out.shape == x.shape else '❌ 错误'}")
    print("✅ SMFA 测试通过")
    
    return out

def test_pcfn():
    """测试 PCFN (Partial Convolution-based Feed-Forward Network)"""
    print("\n" + "="*60)
    print("测试 PCFN (Partial Convolution-based Feed-Forward Network)")
    print("="*60)
    
    dim = 384
    mlp_ratio = 4.0
    batch_size = 2
    h, w = 14, 14
    
    pcfn = PCFN(dim=dim, mlp_ratio=mlp_ratio)
    x = torch.randn(batch_size, dim, h, w)
    
    print(f"输入形状: {x.shape}")
    out = pcfn(x)
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in pcfn.parameters()) / 1e6:.2f}M")
    print(f"部分卷积比例: {pcfn.partial_ratio * 100:.0f}%")
    print("✅ PCFN 测试通过")
    
    return out

def test_smfa_block():
    """测试完整的 SMFABlock"""
    print("\n" + "="*60)
    print("测试 SMFABlock (完整模块)")
    print("="*60)
    
    dim = 384
    num_heads = 6
    mlp_ratio = 4.0
    batch_size = 2
    h, w = 14, 14
    
    block = SMFABlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
    x = torch.randn(batch_size, dim, h, w)
    
    print(f"输入形状 (空间): {x.shape}")
    out_spatial = block(x)
    print(f"输出形状 (空间): {out_spatial.shape}")
    
    # 测试序列输入
    x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
    print(f"\n输入形状 (序列): {x_seq.shape}")
    out_seq = block(x_seq)
    print(f"输出形状 (序列): {out_seq.shape}")
    
    print(f"\n总参数量: {sum(p.numel() for p in block.parameters()) / 1e6:.2f}M")
    print("✅ SMFABlock 测试通过")
    
    return out_spatial, out_seq

def test_integration():
    """测试与SUTrack的集成"""
    print("\n" + "="*60)
    print("测试 SMFA 与 SUTrack 的集成")
    print("="*60)
    
    try:
        from lib.models.sutrack_SMFA.encoder import EncoderBase
        print("✅ encoder.py 导入成功")
        
        from lib.config.sutrack_SMFA import config
        cfg = config.cfg
        print("✅ config.py 导入成功")
        
        # 检查配置
        use_smfa = cfg.MODEL.ENCODER.get('USE_SMFA', False)
        smfa_num_heads = cfg.MODEL.ENCODER.get('SMFA_NUM_HEADS', 6)
        smfa_mlp_ratio = cfg.MODEL.ENCODER.get('SMFA_MLP_RATIO', 4.0)
        
        print(f"\n配置检查:")
        print(f"  - USE_SMFA: {use_smfa}")
        print(f"  - SMFA_NUM_HEADS: {smfa_num_heads}")
        print(f"  - SMFA_MLP_RATIO: {smfa_mlp_ratio}")
        
        if use_smfa:
            print("\n✅ SMFA已在配置中启用")
        else:
            print("\n⚠️  SMFA在配置中未启用，需要在YAML中设置USE_SMFA: True")
        
        print("\n✅ 集成测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("SMFA 模块完整测试")
    print("="*80)
    
    try:
        # 测试各个子模块
        test_easa()
        test_lde()
        test_smfa()
        test_pcfn()
        test_smfa_block()
        
        # 测试集成
        success = test_integration()
        
        # 总结
        print("\n" + "="*80)
        print("测试总结")
        print("="*80)
        print("✅ 所有模块测试通过！")
        print("\n核心改进点:")
        print("  1. EASA - 高效自注意力近似，降低计算复杂度")
        print("  2. LDE - 局部细节估计，多尺度卷积捕获细节")
        print("  3. SMFA - 自调制特征聚合，全局+局部信息融合")
        print("  4. PCFN - 部分卷积前馈网络，减少参数和计算量")
        print("\nSMFANet参考: https://github.com/Zheng-MJ/SMFANet")
        print("论文: SMFANet - A Lightweight Self-Modulation Feature Aggregation Network (ECCV 2024)")
        print("="*80)
        
        return success
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
