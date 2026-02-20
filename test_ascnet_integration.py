"""
测试ASCNet模块集成
验证RHDWT和CNCM模块是否正确工作
"""

import torch
import sys
sys.path.append('.')

from lib.models.sutrack_ascn.ascnet_modules import RHDWT, CNCM, RCSSC, HDWT, CAB, SAB, SCB


def test_hdwt():
    """测试Haar离散小波变换"""
    print("="*60)
    print("测试 HDWT (Haar Discrete Wavelet Transform)")
    print("="*60)
    
    hdwt = HDWT()
    x = torch.randn(2, 64, 32, 32)  # [B, C, H, W]
    
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        out = hdwt(x)
    
    print(f"输出形状: {out.shape}")
    print(f"预期形状: [2, 256, 16, 16] (4*C channels, H/2, W/2)")
    
    assert out.shape == (2, 256, 16, 16), f"HDWT输出形状错误: {out.shape}"
    print("✓ HDWT测试通过\n")


def test_rhdwt():
    """测试残差哈尔离散小波变换"""
    print("="*60)
    print("测试 RHDWT (Residual Haar DWT)")
    print("="*60)
    
    rhdwt = RHDWT(in_channels=64, out_channels=128)
    x = torch.randn(2, 64, 32, 32)
    
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        out = rhdwt(x)
    
    print(f"输出形状: {out.shape}")
    print(f"预期形状: [2, 128, 16, 16] (下采样2x)")
    print(f"特点: 融合了小波方向先验和卷积语义特征")
    
    assert out.shape == (2, 128, 16, 16), f"RHDWT输出形状错误: {out.shape}"
    print("✓ RHDWT测试通过\n")


def test_cab():
    """测试列注意力分支"""
    print("="*60)
    print("测试 CAB (Column Attention Branch)")
    print("="*60)
    
    cab = CAB(channels=64)
    x = torch.randn(2, 64, 16, 16)
    
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        out = cab(x)
    
    print(f"输出形状: {out.shape}")
    print(f"特点: 使用列池化强化列特征")
    
    assert out.shape == x.shape, f"CAB输出形状错误: {out.shape}"
    print("✓ CAB测试通过\n")


def test_sab():
    """测试空间注意力分支"""
    print("="*60)
    print("测试 SAB (Spatial Attention Branch)")
    print("="*60)
    
    sab = SAB(channels=64)
    x = torch.randn(2, 64, 16, 16)
    
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        out = sab(x)
    
    print(f"输出形状: {out.shape}")
    print(f"特点: 增强关键区域的空间相关性")
    
    assert out.shape == x.shape, f"SAB输出形状错误: {out.shape}"
    print("✓ SAB测试通过\n")


def test_scb():
    """测试自校准分支"""
    print("="*60)
    print("测试 SCB (Self-Calibrated Branch)")
    print("="*60)
    
    scb = SCB(channels=64)
    x = torch.randn(2, 64, 16, 16)
    
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        out = scb(x)
    
    print(f"输出形状: {out.shape}")
    print(f"特点: 建立长程依赖，全局上下文聚合")
    
    assert out.shape == x.shape, f"SCB输出形状错误: {out.shape}"
    print("✓ SCB测试通过\n")


def test_rcssc():
    """测试残差列空间自校正块"""
    print("="*60)
    print("测试 RCSSC (Residual Column Spatial Self-Correction)")
    print("="*60)
    
    rcssc = RCSSC(channels=64)
    x = torch.randn(2, 64, 16, 16)
    
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        out = rcssc(x)
    
    print(f"输出形状: {out.shape}")
    print(f"特点: 融合CAB、SAB、SCB三个分支")
    
    assert out.shape == x.shape, f"RCSSC输出形状错误: {out.shape}"
    print("✓ RCSSC测试通过\n")


def test_cncm():
    """测试列非均匀性校正模块"""
    print("="*60)
    print("测试 CNCM (Column Non-uniformity Correction Module)")
    print("="*60)
    
    cncm = CNCM(channels=64, num_blocks=3)
    x = torch.randn(2, 64, 16, 16)
    
    print(f"输入形状: {x.shape}")
    print(f"RCSSC块数量: 3")
    
    with torch.no_grad():
        out = cncm(x)
    
    print(f"输出形状: {out.shape}")
    print(f"特点: 密集连接结构，多级特征融合")
    
    assert out.shape == x.shape, f"CNCM输出形状错误: {out.shape}"
    print("✓ CNCM测试通过\n")


def test_full_pipeline():
    """测试完整的RHDWT+CNCM流水线"""
    print("="*60)
    print("测试完整流水线: RHDWT -> CNCM")
    print("="*60)
    
    # 构建流水线
    rhdwt = RHDWT(in_channels=64, out_channels=128)
    cncm = CNCM(channels=128, num_blocks=3)
    
    x = torch.randn(2, 64, 32, 32)
    
    print(f"输入形状: {x.shape}")
    
    with torch.no_grad():
        # 下采样
        x_down = rhdwt(x)
        print(f"RHDWT输出: {x_down.shape}")
        
        # 特征增强
        x_enhanced = cncm(x_down)
        print(f"CNCM输出: {x_enhanced.shape}")
    
    assert x_down.shape == (2, 128, 16, 16), f"下采样形状错误"
    assert x_enhanced.shape == (2, 128, 16, 16), f"增强后形状错误"
    
    print("✓ 完整流水线测试通过\n")


def main():
    print("\n" + "="*60)
    print("开始测试ASCNet模块")
    print("="*60 + "\n")
    
    # 测试基础模块
    test_hdwt()
    test_rhdwt()
    test_cab()
    test_sab()
    test_scb()
    test_rcssc()
    test_cncm()
    
    # 测试完整流水线
    test_full_pipeline()
    
    print("="*60)
    print("✅ 所有ASCNet模块测试通过！")
    print("="*60)
    print("\n核心模块总结:")
    print("1. RHDWT - 残差哈尔小波变换（下采样）")
    print("   - 模型驱动分支: Haar小波捕获方向先验")
    print("   - 残差分支: 步进卷积捕获数据语义")
    print("   - 双分支融合: 获得更丰富的特征表征")
    print()
    print("2. CNCM - 列非均匀性校正模块（特征增强）")
    print("   - CAB: 列注意力（列池化+双重校正）")
    print("   - SAB: 空间注意力（关键区域增强）")
    print("   - SCB: 自校准（长程依赖建模）")
    print("   - RCSSC: 残差融合三个分支")
    print("   - 密集连接: 多级特征重用")
    print()
    print("应用场景: 条纹噪声抑制、传感器非均匀性校正")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
