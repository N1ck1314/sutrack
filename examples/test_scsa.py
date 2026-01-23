"""
测试SCSA模块的脚本
验证SCSA模块能否正常工作
"""
import sys
sys.path.append('/home/nick/code/code.sutrack/SUTrack')

import torch
from lib.models.sutrack_SCSA.scsa_modules import (
    Shareable_Multi_Semantic_Spatial_Attention,
    Progressive_Channel_wise_Self_Attention,
    SCSA
)

def test_smsa():
    """测试SMSA模块"""
    print("=" * 60)
    print("测试 SMSA (Shareable Multi-Semantic Spatial Attention)")
    print("=" * 60)
    
    # 创建SMSA模块
    dim = 512  # Base模型的通道数
    smsa = Shareable_Multi_Semantic_Spatial_Attention(
        dim=dim,
        group_kernel_sizes=[3, 5, 7, 9],
        gate_layer='sigmoid'
    )
    
    # 创建测试输入 (B, C, H, W)
    batch_size = 2
    height, width = 16, 16  # Stage 3的特征图尺寸
    x = torch.randn(batch_size, dim, height, width)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = smsa(x)
    
    print(f"输出形状: {output.shape}")
    print(f"输出是否包含NaN: {torch.isnan(output).any().item()}")
    print(f"输出是否包含Inf: {torch.isinf(output).any().item()}")
    print(f"参数数量: {sum(p.numel() for p in smsa.parameters())}")
    print("✓ SMSA测试通过\n")
    
    return True

def test_pcsa():
    """测试PCSA模块"""
    print("=" * 60)
    print("测试 PCSA (Progressive Channel-wise Self-Attention)")
    print("=" * 60)
    
    # 创建PCSA模块
    dim = 512
    pcsa = Progressive_Channel_wise_Self_Attention(
        dim=dim,
        reduction_ratio=4
    )
    
    # 创建测试输入
    batch_size = 2
    height, width = 16, 16
    x = torch.randn(batch_size, dim, height, width)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = pcsa(x)
    
    print(f"输出形状: {output.shape}")
    print(f"输出是否包含NaN: {torch.isnan(output).any().item()}")
    print(f"输出是否包含Inf: {torch.isinf(output).any().item()}")
    print(f"参数数量: {sum(p.numel() for p in pcsa.parameters())}")
    print("✓ PCSA测试通过\n")
    
    return True

def test_scsa():
    """测试完整SCSA模块"""
    print("=" * 60)
    print("测试完整 SCSA 模块")
    print("=" * 60)
    
    # 创建SCSA模块
    dim = 512
    scsa = SCSA(
        dim=dim,
        group_kernel_sizes=[3, 5, 7, 9],
        gate_layer='sigmoid',
        reduction_ratio=4
    )
    
    # 创建测试输入
    batch_size = 2
    height, width = 16, 16
    x = torch.randn(batch_size, dim, height, width)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = scsa(x)
    
    print(f"输出形状: {output.shape}")
    print(f"输出是否包含NaN: {torch.isnan(output).any().item()}")
    print(f"输出是否包含Inf: {torch.isinf(output).any().item()}")
    print(f"参数数量: {sum(p.numel() for p in scsa.parameters())}")
    print("✓ SCSA测试通过\n")
    
    return True

def test_different_dimensions():
    """测试不同维度的SCSA"""
    print("=" * 60)
    print("测试不同通道维度")
    print("=" * 60)
    
    dims = [384, 512, 768]  # Tiny, Base, Large的通道数
    
    for dim in dims:
        print(f"\n测试维度: {dim}")
        scsa = SCSA(dim=dim, reduction_ratio=4)
        x = torch.randn(1, dim, 16, 16)
        
        with torch.no_grad():
            output = scsa(x)
        
        assert output.shape == x.shape, f"输出形状不匹配: {output.shape} vs {x.shape}"
        print(f"  ✓ 维度 {dim} 测试通过")
    
    print("\n✓ 所有维度测试通过\n")
    return True

def test_gradient_flow():
    """测试梯度流动"""
    print("=" * 60)
    print("测试梯度流动")
    print("=" * 60)
    
    dim = 512
    scsa = SCSA(dim=dim, reduction_ratio=4)
    x = torch.randn(1, dim, 16, 16, requires_grad=True)
    
    # 前向传播
    output = scsa(x)
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = x.grad is not None
    grad_has_nan = torch.isnan(x.grad).any().item() if has_grad else False
    grad_has_inf = torch.isinf(x.grad).any().item() if has_grad else False
    
    print(f"输入有梯度: {has_grad}")
    print(f"梯度包含NaN: {grad_has_nan}")
    print(f"梯度包含Inf: {grad_has_inf}")
    
    # 检查所有参数是否有梯度
    params_with_grad = sum(1 for p in scsa.parameters() if p.grad is not None)
    total_params = sum(1 for _ in scsa.parameters())
    print(f"有梯度的参数: {params_with_grad}/{total_params}")
    
    assert has_grad and not grad_has_nan and not grad_has_inf
    print("✓ 梯度流动测试通过\n")
    
    return True

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试 SCSA 模块")
    print("=" * 60 + "\n")
    
    tests = [
        ("SMSA模块", test_smsa),
        ("PCSA模块", test_pcsa),
        ("完整SCSA模块", test_scsa),
        ("不同维度", test_different_dimensions),
        ("梯度流动", test_gradient_flow),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "通过" if result else "失败"))
        except Exception as e:
            print(f"✗ {test_name} 失败: {str(e)}\n")
            results.append((test_name, f"失败: {str(e)}"))
    
    # 打印测试总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    for test_name, result in results:
        status = "✓" if result == "通过" else "✗"
        print(f"{status} {test_name}: {result}")
    
    all_passed = all(result == "通过" for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过! SCSA模块工作正常。")
    else:
        print("部分测试失败，请检查错误信息。")
    print("=" * 60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
