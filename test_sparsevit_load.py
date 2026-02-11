#!/usr/bin/env python3
"""测试SparseViT模型能否正常加载"""
import sys
import os

# 添加路径
sys.path.insert(0, '/home/nick/code/code.sutrack/SUTrack')

try:
    print("1. 导入库...")
    import torch
    from lib.test.parameter.sutrack_SparseViT import parameters
    from lib.models.sutrack_SparseViT import build_sutrack
    print("✓ 导入成功")
    
    print("\n2. 加载参数...")
    params = parameters('sutrack_sparsevit_t224')
    print(f"✓ 参数加载成功")
    print(f"  Checkpoint: {params.checkpoint}")
    print(f"  Checkpoint exists: {os.path.exists(params.checkpoint)}")
    
    print("\n3. 构建模型...")
    network = build_sutrack(params.cfg)
    print(f"✓ 模型构建成功")
    
    print("\n4. 加载checkpoint...")
    checkpoint = torch.load(params.checkpoint, map_location='cpu')
    print(f"✓ Checkpoint读取成功")
    print(f"  Keys: {list(checkpoint.keys())}")
    
    print("\n5. 加载模型权重...")
    network.load_state_dict(checkpoint['net'], strict=False)
    print(f"✓ 权重加载成功")
    
    print("\n6. 移动到CUDA...")
    if torch.cuda.is_available():
        network = network.cuda()
        print(f"✓ 模型已移动到GPU")
    else:
        print(f"⚠ CUDA不可用，使用CPU")
    
    print("\n✅ 所有测试通过！SparseViT模型可以正常加载")
    
except Exception as e:
    print(f"\n❌ 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
