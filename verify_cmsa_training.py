#!/usr/bin/env python
"""
快速验证CMSA模块是否在训练中被正确使用
按照记忆中的策略：进行短轮次训练（如5轮）快速验证
"""

import subprocess
import sys

def verify_cmsa_module():
    print("=" * 70)
    print("🚀 验证CMSA模块训练集成")
    print("=" * 70)
    print()
    
    print("📋 验证方法：")
    print("1. 配置确认 - 查看训练启动时打印的CMSA配置")
    print("2. 模块初始化 - 验证CMSA模块是否成功创建")
    print("3. 短轮次训练 - 运行5个epoch验证模块工作正常")
    print()
    
    print("=" * 70)
    print("方法1: 直接查看配置文件")
    print("=" * 70)
    
    config_file = "experiments/sutrack_S4F/sutrack_s4f_cmsa.yaml"
    try:
        with open(config_file, 'r') as f:
            lines = f.readlines()
            print(f"\n📄 配置文件: {config_file}\n")
            in_encoder = False
            for line in lines:
                if 'ENCODER:' in line:
                    in_encoder = True
                if in_encoder and ('USE_CMSA' in line or 'CMSA_MODE' in line or 'USE_SSM' in line):
                    print(f"  {line.rstrip()}")
                if in_encoder and line.strip() and not line.strip().startswith('#') and ':' in line and 'CMSA' not in line and 'SSM' not in line and 'ENCODER' not in line:
                    if not any(x in line for x in ['TYPE', 'DROP', 'PRETRAIN', 'PATCH', 'USE_', 'STRIDE', 'POS', 'TOKEN', 'CLASS']):
                        in_encoder = False
    except Exception as e:
        print(f"❌ 无法读取配置文件: {e}")
    
    print("\n" + "=" * 70)
    print("方法2: 查看默认配置")
    print("=" * 70)
    
    try:
        from lib.config.sutrack_S4F.config import cfg
        print(f"\n📋 默认配置值:")
        print(f"  USE_CMSA: {cfg.MODEL.ENCODER.get('USE_CMSA', 'NOT SET')}")
        print(f"  CMSA_MODE: {cfg.MODEL.ENCODER.get('CMSA_MODE', 'NOT SET')}")
        print(f"  USE_SSM: {cfg.MODEL.ENCODER.get('USE_SSM', 'NOT SET')}")
    except Exception as e:
        print(f"❌ 无法加载配置: {e}")
    
    print("\n" + "=" * 70)
    print("方法3: 启动训练并观察输出")
    print("=" * 70)
    print()
    print("💡 运行以下命令查看完整的训练启动信息：")
    print()
    print("python tracking/train.py \\")
    print("    --script sutrack_S4F \\")
    print("    --config sutrack_s4f_cmsa \\")
    print("    --save_dir ./output \\")
    print("    --mode single")
    print()
    print("⚡ 或者修改配置文件 EPOCH: 5 进行快速验证（5轮训练）")
    print()
    
    print("=" * 70)
    print("📊 训练时应该看到的确认信息：")
    print("=" * 70)
    print("""
🔍 CMSA模块配置确认
============================================================
✓ CMSA启用状态: 🟢 已启用
✓ CMSA融合模式: cmsa
✓ 状态空间模型(SSM): 🟢 启用
✓ 多模态融合策略: 跨模态空间感知 (替代简单拼接)
============================================================

🔍 验证CMSA模块实际初始化状态...
✅ CMSA模块已成功初始化！
   - cmsa_search: MultiModalFusionWithCMSA
   - cmsa_template: MultiModalFusionWithCMSA
    """)
    
    print("=" * 70)
    print("🎯 关键验证点总结")
    print("=" * 70)
    print()
    print("1. ✅ 配置文件中 USE_CMSA: True")
    print("2. ✅ 训练启动时打印 '🟢 已启用'")
    print("3. ✅ 显示 'MultiModalFusionWithCMSA' 模块初始化")
    print("4. ✅ 训练日志中损失正常下降")
    print()
    print("如果看到以上信息，说明CMSA模块已正确集成到训练中！")
    print()

if __name__ == "__main__":
    verify_cmsa_module()
