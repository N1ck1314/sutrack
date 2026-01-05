"""
SUTrack with MFEblock 使用示例
演示如何使用 MFEblock 增强版的 SUTrack 模型
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from lib.config.sutrack_MFE.config import cfg, update_config_from_file
from lib.models.sutrack_MFE import build_sutrack


def test_mfe_module():
    """测试 MFEblock 模块的基本功能"""
    print("=" * 60)
    print("SUTrack with MFEblock 模块测试")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n[1] 加载配置文件...")
    config_path = os.path.join(project_root, 'experiments/sutrack_MFE/sutrack_mfe_t224.yaml')
    update_config_from_file(config_path)
    print(f"✓ 配置加载成功: {config_path}")
    print(f"  - MFE启用: {cfg.MODEL.USE_MFE}")
    print(f"  - 膨胀率: {cfg.MODEL.MFE_ATROUS_RATES}")
    print(f"  - Encoder类型: {cfg.MODEL.ENCODER.TYPE}")
    
    # 2. 构建模型
    print("\n[2] 构建 SUTrack-MFE 模型...")
    try:
        model = build_sutrack(cfg)
        print("✓ 模型构建成功")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - 总参数量: {total_params / 1e6:.2f}M")
        print(f"  - 可训练参数: {trainable_params / 1e6:.2f}M")
        
    except Exception as e:
        print(f"✗ 模型构建失败: {e}")
        return False
    
    # 3. 测试前向传播
    print("\n[3] 测试前向传播...")
    try:
        model.eval()
        batch_size = 2
        
        # 创建虚拟输入
        template = torch.randn(batch_size, 3, 112, 112)
        search = torch.randn(batch_size, 3, 224, 224)
        template_anno = torch.randn(batch_size, 4)
        text_src = torch.randn(batch_size, 1, 384) if cfg.DATA.MULTI_MODAL_LANGUAGE else None
        task_index = torch.zeros(batch_size, dtype=torch.long)
        
        print(f"  输入形状:")
        print(f"    - Template: {template.shape}")
        print(f"    - Search: {search.shape}")
        
        # Encoder 前向传播
        with torch.no_grad():
            features = model(
                template_list=[template],
                search_list=[search],
                template_anno_list=[template_anno],
                text_src=text_src,
                task_index=task_index,
                mode="encoder"
            )
            
            print(f"  ✓ Encoder 输出: {features[0].shape}")
            
            # Decoder 前向传播
            pred_dict, task_pred = model(
                feature=features,
                mode="decoder"
            )
            
            print(f"  ✓ Decoder 输出:")
            print(f"    - pred_boxes: {pred_dict['pred_boxes'].shape}")
            print(f"    - score_map: {pred_dict['score_map'].shape}")
            print(f"    - task_pred: {task_pred.shape}")
        
        print("✓ 前向传播测试通过")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 检查 MFEblock 是否被正确应用
    print("\n[4] 验证 MFEblock 集成...")
    if hasattr(model.encoder, 'use_mfe') and model.encoder.use_mfe:
        print("✓ MFEblock 已启用")
        if hasattr(model.encoder, 'mfe_module'):
            mfe_params = sum(p.numel() for p in model.encoder.mfe_module.parameters())
            print(f"  - MFEblock 参数量: {mfe_params / 1e6:.2f}M")
    else:
        print("✗ MFEblock 未启用")
        return False
    
    print("\n" + "=" * 60)
    print("所有测试通过! SUTrack-MFE 模型可以正常使用")
    print("=" * 60)
    return True


def show_model_structure():
    """显示模型结构概览"""
    print("\n" + "=" * 60)
    print("SUTrack-MFE 模型结构概览")
    print("=" * 60)
    
    config_path = os.path.join(project_root, 'experiments/sutrack_MFE/sutrack_mfe_t224.yaml')
    update_config_from_file(config_path)
    model = build_sutrack(cfg)
    
    print("\n主要模块:")
    print("├── Text Encoder (CLIP)")
    print("├── Visual Encoder (ITPN)")
    print("│   └── MFEblock (多尺度特征增强) ← 新增")
    print("├── Box Decoder (CENTER)")
    print("└── Task Decoder (MLP)")
    
    print("\nMFEblock 配置:")
    print(f"  - 输入通道: {model.encoder.num_channels}")
    print(f"  - 膨胀率: {cfg.MODEL.MFE_ATROUS_RATES}")
    print(f"  - 作用位置: Search Region Features")


if __name__ == "__main__":
    print("\n" + "🚀 SUTrack with MFEblock 演示脚本\n")
    
    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA 不可用，使用 CPU 模式")
    
    # 运行测试
    success = test_mfe_module()
    
    # 显示模型结构
    if success:
        show_model_structure()
    
    print("\n使用提示:")
    print("  训练: cd tracking && python train.py --config ../experiments/sutrack_MFE/sutrack_mfe_t224.yaml --model sutrack_MFE")
    print("  测试: cd tracking && python test.py --config ../experiments/sutrack_MFE/sutrack_mfe_t224.yaml --model sutrack_MFE")
