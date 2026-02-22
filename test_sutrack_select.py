"""
Test script for SUTrack-Select model
用于测试选择性深度集成模块的脚本
"""
import torch
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lib.config.sutrack_select.config import cfg, update_config_from_file
from lib.models.sutrack_select import build_sutrack_select


def test_model_creation():
    """测试模型创建"""
    print("="*60)
    print("测试 1: 模型创建")
    print("="*60)
    
    # 加载配置
    config_file = 'experiments/sutrack_select/sutrack_select_t224.yaml'
    if os.path.exists(config_file):
        update_config_from_file(config_file)
        print(f"✓ 成功加载配置文件: {config_file}")
    else:
        print(f"⚠️  配置文件不存在: {config_file}，使用默认配置")
    
    # 创建模型
    try:
        model = build_sutrack_select(cfg)
        print("✓ 模型创建成功")
        
        # 检查选择性深度模块
        if hasattr(model, 'use_selective_depth') and model.use_selective_depth:
            print("✓ 选择性深度集成已启用")
            
            # 检查 encoder 中的模块
            encoder_body = model.encoder.body
            if hasattr(encoder_body, 'use_selective_depth'):
                print(f"  - Encoder use_selective_depth: {encoder_body.use_selective_depth}")
            if hasattr(encoder_body, 'selective_depth_module'):
                print(f"  - Selective depth module: {type(encoder_body.selective_depth_module).__name__}")
        else:
            print("⚠️  选择性深度集成未启用")
        
        return model
    except Exception as e:
        print(f"✗ 模型创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "="*60)
    print("测试 2: 前向传播")
    print("="*60)
    
    if model is None:
        print("✗ 模型未创建，跳过测试")
        return
    
    try:
        # 设置为评估模式
        model.eval()
        
        # 创建虚拟输入
        batch_size = 2
        template_size = cfg.DATA.TEMPLATE.SIZE
        search_size = cfg.DATA.SEARCH.SIZE
        
        template_list = [torch.randn(batch_size, 3, template_size, template_size)]
        search_list = [torch.randn(batch_size, 3, search_size, search_size)]
        template_anno_list = [torch.tensor([[0.5, 0.5, 0.2, 0.2]] * batch_size)]
        
        print(f"输入尺寸:")
        print(f"  - Template: {template_list[0].shape}")
        print(f"  - Search: {search_list[0].shape}")
        
        # 前向传播
        with torch.no_grad():
            # Encoder forward
            output = model(
                template_list=template_list,
                search_list=search_list,
                template_anno_list=template_anno_list,
                text_src=None,
                task_index=torch.tensor([0] * batch_size),
                mode="encoder"
            )
            print(f"✓ Encoder 前向传播成功")
            print(f"  - 输出形状: {output[0].shape}")
            
            # Decoder forward
            pred, task_pred = model(
                feature=output,
                mode="decoder"
            )
            print(f"✓ Decoder 前向传播成功")
            print(f"  - 预测框形状: {pred['pred_boxes'].shape}")
            
            # 检查选择性深度统计
            if hasattr(model.encoder.body, 'selective_depth_module'):
                stats = model.encoder.body.selective_depth_module.get_depth_usage_stats()
                if stats is not None:
                    print(f"\n深度使用统计:")
                    print(f"  - 总前向次数: {stats['total_forwards']}")
                    print(f"  - 平均使用率: {stats['avg_usage_rate']:.2%}")
                    print(f"  - 各层使用率: {stats['usage_rate_per_layer']}")
        
        print("✓ 所有前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 前向传播失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_selection_loss(model):
    """测试选择损失计算"""
    print("\n" + "="*60)
    print("测试 3: 选择损失计算")
    print("="*60)
    
    if model is None:
        print("✗ 模型未创建，跳过测试")
        return
    
    try:
        # 设置为训练模式
        model.train()
        
        # 创建虚拟输入
        batch_size = 2
        template_size = cfg.DATA.TEMPLATE.SIZE
        search_size = cfg.DATA.SEARCH.SIZE
        
        template_list = [torch.randn(batch_size, 3, template_size, template_size)]
        search_list = [torch.randn(batch_size, 3, search_size, search_size)]
        template_anno_list = [torch.tensor([[0.5, 0.5, 0.2, 0.2]] * batch_size)]
        
        # 前向传播
        output = model(
            template_list=template_list,
            search_list=search_list,
            template_anno_list=template_anno_list,
            text_src=None,
            task_index=torch.tensor([0] * batch_size),
            mode="encoder"
        )
        
        # 获取选择损失
        selection_loss = model.get_selection_loss()
        print(f"✓ 选择损失计算成功")
        print(f"  - 损失值: {selection_loss if isinstance(selection_loss, float) else selection_loss.item()}")
        
        # 检查是否可以反向传播
        if not isinstance(selection_loss, float):
            print(f"  - 是否需要梯度: {selection_loss.requires_grad}")
        
        return True
        
    except Exception as e:
        print(f"✗ 选择损失计算失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("SUTrack-Select 模型测试")
    print("="*60)
    
    # 测试1: 模型创建
    model = test_model_creation()
    
    # 测试2: 前向传播
    test_forward_pass(model)
    
    # 测试3: 选择损失
    test_selection_loss(model)
    
    print("\n" + "="*60)
    print("所有测试完成")
    print("="*60)


if __name__ == '__main__':
    main()
