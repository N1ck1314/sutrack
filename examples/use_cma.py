"""
使用SUTrack-CMA模型的示例
展示如何使用跨模态注意力增强的SUTrack模型进行训练和测试
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from lib.config.sutrack_CMA.config import cfg, update_config_from_file
from lib.models.sutrack_CMA import build_sutrack_cma


def load_config():
    """加载配置文件"""
    config_file = os.path.join(project_root, 'experiments/sutrack_CMA/sutrack_cma_t224.yaml')
    
    if not os.path.exists(config_file):
        print(f"⚠️  配置文件未找到: {config_file}")
        print("使用默认配置...")
    else:
        print(f"✓ 加载配置文件: {config_file}")
        update_config_from_file(config_file)
    
    return cfg


def build_model(cfg):
    """构建SUTrack-CMA模型"""
    print("\n" + "="*60)
    print("开始构建 SUTrack-CMA 模型...")
    print("="*60)
    
    # 打印关键配置
    print(f"\n📋 关键配置:")
    print(f"  - Encoder类型: {cfg.MODEL.ENCODER.TYPE}")
    print(f"  - 使用CMA: {cfg.MODEL.USE_CMA}")
    if cfg.MODEL.USE_CMA:
        print(f"  - CMA隐藏层比例: {cfg.MODEL.CMA.HIDDEN_RATIO}")
    print(f"  - Decoder类型: {cfg.MODEL.DECODER.TYPE}")
    print(f"  - 搜索区域大小: {cfg.DATA.SEARCH.SIZE}")
    print(f"  - 模板区域大小: {cfg.DATA.TEMPLATE.SIZE}")
    
    # 构建模型
    model = build_sutrack_cma(cfg)
    
    print(f"\n✓ 模型构建成功!")
    return model


def test_forward_pass(model, cfg):
    """测试模型前向传播"""
    print("\n" + "="*60)
    print("测试模型前向传播...")
    print("="*60)
    
    batch_size = 2
    search_size = cfg.DATA.SEARCH.SIZE
    template_size = cfg.DATA.TEMPLATE.SIZE
    
    # 创建模拟输入
    template_list = [torch.randn(batch_size, 3, template_size, template_size)]
    search_list = [torch.randn(batch_size, 3, search_size, search_size)]
    template_anno_list = [torch.randn(batch_size, 3, template_size, template_size)]
    
    print(f"\n📊 输入尺寸:")
    print(f"  - 模板区域: {template_list[0].shape}")
    print(f"  - 搜索区域: {search_list[0].shape}")
    
    # 如果使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - 使用设备: {device}")
    
    if device.type == 'cuda':
        model = model.cuda()
        template_list = [t.cuda() for t in template_list]
        search_list = [s.cuda() for s in search_list]
        template_anno_list = [a.cuda() for a in template_anno_list]
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        # Encoder
        features = model.forward(
            template_list=template_list,
            search_list=search_list,
            template_anno_list=template_anno_list,
            text_src=None,
            task_index=None,
            mode="encoder"
        )
        print(f"\n✓ Encoder输出形状: {features[0].shape}")
        
        # Decoder
        pred_dict, task_pred = model.forward(
            feature=features,
            mode="decoder"
        )
        print(f"\n✓ Decoder输出:")
        print(f"  - 预测边界框: {pred_dict['pred_boxes'].shape}")
        print(f"  - 得分图: {pred_dict['score_map'].shape}")
        if 'size_map' in pred_dict:
            print(f"  - 尺寸图: {pred_dict['size_map'].shape}")
        if 'offset_map' in pred_dict:
            print(f"  - 偏移图: {pred_dict['offset_map'].shape}")
        print(f"  - 任务预测: {task_pred.shape}")
    
    print("\n✓ 前向传播测试成功!")
    return pred_dict


def count_parameters(model):
    """统计模型参数量"""
    print("\n" + "="*60)
    print("模型参数统计")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 参数统计:")
    print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  - 可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  - 冻结参数: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
    
    # 分模块统计
    print(f"\n📊 各模块参数量:")
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    task_decoder_params = sum(p.numel() for p in model.task_decoder.parameters())
    
    print(f"  - Encoder: {encoder_params:,} ({encoder_params/1e6:.2f}M)")
    if hasattr(model.encoder, 'cma_module'):
        cma_params = sum(p.numel() for p in model.encoder.cma_module.parameters())
        print(f"    └─ CMA模块: {cma_params:,} ({cma_params/1e6:.2f}M)")
    print(f"  - Decoder: {decoder_params:,} ({decoder_params/1e6:.2f}M)")
    print(f"  - Task Decoder: {task_decoder_params:,} ({task_decoder_params/1e6:.2f}M)")
    if model.text_encoder is not None:
        text_encoder_params = sum(p.numel() for p in model.text_encoder.parameters())
        print(f"  - Text Encoder: {text_encoder_params:,} ({text_encoder_params/1e6:.2f}M)")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("SUTrack-CMA 使用示例")
    print("="*60)
    
    # 1. 加载配置
    cfg = load_config()
    
    # 2. 构建模型
    model = build_model(cfg)
    
    # 3. 统计参数
    count_parameters(model)
    
    # 4. 测试前向传播
    try:
        test_forward_pass(model, cfg)
    except Exception as e:
        print(f"\n⚠️  前向传播测试失败: {e}")
        print("提示: 请确保已正确安装所有依赖并下载预训练模型")
    
    print("\n" + "="*60)
    print("示例运行完成!")
    print("="*60)
    
    print("\n📝 训练命令示例:")
    print("python tracking/train.py --script sutrack_CMA --config sutrack_cma_t224 --save_dir output/sutrack_cma --mode multiple --nproc_per_node 4")
    
    print("\n📝 测试命令示例:")
    print("python tracking/test.py sutrack_CMA sutrack_cma_t224 --dataset lasot --threads 4 --num_gpus 1")
    

if __name__ == '__main__':
    main()
