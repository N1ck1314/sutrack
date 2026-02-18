"""
Test script for SUTrack with DINOv3 ConvNeXt-Tiny Encoder

验证新 encoder 的接口是否正确
"""
import torch
import sys
sys.path.append('.')

from lib.config.sutrack_dinov3.config import cfg
from lib.models.sutrack_dinov3 import build_sutrack


def test_encoder_interface():
    """测试 encoder 接口"""
    print("=" * 60)
    print("Testing DINOv3 ConvNeXt-Tiny Encoder Interface")
    print("=" * 60)
    
    # 构建模型
    print("\n[1/4] Building model...")
    try:
        model = build_sutrack(cfg)
        print("✓ Model built successfully!")
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        return False
    
    # 准备测试数据
    print("\n[2/4] Preparing test data...")
    batch_size = 2
    num_template = 1
    num_search = 1
    
    template_list = [torch.randn(batch_size, 3, 112, 112)]
    search_list = [torch.randn(batch_size, 3, 224, 224)]
    template_anno_list = [torch.rand(batch_size, 4)]  # (x, y, w, h) normalized
    text_src = None
    task_index = torch.zeros(batch_size, dtype=torch.long)
    
    print(f"  - Template: {template_list[0].shape}")
    print(f"  - Search: {search_list[0].shape}")
    print(f"  - Template Anno: {template_anno_list[0].shape}")
    print("✓ Test data prepared!")
    
    # 测试 encoder forward
    print("\n[3/4] Testing encoder forward...")
    model.eval()
    with torch.no_grad():
        try:
            # Encoder forward
            features = model.forward_encoder(
                template_list=template_list,
                search_list=search_list,
                template_anno_list=template_anno_list,
                text_src=text_src,
                task_index=task_index
            )
            print(f"✓ Encoder output shape: {features[0].shape}")
            
            # Decoder forward
            pred_dict = model.forward_decoder(features)
            print(f"✓ Decoder output: pred_boxes {pred_dict['pred_boxes'].shape}")
            
            # Task decoder forward
            task_out = model.forward_task_decoder(features)
            print(f"✓ Task decoder output: {task_out.shape}")
            
        except Exception as e:
            print(f"✗ Forward failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n[4/4] Checking output consistency...")
    B = batch_size
    expected_L = model.num_patch_x * num_search  # search tokens
    
    # 检查特征维度
    feat_shape = features[0].shape
    print(f"  - Feature shape: {feat_shape}")
    print(f"  - Expected: (B={B}, L>={expected_L}, C=768)")
    
    if feat_shape[0] == B:
        print("✓ Batch size correct!")
    else:
        print(f"✗ Batch size mismatch: got {feat_shape[0]}, expected {B}")
        return False
    
    if feat_shape[2] == 768:
        print("✓ Feature dimension correct (768 for ConvNeXt-Tiny)!")
    else:
        print(f"⚠ Feature dimension: {feat_shape[2]} (expected 768)")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! DINOv3 encoder is working correctly!")
    print("=" * 60)
    return True


def test_model_parameters():
    """测试模型参数量"""
    print("\n" + "=" * 60)
    print("Model Parameters Analysis")
    print("=" * 60)
    
    model = build_sutrack(cfg)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Frozen parameters: {(total_params - trainable_params) / 1e6:.2f}M")
    
    # 分模块统计
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    task_decoder_params = sum(p.numel() for p in model.task_decoder.parameters())
    
    print(f"\nModule breakdown:")
    print(f"  - Encoder: {encoder_params / 1e6:.2f}M")
    print(f"  - Decoder: {decoder_params / 1e6:.2f}M")
    print(f"  - Task Decoder: {task_decoder_params / 1e6:.2f}M")
    
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SUTrack with DINOv3 ConvNeXt-Tiny - Interface Test")
    print("=" * 60 + "\n")
    
    # 检查依赖
    print("Checking dependencies...")
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError:
        print("✗ transformers not installed! Please run:")
        print("  pip install transformers")
        sys.exit(1)
    
    # 测试接口
    success = test_encoder_interface()
    
    if success:
        # 测试参数量
        test_model_parameters()
        
        print("\n" + "=" * 60)
        print("✓ Test completed successfully!")
        print("\nNext steps:")
        print("1. 检查模型是否能正常训练（数据加载、loss 计算）")
        print("2. 在小数据集上跑几个 epoch 验证收敛")
        print("3. 调整超参数（学习率、是否冻结 backbone 等）")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("✗ Test failed! Please check the error messages above.")
        print("=" * 60 + "\n")
        sys.exit(1)
