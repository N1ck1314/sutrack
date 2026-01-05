"""
调试 EUCB Encoder NaN 问题
"""
import torch
import sys
sys.path.append('.')

from lib.config.sutrack_EUCB.config import cfg, update_config_from_file
from lib.models.sutrack_EUCB import build_sutrack

print("="*60)
print("🔍 调试 EUCB Encoder NaN 问题")
print("="*60)

# 加载配置
config_path = "experiments/sutrack_EUCB/sutrack_eucb_t224.yaml"
update_config_from_file(config_path)

print("\n📋 关键配置:")
print(f"  - ENCODER.TYPE: {cfg.MODEL.ENCODER.TYPE}")
print(f"  - ENCODER.DROP_PATH: {cfg.MODEL.ENCODER.DROP_PATH}")
print(f"  - ENCODER.PRETRAIN_TYPE: {cfg.MODEL.ENCODER.PRETRAIN_TYPE}")
print(f"  - DECODER.USE_EUCB: {cfg.MODEL.DECODER.USE_EUCB}")

# 构建模型
print("\n🏗️  构建模型...")
model = build_sutrack(cfg)
model.cuda()
model.eval()

print("✅ 模型构建成功")

# 创建测试输入
batch_size = 2
template_size = cfg.DATA.TEMPLATE.SIZE
search_size = cfg.DATA.SEARCH.SIZE

print(f"\n🧪 创建测试输入 (batch_size={batch_size})...")
print(f"  - template: {batch_size} x 3 x {template_size} x {template_size}")
print(f"  - search: {batch_size} x 3 x {search_size} x {search_size}")

template = torch.randn(batch_size, 3, template_size, template_size).cuda()
search = torch.randn(batch_size, 3, search_size, search_size).cuda()
template_anno = torch.tensor([[0.4, 0.4, 0.2, 0.2]] * batch_size).cuda()
task_index = torch.tensor([0, 1]).cuda()

# 检查输入是否有 NaN
assert not torch.isnan(template).any(), "Template 输入有 NaN！"
assert not torch.isnan(search).any(), "Search 输入有 NaN！"
print("✅ 输入数据正常（无 NaN）")

# 前向传播 - Encoder
print("\n🚀 测试 Encoder 前向传播...")
with torch.no_grad():
    try:
        encoder_output = model.forward_encoder(
            template_list=[template],
            search_list=[search],
            template_anno_list=[template_anno],
            text_src=None,
            task_index=task_index
        )
        
        # 检查输出
        if isinstance(encoder_output, (list, tuple)):
            enc_tensor = encoder_output[0]
        else:
            enc_tensor = encoder_output
        
        print(f"✅ Encoder 输出形状: {enc_tensor.shape}")
        print(f"   - dtype: {enc_tensor.dtype}")
        print(f"   - device: {enc_tensor.device}")
        
        # 检查 NaN
        nan_count = torch.isnan(enc_tensor).sum().item()
        total_elements = enc_tensor.numel()
        
        if nan_count > 0:
            print(f"\n❌ 检测到 NaN！")
            print(f"   - NaN 数量: {nan_count} / {total_elements}")
            print(f"   - NaN 比例: {nan_count / total_elements * 100:.2f}%")
            
            # 检查是否全是 NaN
            if nan_count == total_elements:
                print("   - ⚠️  所有输出都是 NaN！")
            
            # 尝试找出哪一层产生了 NaN
            print("\n🔍 尝试定位 NaN 来源...")
            
            # 检查 patch embedding
            print("   检查 patch_embed...")
            encoder_body = model.encoder.body
            with torch.no_grad():
                # Template patch embedding
                template_embed = encoder_body.patch_embed(template)
                if torch.isnan(template_embed).any():
                    print("   ❌ Template patch_embed 输出有 NaN！")
                else:
                    print("   ✅ Template patch_embed 正常")
                
                # Search patch embedding
                search_embed = encoder_body.patch_embed(search)
                if torch.isnan(search_embed).any():
                    print("   ❌ Search patch_embed 输出有 NaN！")
                else:
                    print("   ✅ Search patch_embed 正常")
        else:
            print("✅ Encoder 输出正常（无 NaN）")
            print(f"   - min: {enc_tensor.min().item():.6f}")
            print(f"   - max: {enc_tensor.max().item():.6f}")
            print(f"   - mean: {enc_tensor.mean().item():.6f}")
            print(f"   - std: {enc_tensor.std().item():.6f}")
            
            # 测试 Decoder
            print("\n🚀 测试 Decoder 前向传播...")
            decoder_output, task_output = model.forward_decoder(encoder_output)
            print(f"✅ Decoder 输出正常")
            print(f"   - pred_boxes: {decoder_output['pred_boxes'].shape}")
            
    except Exception as e:
        print(f"\n❌ 前向传播失败！")
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("调试完成")
print("="*60)
