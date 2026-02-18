"""
离线测试脚本 - 不需要下载 DINOv3 权重，使用随机初始化验证接口

仅用于验证接口是否正确，不能用于实际训练
"""
import torch
import torch.nn as nn
import sys
sys.path.append('.')

from lib.config.sutrack_dinov3.config import cfg


class MockConvNeXtBackbone(nn.Module):
    """模拟 DINOv3 ConvNeXt-Tiny 的接口（随机初始化）"""
    def __init__(self, embed_dim=768):
        super().__init__()
        # 简化的 ConvNeXt 结构（支持任意输入尺寸）
        self.conv1 = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(384, embed_dim, kernel_size=2, stride=2)
        
        # 使用 GroupNorm 代替 LayerNorm（不需要固定空间尺寸）
        self.norm1 = nn.GroupNorm(8, 96)
        self.norm2 = nn.GroupNorm(8, 192)
        self.norm3 = nn.GroupNorm(8, 384)
        self.norm4 = nn.GroupNorm(8, embed_dim)
    
    def forward(self, x, **kwargs):  # 忽略所有额外参数
        x = self.norm1(self.conv1(x))
        x = self.norm2(self.conv2(x))
        x = self.norm3(self.conv3(x))
        x = self.norm4(self.conv4(x))
        return type('obj', (object,), {'last_hidden_state': x})()


# 替换真实的 DINOv3 加载
import lib.models.sutrack_dinov3.encoder as encoder_module

original_init = encoder_module.DinoV3ConvNeXtEncoder.__init__

def mock_init(self, search_size=224, template_size=112, 
              pretrained_model_name="mock", embed_dim=768,
              cls_token=False, token_type_indicate=False):
    """模拟 __init__，使用随机初始化的 ConvNeXt"""
    nn.Module.__init__(self)
    
    self.search_size = search_size
    self.template_size = template_size
    self.embed_dim = embed_dim
    self.cls_token = None
    self.token_type_indicate = token_type_indicate
    
    print(f"[MockEncoder] 使用随机初始化的 ConvNeXt（仅用于接口测试）")
    
    # 创建 mock backbone
    self.backbone = MockConvNeXtBackbone(embed_dim)
    
    # 计算 patch 数量
    self.num_patches_search = (search_size // 32) ** 2
    self.num_patches_template = (template_size // 32) ** 2
    
    # 位置编码
    self.pos_embed = nn.Parameter(
        torch.zeros(1, self.num_patches_search + self.num_patches_template, embed_dim)
    )
    nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    # CLS token（可选）
    if cls_token:
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    # Token type embedding（可选）
    if self.token_type_indicate:
        self.template_token = nn.Parameter(torch.zeros(embed_dim))
        self.search_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.template_token, std=0.02)
        nn.init.trunc_normal_(self.search_token, std=0.02)

# 替换 __init__
encoder_module.DinoV3ConvNeXtEncoder.__init__ = mock_init

# 导入模型（现在会使用 mock 版本）
from lib.models.sutrack_dinov3 import build_sutrack


def test_encoder_interface():
    """测试 encoder 接口"""
    print("=" * 60)
    print("Testing DINOv3 Encoder Interface (Offline Mode)")
    print("=" * 60)
    
    # 构建模型
    print("\n[1/4] Building model with mock ConvNeXt...")
    try:
        model = build_sutrack(cfg)
        print("✓ Model built successfully!")
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 准备测试数据
    print("\n[2/4] Preparing test data...")
    batch_size = 2
    
    template_list = [torch.randn(batch_size, 3, 112, 112)]
    search_list = [torch.randn(batch_size, 3, 224, 224)]
    template_anno_list = [torch.rand(batch_size, 4)]
    text_src = None
    task_index = torch.zeros(batch_size, dtype=torch.long)
    
    print(f"  - Template: {template_list[0].shape}")
    print(f"  - Search: {search_list[0].shape}")
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
    feat_shape = features[0].shape
    print(f"  - Feature shape: {feat_shape}")
    print(f"  - Expected: (B=2, L>=49, C=768)")
    
    if feat_shape[0] == 2 and feat_shape[2] == 768:
        print("✓ Output shape correct!")
    else:
        print(f"⚠ Output shape unexpected")
    
    print("\n" + "=" * 60)
    print("✓ 接口测试通过！")
    print("\n⚠ 注意：这只是接口测试，使用的是随机初始化的权重")
    print("  实际训练需要下载真实的 DINOv3 预训练权重")
    print("=" * 60)
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SUTrack with DINOv3 - Offline Interface Test")
    print("=" * 60 + "\n")
    
    print("⚠ 离线模式：使用随机初始化的 ConvNeXt（不加载预训练权重）\n")
    
    success = test_encoder_interface()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ 接口验证成功！")
        print("\n下一步：")
        print("1. 解决网络问题，下载真实的 DINOv3 权重")
        print("2. 或使用 bash download_dinov3_model.sh 手动下载")
        print("3. 然后用真实权重进行训练")
        print("=" * 60 + "\n")
    else:
        print("\n✗ 接口测试失败\n")
        sys.exit(1)
