"""
Encoder module using DINOv3 ConvNeXt-Tiny as backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from lib.utils.misc import is_main_process


class DinoV3ConvNeXtEncoder(nn.Module):
    """
    DINOv3 ConvNeXt-Tiny Encoder for SUTrack
    
    将 template/search/text 打包成统一的 token 序列，输出给 decoder
    """
    
    def __init__(self, 
                 search_size=224, 
                 template_size=112,
                 pretrained_model_name="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
                 embed_dim=768,  # ConvNeXt-Tiny 默认输出维度
                 cls_token=False,
                 token_type_indicate=False):
        """
        Args:
            search_size: 搜索帧尺寸
            template_size: 模板帧尺寸
            pretrained_model_name: HuggingFace 模型名称
            embed_dim: 输出特征维度
            cls_token: 是否添加 cls token
            token_type_indicate: 是否使用 token type embedding（暂时简化，先不实现）
        """
        super().__init__()
        
        self.search_size = search_size
        self.template_size = template_size
        self.embed_dim = embed_dim
        self.cls_token = None
        self.token_type_indicate = token_type_indicate
        
        # 加载 DINOv3 ConvNeXt-Tiny backbone
        print(f"[DinoV3Encoder] Loading DINOv3 ConvNeXt-Tiny from: {pretrained_model_name}")
        
        # 检查是否是本地路径
        import os
        if os.path.exists(pretrained_model_name):
            print(f"[DinoV3Encoder] Loading from local path...")
            try:
                self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name, local_files_only=True)
                self.backbone = AutoModel.from_pretrained(pretrained_model_name, local_files_only=True)
                print(f"[DinoV3Encoder] ✓ Successfully loaded from local path!")
            except Exception as e:
                print(f"[DinoV3Encoder] ✗ Failed to load from local path: {e}")
                raise RuntimeError(
                    f"本地路径加载失败。请检查以下文件是否存在：\n"
                    f"  {pretrained_model_name}/config.json\n"
                    f"  {pretrained_model_name}/preprocessor_config.json\n"
                    f"  {pretrained_model_name}/model.safetensors"
                )
        else:
            # 在线加载
            try:
                print(f"[DinoV3Encoder] Downloading from HuggingFace...")
                self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
                self.backbone = AutoModel.from_pretrained(pretrained_model_name)
                print(f"[DinoV3Encoder] ✓ Successfully loaded from HuggingFace!")
            except Exception as e:
                print(f"[DinoV3Encoder] ✗ Failed to load from HuggingFace: {e}")
                print(f"[DinoV3Encoder] ⚠ 使用随机初始化的 ConvNeXt 替代（仅用于接口测试）")
                print(f"[DinoV3Encoder] ⚠ 警告：这不是预训练模型，性能会很差！")
                
                # 创建简化的 ConvNeXt backbone（随机初始化）
                self.backbone = self._create_mock_convnext(embed_dim)
                self.processor = None  # Mock processor
                print(f"[DinoV3Encoder] ✓ 使用随机初始化的 ConvNeXt（stride=32, dim={embed_dim}）")
        
        # ConvNeXt 输出是 (B, C, H', W') 格式，需要转换成 (B, L, C)
        # ConvNeXt-Tiny 最后一层输出 stride=32，对于 224x224 输入 -> 7x7 feature map
        # 对于 112x112 模板 -> 3x3（实际是 4x4，因为会 padding）
        
        # 计算 patch 数量（基于 stride=32）
        self.num_patches_search = (search_size // 32) ** 2  # 224/32=7, 7*7=49
        self.num_patches_template = (template_size // 32) ** 2  # 112/32=3.5 -> 实际会是 4, 4*4=16
        
        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches_search + self.num_patches_template, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # CLS token（可选）
        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Token type embedding（可选，暂时简化不实现前景/背景 mask）
        if self.token_type_indicate:
            self.template_token = nn.Parameter(torch.zeros(embed_dim))
            self.search_token = nn.Parameter(torch.zeros(embed_dim))
            nn.init.trunc_normal_(self.template_token, std=0.02)
            nn.init.trunc_normal_(self.search_token, std=0.02)
    
    def _create_mock_convnext(self, embed_dim=768):
        """创建一个简化的 ConvNeXt 用于随机初始化（仅用于接口测试）"""
        class MockConvNeXt(nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                # 简化的 ConvNeXt 结构（stride=32）
                # 支持 RGB-D 输入（6通道）
                self.conv1 = nn.Conv2d(6, 96, kernel_size=4, stride=4)  # 改为 6 通道
                self.conv2 = nn.Conv2d(96, 192, kernel_size=2, stride=2)
                self.conv3 = nn.Conv2d(192, 384, kernel_size=2, stride=2)
                self.conv4 = nn.Conv2d(384, embed_dim, kernel_size=2, stride=2)
                
                # 使用 GroupNorm
                self.norm1 = nn.GroupNorm(8, 96)
                self.norm2 = nn.GroupNorm(8, 192)
                self.norm3 = nn.GroupNorm(8, 384)
                self.norm4 = nn.GroupNorm(8, embed_dim)
            
            def forward(self, x, **kwargs):
                x = self.norm1(self.conv1(x))
                x = self.norm2(self.conv2(x))
                x = self.norm3(self.conv3(x))
                x = self.norm4(self.conv4(x))
                # 返回类似 HuggingFace 模型的输出格式
                return type('obj', (object,), {'last_hidden_state': x})()
        
        return MockConvNeXt(embed_dim)
        
    def extract_features(self, images):
        """
        使用 DINOv3 backbone 提取特征
        
        Args:
            images: (B, 3, H, W)
        Returns:
            features: (B, L, C) 其中 L=H'*W', C=embed_dim
        """
        # HuggingFace 模型需要预处理
        # 注意：这里直接使用 backbone，不用 processor（因为输入已经是归一化的 tensor）
        with torch.cuda.amp.autocast(enabled=False):  # DINOv3 在 fp16 可能不稳定
            outputs = self.backbone(images, output_hidden_states=False)
        
        # ConvNeXt 输出 last_hidden_state: (B, C, H', W')
        # 需要转换成 (B, L, C)
        if hasattr(outputs, 'last_hidden_state'):
            feat = outputs.last_hidden_state  # (B, C, H', W')
        else:
            # 备用方案：直接用 pooler_output 或第一个输出
            feat = outputs[0] if isinstance(outputs, tuple) else outputs.pooler_output
        
        # 展平 spatial 维度: (B, C, H', W') -> (B, C, H'*W') -> (B, H'*W', C)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        
        return feat
    
    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        """
        Forward pass
        
        Args:
            template_list: List[Tensor(B, 3, H_t, W_t)] 模板帧列表
            search_list: List[Tensor(B, 3, H_s, W_s)] 搜索帧列表
            template_anno_list: List[Tensor(B, 4)] 模板 bbox（归一化坐标）
            text_src: Tensor(B, L_text, C) 文本 token（可选）
            task_index: 任务索引
        
        Returns:
            xz: List[Tensor(B, L_total, C)] 拼接后的 token 序列
        """
        B = search_list[0].size(0)
        num_template = len(template_list)
        num_search = len(search_list)
        
        # 1. 堆叠模板和搜索帧
        z = torch.stack(template_list, dim=1)  # (B, num_template, 3, H_t, W_t)
        z = z.view(-1, *z.size()[2:])  # (B*num_template, 3, H_t, W_t)
        
        x = torch.stack(search_list, dim=1)  # (B, num_search, 3, H_s, W_s)
        x = x.view(-1, *x.size()[2:])  # (B*num_search, 3, H_s, W_s)
        
        # 2. 提取特征
        z_feat = self.extract_features(z)  # (B*num_template, L_t, C)
        x_feat = self.extract_features(x)  # (B*num_search, L_s, C)
        
        # 3. 加位置编码
        # 注意：DINOv3 ConvNeXt 可能已经有内置的位置编码，这里我们额外加一个可学习的
        # 分别给 search 和 template 加上对应区间的 pos_embed
        L_s = x_feat.size(1)
        L_t = z_feat.size(1)
        
        # 为了简化，假设 L_s 和 L_t 和我们预定义的 num_patches 一致
        # 如果不一致，需要做插值（这里先简单处理）
        if L_s <= self.num_patches_search and L_t <= self.num_patches_template:
            x_feat = x_feat + self.pos_embed[:, :L_s, :]
            z_feat = z_feat + self.pos_embed[:, self.num_patches_search:self.num_patches_search+L_t, :]
        else:
            # 需要插值 pos_embed（暂时跳过，实际使用时需要实现）
            pass
        
        # 4. Token type embedding（可选）
        if self.token_type_indicate:
            x_indicate = self.search_token.view(1, 1, -1).expand(x_feat.size(0), x_feat.size(1), -1)
            z_indicate = self.template_token.view(1, 1, -1).expand(z_feat.size(0), z_feat.size(1), -1)
            x_feat = x_feat + x_indicate
            z_feat = z_feat + z_indicate
        
        # 5. 重组：(B*num, L, C) -> (B, num*L, C)
        x_feat = x_feat.view(B, num_search, L_s, self.embed_dim)
        x_feat = x_feat.reshape(B, -1, self.embed_dim)  # (B, num_search*L_s, C)
        
        z_feat = z_feat.view(B, num_template, L_t, self.embed_dim)
        z_feat = z_feat.reshape(B, -1, self.embed_dim)  # (B, num_template*L_t, C)
        
        # 6. 拼接 search + template + text
        if text_src is not None:
            xz = torch.cat([x_feat, z_feat, text_src], dim=1)
        else:
            xz = torch.cat([x_feat, z_feat], dim=1)
        
        # 7. 添加 CLS token（可选）
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            xz = torch.cat([cls_tokens, xz], dim=1)
        
        # 返回 list 格式，兼容原有接口
        return [xz]


class EncoderBase(nn.Module):
    """
    Encoder wrapper，兼容 SUTrack 的训练/冻结逻辑
    """
    
    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
        super().__init__()
        
        # 冻结/解冻逻辑（和原来的 Fast_iTPN 一致）
        if not train_encoder:
            for name, parameter in encoder.named_parameters():
                # 默认冻结 backbone，只训练 pos_embed / cls_token / token_type
                if any(key in name for key in ['pos_embed', 'cls_token', 'template_token', 'search_token']):
                    parameter.requires_grad_(True)
                else:
                    # 检查 open_layers（如果用户指定了要打开的层）
                    freeze = True
                    for open_layer in open_layers:
                        if open_layer in name:
                            freeze = False
                            break
                    if freeze:
                        parameter.requires_grad_(False)
        
        self.body = encoder
        self.num_channels = num_channels
    
    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        return xs


class Encoder(EncoderBase):
    """
    DINOv3 ConvNeXt Encoder for SUTrack
    """
    
    def __init__(self, 
                 name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None):
        
        # 根据名字选择模型
        if "dinov3_convnext_tiny" in name.lower():
            pretrained_model = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
            embed_dim = 768  # ConvNeXt-Tiny 输出维度
            num_channels = 768
        elif "dinov3_vits16" in name.lower():
            pretrained_model = "facebook/dinov3-vits16-pretrain-lvd1689m"
            embed_dim = 384
            num_channels = 384
        else:
            raise ValueError(f"Unknown encoder type: {name}")
        
        encoder = DinoV3ConvNeXtEncoder(
            search_size=search_size,
            template_size=template_size,
            pretrained_model_name=pretrained_model,
            embed_dim=embed_dim,
            cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN if cfg else False,
            token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE if cfg else False
        )
        
        super().__init__(encoder, train_encoder, open_layers, num_channels)


def build_encoder(cfg):
    """
    构建 DINOv3 encoder
    """
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(
        cfg.MODEL.ENCODER.TYPE,
        train_encoder,
        cfg.DATA.SEARCH.SIZE,
        cfg.DATA.TEMPLATE.SIZE,
        cfg.TRAIN.ENCODER_OPEN,
        cfg
    )
    return encoder
