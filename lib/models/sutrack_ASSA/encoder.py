"""
Encoder modules with ASSA: 集成自适应稀疏注意力的编码器
Based on paper: "Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration"

核心改进:
1. 将标准 Self-Attention 替换为 ASSA (Adaptive Sparse Self-Attention)
2. 在 encoder 输出后应用 ASSA 进行特征增强
3. ASSA 通过双分支(稀疏+稠密)自适应平衡信息保留与噪声抑制
"""

import torch
from torch import nn
from lib.utils.misc import is_main_process
from lib.models.sutrack import fastitpn as fastitpn_module
from lib.models.sutrack import itpn as oriitpn_module
from .assa_modules import ASSA_TransformerBlock


class EncoderBase(nn.Module):
    """
    SUTrack编码器基类，集成 ASSA 自适应稀疏注意力
    """
    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, 
                 num_channels: int, use_assa: bool = True, assa_num_blocks: int = 2,
                 assa_num_heads: int = 8):
        super().__init__()
        open_blocks = open_layers[2:]
        open_items = open_layers[0:2]
        for name, parameter in encoder.named_parameters():
            if not train_encoder:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze == True:
                    parameter.requires_grad_(False)

        self.body = encoder
        self.num_channels = num_channels
        
        # 添加 ASSA 模块
        self.use_assa = use_assa
        if self.use_assa:
            # 计算特征图尺寸 (假设 search size=224, patch_size=16)
            num_patches_search = self.body.num_patches_search
            h = w = int(num_patches_search ** 0.5)
            
            # 创建多个 ASSA Transformer Blocks
            self.assa_blocks = nn.ModuleList([
                ASSA_TransformerBlock(
                    dim=num_channels,
                    num_heads=assa_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    h=h,
                    w=w,
                    drop_path=0.1 * (i / max(1, assa_num_blocks - 1))  # stochastic depth
                )
                for i in range(assa_num_blocks)
            ])
            
            print(f"[ASSA Encoder] ASSA模块初始化完成:")
            print(f"  - 通道数: {num_channels}")
            print(f"  - ASSA块数量: {assa_num_blocks}")
            print(f"  - 注意力头数: {assa_num_heads}")
            print(f"  - 特征图尺寸: {h}x{w}")

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        """
        前向传播
        在encoder输出后应用ASSA进行自适应稀疏注意力增强
        主要增强search region特征（检测目标区域）
        """
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        
        # 应用ASSA增强search region特征
        if self.use_assa:
            feature = xs[0].clone()  # 克隆避免 inplace 操作
            
            # 获取搜索区域的patch数量和尺寸
            num_patches_search = self.body.num_patches_search
            num_frames = 1  # 默认搜索帧数
            
            # 提取search region特征
            if self.body.cls_token is not None:
                # 有class token: [cls, template_patches, search_patches]
                cls_token = feature[:, 0:1]  # 保存 cls token
                template_feat = feature[:, 1:1 + self.body.num_patches_template]  # 保存 template
                search_feat = feature[:, 1 + self.body.num_patches_template:
                                     1 + self.body.num_patches_template + num_patches_search * num_frames]
            else:
                # 无class token: [template_patches, search_patches]
                template_feat = feature[:, :self.body.num_patches_template]  # 保存 template
                search_feat = feature[:, self.body.num_patches_template:
                                     self.body.num_patches_template + num_patches_search * num_frames]
            
            # 应用 ASSA Transformer Blocks
            for assa_block in self.assa_blocks:
                search_feat = assa_block(search_feat)
            
            # 重新组合特征（避免 inplace 操作）
            if self.body.cls_token is not None:
                feature = torch.cat([cls_token, template_feat, search_feat], dim=1)
            else:
                feature = torch.cat([template_feat, search_feat], dim=1)
            
            xs = (feature,) + xs[1:] if len(xs) > 1 else (feature,)
                
        return xs


class Encoder(EncoderBase):
    """ViT encoder with ASSA."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None,
                 use_assa: bool = True):
        if "fastitpn" in name.lower():
            encoder = getattr(fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type = cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init = cfg.MODEL.ENCODER.PATCHEMBED_INIT
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "oriitpn" in name.lower():
            encoder = getattr(oriitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_path_rate=0.1,
                init_values=0.1,
                use_mean_pooling=True,
                ape=True,
                rpe=True,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                task_num=cfg.MODEL.TASK_NUM,
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE
            )
            if "itpnb" in name:
                num_channels = 512
            else:
                num_channels = 512
        else:
            raise ValueError()
        
        # 从配置获取ASSA参数
        assa_num_blocks = getattr(cfg.MODEL, 'ASSA_NUM_BLOCKS', 2)
        assa_num_heads = getattr(cfg.MODEL, 'ASSA_NUM_HEADS', 8)
        
        super().__init__(encoder, train_encoder, open_layers, num_channels, use_assa, 
                        assa_num_blocks, assa_num_heads)


def build_encoder(cfg):
    """构建带ASSA的编码器"""
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    use_assa = getattr(cfg.MODEL, 'USE_ASSA', True)  # 默认使用ASSA
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg, use_assa)
    return encoder
