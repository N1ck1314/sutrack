"""
Encoder modules with Decoupled Spatio-Temporal Consistency Learning (DSCL)
Based on SSTrack: Decoupled Spatio-Temporal Consistency Learning for Self-Supervised Tracking
"""

import torch
from torch import nn
from lib.utils.misc import is_main_process
from lib.models.sutrack import fastitpn as fastitpn_module
from lib.models.sutrack import itpn as oriitpn_module
from .dscl import DecoupledSTConsistency, SSTrackLoss


class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int, cfg=None):
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
                    parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !

        self.body = encoder
        self.num_channels = num_channels
        self.cfg = cfg
        
        # DSCL模块配置
        self.use_dscl = getattr(cfg.MODEL, 'USE_DSCL', False)
        if self.use_dscl:
            self.dscl = DecoupledSTConsistency(
                dim=num_channels,
                num_frames=getattr(cfg.MODEL.DSCL, 'NUM_FRAMES', 2),
                spatial_heads=getattr(cfg.MODEL.DSCL, 'SPATIAL_HEADS', 8),
                temporal_heads=getattr(cfg.MODEL.DSCL, 'TEMPORAL_HEADS', 4),
                drop_path=getattr(cfg.MODEL.DSCL, 'DROP_PATH', 0.0)
            )
            print(f"✅ DSCL模块已启用 - 空间头数:{cfg.MODEL.DSCL.SPATIAL_HEADS}, 时间头数:{cfg.MODEL.DSCL.TEMPORAL_HEADS}")
        else:
            self.dscl = None

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        
        # 应用DSCL模块增强特征
        if self.use_dscl and self.dscl is not None and self.training:
            # xs可能是列表或张量，DSCL期望张量输入 (B, N, C)
            if isinstance(xs, list):
                # 如果是列表，取第一个元素（通常是主要特征）
                xs_tensor = xs[0] if len(xs) > 0 else xs
                xs_enhanced, aux_dict = self.dscl(xs_tensor)
                # 将增强后的特征放回列表
                xs[0] = xs_enhanced
                return xs, aux_dict
            else:
                # xs: (B, N, C) - 包含template和search的拼接特征
                xs_enhanced, aux_dict = self.dscl(xs)
                return xs_enhanced, aux_dict
        
        return xs, {}


class Encoder(EncoderBase):
    """ViT encoder."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None):
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
        super().__init__(encoder, train_encoder, open_layers, num_channels, cfg)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder
