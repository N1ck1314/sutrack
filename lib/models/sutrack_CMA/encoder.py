"""
Encoder modules with CMA: we use ITPN for the encoder and add CMA modules.
"""

import torch
from torch import nn
from lib.utils.misc import is_main_process
from lib.models.sutrack import fastitpn as fastitpn_module
from lib.models.sutrack import itpn as oriitpn_module
from .cma import CMA_Module


class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int, use_cma: bool = True):
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
        
        # 添加CMA模块
        self.use_cma = use_cma
        if self.use_cma:
            self.cma_module = CMA_Module(
                in_channel=num_channels,
                hidden_channel=num_channels // 2,
                out_channel=num_channels
            )
            print(f"[CMA Encoder] CMA Module initialized with {num_channels} channels")

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        
        # 在encoder输出后应用CMA模块
        # 只对search region特征应用CMA（参考MLKA的实现）
        if self.use_cma:
            feature = xs[0]  # (B, N, C) where N = 1(cls) + num_template + num_search
            
            # 获取搜索区域的patch数量和尺寸
            num_patches_search = self.body.num_patches_search
            num_frames = 1  # 默认搜索帧数
            fx_sz = int(num_patches_search ** 0.5)
            
            # 提取search region特征
            if self.body.cls_token is not None:
                # 有class token: [cls, template_patches, search_patches]
                search_feat = feature[:, 1 + self.body.num_patches_template:1 + self.body.num_patches_template + num_patches_search * num_frames]
            else:
                # 无class token: [template_patches, search_patches]
                search_feat = feature[:, self.body.num_patches_template:self.body.num_patches_template + num_patches_search * num_frames]
            
            bs, HW, C = search_feat.size()
            
            # 重塑为2D特征图用于CMA处理
            search_feat_2d = search_feat.permute(0, 2, 1).contiguous().view(bs, C, fx_sz, fx_sz)
            
            # 应用CMA模块增强
            search_feat_2d = self.cma_module(search_feat_2d)
            
            # 重塑回序列格式
            search_feat = search_feat_2d.view(bs, C, -1).permute(0, 2, 1).contiguous()
            
            # 替换增强后的search特征
            if self.body.cls_token is not None:
                feature[:, 1 + self.body.num_patches_template:1 + self.body.num_patches_template + num_patches_search * num_frames] = search_feat
            else:
                feature[:, self.body.num_patches_template:self.body.num_patches_template + num_patches_search * num_frames] = search_feat
            
            xs = (feature,) + xs[1:] if len(xs) > 1 else (feature,)
                
        return xs


class Encoder(EncoderBase):
    """ViT encoder with CMA."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None,
                 use_cma: bool = True):
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
        super().__init__(encoder, train_encoder, open_layers, num_channels, use_cma)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    use_cma = getattr(cfg.MODEL, 'USE_CMA', True)  # 默认使用CMA
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg, use_cma)
    return encoder
