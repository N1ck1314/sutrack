"""
Encoder modules with CPAM: 集成通道和位置注意力的编码器
Based on paper: "ASF-YOLO: A novel YOLO model with attentional scale sequence fusion for cell instance segmentation"

核心改进:
1. 在 encoder 输出后应用 CPAM 进行特征增强
2. CPAM 联合通道注意力与位置注意力
3. 同时关注"看什么"（通道）和"看哪里"（位置）
"""

import torch
from torch import nn
from lib.utils.misc import is_main_process
from lib.models.sutrack import fastitpn as fastitpn_module
from lib.models.sutrack import itpn as oriitpn_module
from .cpam_modules import CPAM_SingleInput


class EncoderBase(nn.Module):
    """
    SUTrack编码器基类，集成 CPAM 通道和位置注意力
    """
    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, 
                 num_channels: int, use_cpam: bool = True, cpam_reduction: int = 16):
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
        
        # 添加 CPAM 模块
        self.use_cpam = use_cpam
        if self.use_cpam:
            self.cpam_module = CPAM_SingleInput(
                channel=num_channels,
                reduction=cpam_reduction
            )
            print(f"[CPAM Encoder] CPAM模块初始化完成:")
            print(f"  - 通道数: {num_channels}")
            print(f"  - 通道压缩比: {cpam_reduction}")
            print(f"  - 模式: 单输入增强")

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        """
        前向传播
        在encoder输出后应用CPAM进行通道+位置双重注意力增强
        主要增强search region特征（检测目标区域）
        """
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        
        # 应用CPAM增强search region特征
        if self.use_cpam:
            feature = xs[0].clone()  # 克隆避免 inplace 操作
            
            # 获取搜索区域的patch数量和尺寸
            num_patches_search = self.body.num_patches_search
            num_frames = 1  # 默认搜索帧数
            fx_sz = int(num_patches_search ** 0.5)
            
            # 提取search region特征
            if self.body.cls_token is not None:
                # 有class token: [cls, template_patches, search_patches]
                cls_token = feature[:, 0:1]
                template_feat = feature[:, 1:1 + self.body.num_patches_template]
                search_feat = feature[:, 1 + self.body.num_patches_template:
                                     1 + self.body.num_patches_template + num_patches_search * num_frames]
            else:
                # 无class token: [template_patches, search_patches]
                template_feat = feature[:, :self.body.num_patches_template]
                search_feat = feature[:, self.body.num_patches_template:
                                     self.body.num_patches_template + num_patches_search * num_frames]
            
            bs, HW, C = search_feat.size()
            
            # 重塑为2D特征图用于CPAM处理
            search_feat_2d = search_feat.permute(0, 2, 1).contiguous().view(bs, C, fx_sz, fx_sz)
            
            # 应用CPAM模块增强（通道+位置双重注意力）
            search_feat_2d = self.cpam_module(search_feat_2d)
            
            # 重塑回序列格式
            search_feat = search_feat_2d.view(bs, C, -1).permute(0, 2, 1).contiguous()
            
            # 重新组合特征（避免 inplace 操作）
            if self.body.cls_token is not None:
                feature = torch.cat([cls_token, template_feat, search_feat], dim=1)
            else:
                feature = torch.cat([template_feat, search_feat], dim=1)
            
            xs = (feature,) + xs[1:] if len(xs) > 1 else (feature,)
                
        return xs


class Encoder(EncoderBase):
    """ViT encoder with CPAM."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None,
                 use_cpam: bool = True):
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
        
        # 从配置获取CPAM参数
        cpam_reduction = getattr(cfg.MODEL, 'CPAM_REDUCTION', 16)
        
        super().__init__(encoder, train_encoder, open_layers, num_channels, use_cpam, cpam_reduction)


def build_encoder(cfg):
    """构建带CPAM的编码器"""
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    use_cpam = getattr(cfg.MODEL, 'USE_CPAM', True)  # 默认使用CPAM
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg, use_cpam)
    return encoder
