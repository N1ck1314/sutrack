"""
Encoder modules: we use ITPN for the encoder with ORR support.
"""

from torch import nn
import torch
from lib.utils.misc import is_main_process
from . import fastitpn as fastitpn_module
from . import itpn as oriitpn_module



class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int,
                 use_orr: bool = False, orr_mask_ratio: float = 0.3, orr_mask_strategy: str = 'cox',
                 orr_loss_weight: float = 0.5):
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
        
        # 添加ORR模块
        self.use_orr = use_orr
        if self.use_orr:
            from lib.models.sutrack_OR.orr_modules import OcclusionRobustEncoder
            self.orr_module = OcclusionRobustEncoder(
                use_orr=True,
                mask_ratio=orr_mask_ratio,
                mask_strategy=orr_mask_strategy,
                invariance_loss_weight=orr_loss_weight
            )
            print(f"[ORR Encoder] ORR module initialized: mask_ratio={orr_mask_ratio}, "
                  f"strategy={orr_mask_strategy}, loss_weight={orr_loss_weight}")
        
        self.orr_loss = None  # 存储ORR损失

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        # 标准前向传播
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        
        # 应用ORR（仅在训练时）
        if self.use_orr and self.training:
            feature = xs[0]  # [B, N, C]
            B, N, C = feature.shape
            
            # 提取search region特征
            num_patches_search = self.body.num_patches_search
            cls_token = self.body.cls_token is not None
            
            if cls_token:
                search_start_idx = 1
            else:
                search_start_idx = 0
            search_end_idx = search_start_idx + num_patches_search
            
            # 提取search特征
            search_features = feature[:, search_start_idx:search_end_idx, :]  # [B, num_patches_search, C]
            
            # 计算空间维度
            H = W = int(num_patches_search ** 0.5)
            
            if H * W == num_patches_search:
                # 保存干净特征
                clean_search_features = search_features.clone()
                
                # 应用ORR遮挡模拟
                masked_search_features, _ = self.orr_module(
                    search_features, H, W, training=True
                )
                
                # 计算特征不变性损失
                self.orr_loss = self.orr_module.compute_invariance_loss(
                    clean_search_features, masked_search_features
                )
                
                # 使用遮挡后的特征替换原始search特征
                feature_before = feature[:, :search_start_idx, :]
                feature_after = feature[:, search_end_idx:, :]
                
                if search_start_idx > 0:
                    feature_new = torch.cat([feature_before, masked_search_features, feature_after], dim=1)
                else:
                    feature_new = torch.cat([masked_search_features, feature_after], dim=1)
                
                xs = [feature_new]
        
        return xs
    
    def get_orr_loss(self):
        """获取ORR损失用于训练"""
        return self.orr_loss if hasattr(self, 'orr_loss') else None


class Encoder(EncoderBase):
    """ViT encoder."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None):
        if "fastitpn" in name.lower():
            # 获取ORR配置
            use_orr = cfg.MODEL.ENCODER.get('USE_ORR', False)
            orr_mask_ratio = cfg.MODEL.ENCODER.get('ORR_MASK_RATIO', 0.3)
            orr_mask_strategy = cfg.MODEL.ENCODER.get('ORR_MASK_STRATEGY', 'cox')
            orr_loss_weight = cfg.MODEL.ENCODER.get('ORR_LOSS_WEIGHT', 0.5)
            
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
        super().__init__(encoder, train_encoder, open_layers, num_channels,
                        use_orr, orr_mask_ratio, orr_mask_strategy, orr_loss_weight)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder
