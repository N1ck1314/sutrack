"""
Encoder modules: we use ITPN for the encoder with SMFA support.
"""

from torch import nn
import torch
from lib.utils.misc import is_main_process
from lib.models.sutrack import fastitpn as fastitpn_module
from lib.models.sutrack import itpn as oriitpn_module



class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int,
                 use_smfa: bool = False, smfa_num_heads: int = 6, smfa_mlp_ratio: float = 4.0):
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
        
        # 添加SMFA模块在encoder输出层
        self.use_smfa = use_smfa
        if self.use_smfa:
            from lib.models.sutrack_SMFA.smfa_modules import SMFABlock
            self.smfa_block = SMFABlock(
                dim=num_channels,
                num_heads=smfa_num_heads,
                mlp_ratio=smfa_mlp_ratio,
                qkv_bias=True,
                attn_drop=0.,
                proj_drop=0.,
                drop_path=0.1
            )
            print(f"[SMFA Encoder] SMFABlock initialized with {num_channels} channels, "
                  f"num_heads={smfa_num_heads}, mlp_ratio={smfa_mlp_ratio}")

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        
        # 应用SMFA模块增强search region特征
        if self.use_smfa and hasattr(self, 'smfa_block'):
            feature = xs[0]  # [B, N, C]
            B, N, C = feature.shape
            
            # 提取search region特征
            num_patches_search = self.body.num_patches_search
            num_patches_template = self.body.num_patches_template
            cls_token = self.body.cls_token is not None
            
            if cls_token:
                search_start_idx = 1
            else:
                search_start_idx = 0
            search_end_idx = search_start_idx + num_patches_search
            
            # 提取search特征
            search_features = feature[:, search_start_idx:search_end_idx, :]  # [B, num_patches_search, C]
            
            # 重塑为空间格式用于SMFA
            H = W = int(num_patches_search ** 0.5)
            if H * W == num_patches_search:
                search_spatial = search_features.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
                
                # 应用SMFA增强
                search_enhanced_spatial = self.smfa_block(search_spatial)  # [B, C, H, W]
                
                # 重塑回序列格式
                search_enhanced = search_enhanced_spatial.flatten(2).transpose(1, 2)  # [B, num_patches_search, C]
                
                # 拼接: [cls(optional)] + enhanced_search + template + [text(optional)]
                feature_before_search = feature[:, :search_start_idx, :]
                feature_after_search = feature[:, search_end_idx:, :]
                
                if search_start_idx > 0:
                    feature_enhanced = torch.cat([feature_before_search, search_enhanced, feature_after_search], dim=1)
                else:
                    feature_enhanced = torch.cat([search_enhanced, feature_after_search], dim=1)
                
                xs = [feature_enhanced]
        
        return xs


class Encoder(EncoderBase):
    """ViT encoder."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None):
        if "fastitpn" in name.lower():
            # 获取SMFA配置
            use_smfa = cfg.MODEL.ENCODER.get('USE_SMFA', False)
            smfa_num_heads = cfg.MODEL.ENCODER.get('SMFA_NUM_HEADS', 6)
            smfa_mlp_ratio = cfg.MODEL.ENCODER.get('SMFA_MLP_RATIO', 4.0)
            
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
                        use_smfa, smfa_num_heads, smfa_mlp_ratio)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder
