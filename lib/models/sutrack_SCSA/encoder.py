"""
Encoder modules: we use ITPN for the encoder.
"""

from torch import nn
import torch
from lib.utils.misc import is_main_process
from lib.models.sutrack_SCSA import fastitpn as fastitpn_module
from lib.models.sutrack import itpn as oriitpn_module



class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int, use_scsa=False, scsa_reduction_ratio=4):
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
        
        # Add SCSA module at encoder output
        self.use_scsa = use_scsa
        if use_scsa:
            from lib.models.sutrack_SCSA.scsa_modules import SCSA
            self.scsa = SCSA(
                dim=num_channels,
                group_kernel_sizes=[3, 5, 7, 9],
                gate_layer='sigmoid',
                reduction_ratio=scsa_reduction_ratio
            )
        else:
            self.scsa = None

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        
        # Apply SCSA to search region features if enabled
        if self.use_scsa and self.scsa is not None:
            # xs is a list with one element: [B, N, C] where N = num_patches_search + num_patches_template + other_tokens
            feature = xs[0]  # [B, N, C]
            B, N, C = feature.shape
            
            # Extract search region features
            # Assuming: [cls_token(optional)] + search_tokens + template_tokens + [text_token(optional)]
            num_patches_search = self.body.num_patches_search
            num_patches_template = self.body.num_patches_template
            cls_token = self.body.cls_token is not None
            
            if cls_token:
                search_start_idx = 1
            else:
                search_start_idx = 0
            search_end_idx = search_start_idx + num_patches_search
            
            # Extract search features
            search_features = feature[:, search_start_idx:search_end_idx, :]  # [B, num_patches_search, C]
            
            # Reshape to spatial format for SCSA: [B, num_patches_search, C] -> [B, C, H, W]
            H = W = int(num_patches_search ** 0.5)
            if H * W == num_patches_search:
                search_spatial = search_features.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
                
                # Apply SCSA
                search_spatial_enhanced = self.scsa(search_spatial)  # [B, C, H, W]
                
                # Reshape back to sequence format
                search_enhanced = search_spatial_enhanced.reshape(B, C, num_patches_search).transpose(1, 2)  # [B, num_patches_search, C]
                
                # Replace search features by creating a new tensor (avoid in-place operation)
                # Split feature into parts and concatenate with enhanced search features
                feature_before_search = feature[:, :search_start_idx, :]  # [B, search_start_idx, C]
                feature_after_search = feature[:, search_end_idx:, :]  # [B, remaining, C]
                
                # Concatenate: [cls_token(optional)] + enhanced_search + template + [text_token(optional)]
                if search_start_idx > 0:  # Has cls_token
                    feature_enhanced = torch.cat([feature_before_search, search_enhanced, feature_after_search], dim=1)
                else:  # No cls_token
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
            # Get SCSA configuration
            use_scsa = cfg.MODEL.ENCODER.get('USE_SCSA', False)
            scsa_reduction_ratio = cfg.MODEL.ENCODER.get('SCSA_REDUCTION_RATIO', 4)
            
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
                patchembed_init = cfg.MODEL.ENCODER.PATCHEMBED_INIT,
                use_scsa=use_scsa,
                scsa_reduction_ratio=scsa_reduction_ratio
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
        super().__init__(encoder, train_encoder, open_layers, num_channels, use_scsa, scsa_reduction_ratio)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder
