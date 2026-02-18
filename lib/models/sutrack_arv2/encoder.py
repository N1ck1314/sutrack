"""
Encoder modules: we use ITPN for the encoder.
Integrated with ARTrackV2 modules for speedup and appearance evolution.
"""

from torch import nn
from lib.utils.misc import is_main_process
from lib.models.sutrack import fastitpn as fastitpn_module
from lib.models.sutrack import itpn as oriitpn_module
from .artrackv2_modules import PureEncoderDecoder



class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int, 
                 use_artrackv2=False, cfg=None):
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
        
        # ARTrackV2集成
        self.use_artrackv2 = use_artrackv2
        self.pure_encoder_decoder = None
        if use_artrackv2:
            self.pure_encoder_decoder = PureEncoderDecoder(
                dim=num_channels,
                num_trajectory_tokens=4,
                num_appearance_tokens=cfg.MODEL.ARTRACKV2.NUM_APPEARANCE_TOKENS if cfg else 4,
                num_heads=8,
                drop_path=0.1
            )
            print(f"[ARTrackV2] Initialized with {cfg.MODEL.ARTRACKV2.NUM_APPEARANCE_TOKENS if cfg else 4} appearance tokens")

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        
        # 如果启用ARTrackV2，返回带有元信息的字典
        if self.use_artrackv2 and self.pure_encoder_decoder is not None:
            return xs, {'use_artrackv2': True, 'pure_encoder': self.pure_encoder_decoder}
        
        return xs, {}


class Encoder(EncoderBase):
    """ViT encoder with ARTrackV2 integration."""
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
        
        # 检查是否启用ARTrackV2
        use_artrackv2 = cfg.MODEL.ARTRACKV2.ENABLE if hasattr(cfg.MODEL, 'ARTRACKV2') else False
        super().__init__(encoder, train_encoder, open_layers, num_channels, use_artrackv2, cfg)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder
