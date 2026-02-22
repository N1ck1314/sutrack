"""
Encoder modules for SGLA-RGBD: 集成SGLA-inspired RGBD fusion
"""

from torch import nn
from lib.utils.misc import is_main_process
from . import fastitpn as fastitpn_module
from . import itpn as oriitpn_module
from .sgla_rgbd_modules import SGLA_RGBD_Encoder


class EncoderBase(nn.Module):
    """SGLA-RGBD编码器基类"""
    
    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int,
                 use_sgla_rgbd: bool = False, sgla_rgbd_config: dict = None):
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
                if freeze:
                    parameter.requires_grad_(False)
        
        self.body = encoder
        self.num_channels = num_channels
        self.use_sgla_rgbd = use_sgla_rgbd
        
        # SGLA Loss support (保留原有SGLA功能)
        if getattr(self.body, 'use_sgla', False):
            from .sgla_modules import SimilarityLoss
            self.sgla_loss_fn = SimilarityLoss()
        
        # SGLA-RGBD模块集成
        if use_sgla_rgbd:
            if sgla_rgbd_config is None:
                sgla_rgbd_config = {}
            
            # 获取encoder维度
            if hasattr(encoder, 'embed_dim'):
                dim = encoder.embed_dim
            elif hasattr(encoder, 'num_features'):
                dim = encoder.num_features
            else:
                dim = num_channels
            
            # 获取层数
            if hasattr(encoder, 'num_main_blocks'):
                num_layers = encoder.num_main_blocks
            elif hasattr(encoder, 'depth'):
                num_layers = encoder.depth
            else:
                num_layers = 12  # 默认值
            
            self.sgla_rgbd_encoder = SGLA_RGBD_Encoder(
                base_encoder=encoder,
                dim=dim,
                num_layers=num_layers,
                **sgla_rgbd_config
            )
            print(f"✓ SGLA-RGBD Encoder initialized: dim={dim}, layers={num_layers}")
    
    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        if self.use_sgla_rgbd:
            # 使用SGLA-RGBD融合
            xs, aux_info = self.sgla_rgbd_encoder(
                template_list, search_list, template_anno_list, text_src, task_index
            )
            # 保存aux_info供损失计算使用
            self.sgla_rgbd_aux_info = aux_info
            return xs
        else:
            # 标准forward
            xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
            return xs
    
    def get_sgla_loss(self):
        """获取SGLA相似度损失"""
        if hasattr(self, 'sgla_loss_fn') and hasattr(self.body, 'layer_features'):
            if len(self.body.layer_features) > 1:
                return self.sgla_loss_fn(self.body.layer_features)
        return 0.0
    
    def get_sgla_rgbd_loss(self):
        """获取SGLA-RGBD互补性损失"""
        if self.use_sgla_rgbd and hasattr(self, 'sgla_rgbd_aux_info'):
            return self.sgla_rgbd_aux_info.get('complementarity_loss', 0.0)
        return 0.0
    
    def get_sgla_rgbd_stats(self):
        """获取SGLA-RGBD统计信息"""
        if self.use_sgla_rgbd:
            return self.sgla_rgbd_encoder.get_statistics()
        return {}


class Encoder(EncoderBase):
    """SGLA-RGBD ViT encoder"""
    
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None,
                 use_sgla_rgbd: bool = False):
        
        # 构建基础编码器
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
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init=cfg.MODEL.ENCODER.PATCHEMBED_INIT,
                use_sgla=cfg.MODEL.ENCODER.get('USE_SGLA', False)  # 保留原有SGLA
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
            raise ValueError(f"Unknown encoder type: {name}")
        
        # SGLA-RGBD配置
        sgla_rgbd_config = None
        if use_sgla_rgbd and hasattr(cfg.MODEL.ENCODER, 'SGLA_RGBD'):
            sgla_rgbd_config = {
                'use_modal_selection': cfg.MODEL.ENCODER.SGLA_RGBD.get('USE_MODAL_SELECTION', True),
                'use_layerwise_fusion': cfg.MODEL.ENCODER.SGLA_RGBD.get('USE_LAYERWISE_FUSION', True),
                'use_selective_depth': cfg.MODEL.ENCODER.SGLA_RGBD.get('USE_SELECTIVE_DEPTH', True),
                'use_complementarity_loss': cfg.MODEL.ENCODER.SGLA_RGBD.get('USE_COMPLEMENTARITY_LOSS', True)
            }
        
        super().__init__(encoder, train_encoder, open_layers, num_channels, 
                         use_sgla_rgbd=use_sgla_rgbd, sgla_rgbd_config=sgla_rgbd_config)


def build_encoder(cfg):
    """构建编码器"""
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    
    # 检查是否启用SGLA-RGBD
    use_sgla_rgbd = cfg.MODEL.ENCODER.get('USE_SGLA_RGBD', False)
    
    encoder = Encoder(
        cfg.MODEL.ENCODER.TYPE, 
        train_encoder,
        cfg.DATA.SEARCH.SIZE,
        cfg.DATA.TEMPLATE.SIZE,
        cfg.TRAIN.ENCODER_OPEN, 
        cfg,
        use_sgla_rgbd=use_sgla_rgbd
    )
    
    return encoder
