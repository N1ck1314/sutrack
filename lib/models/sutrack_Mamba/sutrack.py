"""
SUTrack Model with Mamba
集成 Mamba 模块的 SUTrack，用于高效的上下文信息传递

核心创新:
1. 在 Encoder 之后添加 Mamba Neck 模块
2. 使用隐藏状态高效传递上下文信息
3. 线性复杂度的序列建模

参考: MCITrack (AAAI 2025)
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
from .clip import build_textencoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder
from .mamba_module import SimpleMambaFusion, MambaNeck
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class SUTRACK_Mamba(nn.Module):
    """
    SUTrack with Mamba: 集成 Mamba 模块进行上下文信息增强
    
    架构:
    1. Encoder: ViT-based 特征提取
    2. Mamba Fusion: 使用 SSM 进行上下文信息融合
    3. Decoder: 边界框预测
    """
    def __init__(self, text_encoder, encoder, decoder, task_decoder, mamba_fusion,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average",
                 use_mamba=True):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type
        self.use_mamba = use_mamba

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.task_decoder = task_decoder
        self.decoder = decoder
        
        # Mamba 融合模块
        self.mamba_fusion = mamba_fusion

        self.num_frames = num_frames
        self.num_template = num_template

    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,
                text_src=None, task_index=None,
                feature=None, mode="encoder"):
        if mode == "text":
            return self.forward_textencoder(text_data)
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
        elif mode == "decoder":
            return self.forward_decoder(feature), self.forward_task_decoder(feature)
        else:
            raise ValueError

    def forward_textencoder(self, text_data):
        text_src = self.text_encoder(text_data)
        return text_src

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        # Forward the encoder
        xz = self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        
        # 应用 Mamba 融合
        if self.use_mamba and self.mamba_fusion is not None:
            # xz 是 list/tuple: [feature, ...]
            # 只对第一个元素（特征）应用 Mamba
            if isinstance(xz, (tuple, list)):
                feature = xz[0]
                # Mamba 融合
                feature = self.mamba_fusion(feature)
                # 重构返回值
                if isinstance(xz, tuple):
                    xz = (feature,) + xz[1:]
                else:
                    xz = [feature] + list(xz[1:])
            else:
                xz = self.mamba_fusion(xz)
        
        return xz

    def forward_decoder(self, feature, gt_score_map=None):
        feature = feature[0]
        if self.class_token:
            feature = feature[:, 1:self.num_patch_x * self.num_frames + 1]
        else:
            feature = feature[:, 0:self.num_patch_x * self.num_frames]  # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_task_decoder(self, feature):
        feature = feature[0]
        if self.task_feature_type == 'class':
            feature = feature[:, 0:1]
        elif self.task_feature_type == 'text':
            feature = feature[:, -1:]
        elif self.task_feature_type == 'average':
            feature = feature.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError('task_feature_type must be choosen from class, text, and average')
        feature = self.task_decoder(feature)
        return feature


def build_sutrack(cfg):
    """构建 SUTrack-Mamba 模型"""
    encoder = build_encoder(cfg)
    
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, encoder)
    else:
        text_encoder = None
    
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    
    # 构建 Mamba 融合模块
    use_mamba = getattr(cfg.MODEL, 'USE_MAMBA', True)
    if use_mamba:
        n_layers = getattr(cfg.MODEL, 'MAMBA_LAYERS', 2)
        d_state = getattr(cfg.MODEL, 'MAMBA_D_STATE', 16)
        mamba_fusion = SimpleMambaFusion(
            d_model=encoder.num_channels,
            n_layers=n_layers,
            d_state=d_state
        )
        print(f"\n✅ Mamba Fusion 模块已启用")
        print(f"   - 层数: {n_layers}")
        print(f"   - 状态维度: {d_state}")
        print(f"   - 特征维度: {encoder.num_channels}")
        mamba_params = sum(p.numel() for p in mamba_fusion.parameters())
        print(f"   - Mamba 参数量: {mamba_params / 1e6:.2f}M\n")
    else:
        mamba_fusion = None
        print("\n⚠️  Mamba Fusion 未启用\n")
    
    model = SUTRACK_Mamba(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        mamba_fusion,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE,
        use_mamba=use_mamba
    )

    return model
