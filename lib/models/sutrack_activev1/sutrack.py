"""
SUTrack Active V1 Model with Feature Enhancement
集成特征增强模块的改进版本
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder_rgbd import build_encoder_rgbd, build_encoder
from .clip import build_textencoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder
from .feature_enhancement import CrossAttentionModule, CBAM, TaskAdaptiveModule
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class SUTRACK(nn.Module):
    """ This is the base class for SUTrack Active V1 with Feature Enhancement """
    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average",
                 use_cross_attention=True, use_cbam=True, use_task_adaptive=True,
                 num_tasks=5):
        """ Initializes the model with feature enhancement modules.
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.task_decoder = task_decoder
        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template
        
        # 获取特征维度
        dim = encoder.num_channels
        
        # 特征增强模块
        self.use_cross_attention = use_cross_attention
        self.use_cbam = use_cbam
        self.use_task_adaptive = use_task_adaptive
        
        if use_cross_attention:
            self.cross_attn = CrossAttentionModule(dim, num_heads=8)
        if use_cbam:
            self.cbam = CBAM(dim)
        if use_task_adaptive:
            self.task_adaptive = TaskAdaptiveModule(dim, num_tasks)


    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,
                text_src=None, task_index=None,
                feature=None, mode="encoder"):
        if mode == "text":
            return self.forward_textencoder(text_data)
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
        elif mode == "decoder":
            return self.forward_decoder(feature, task_index=task_index), self.forward_task_decoder(feature)
        else:
            raise ValueError

    def forward_textencoder(self, text_data):
        # Forward the encoder
        text_src = self.text_encoder(text_data)
        return text_src

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        # Forward the encoder, 返回特征和激活概率
        xz, probs_active = self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        return xz, probs_active

    def forward_decoder(self, feature, gt_score_map=None, task_index=None):
        """
        解码器前向传播，集成特征增强模块
        token顺序: [cls?, search, template, text?]
        """
        feature = feature[0]
        if isinstance(feature, list):
            feature = feature[0]
            
        # 分离 search 和 template 特征
        # token顺序: [cls?, search(num_patch_x*num_frames), template(num_patch_z*num_template), text?]
        cls_offset = 1 if self.class_token else 0
        search_start = cls_offset
        search_end = cls_offset + self.num_patch_x * self.num_frames
        template_start = search_end
        template_end = search_end + self.num_patch_z * self.num_template
        
        search_feat = feature[:, search_start:search_end]  # (B, num_search_patches, C)
        template_feat = feature[:, template_start:template_end]  # (B, num_template_patches, C)
        
        # 应用 Cross-Attention 增强 search 特征
        if self.use_cross_attention:
            search_feat = self.cross_attn(search_feat, template_feat)
        
        # 应用 Task Adaptive 模块
        if self.use_task_adaptive and task_index is not None:
            search_feat = self.task_adaptive(search_feat, task_index)
        
        # 提取 search 区域用于解码
        feature = search_feat  # (B, HW, C)
        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
            
        # 应用 CBAM 注意力机制
        if self.use_cbam:
            feature = self.cbam(feature)
            
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
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
        # 如果 feature 是 list，转换为 tensor
        if isinstance(feature, list):
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
    encoder = build_encoder(cfg)
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, encoder)
    else:
        text_encoder = None
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    model = SUTRACK(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE
    )

    return model

def build_sutrack_activev1(cfg):
    """
    Build function for SUTrack Active V1 with Feature Enhancement.
    集成特征增强模块和自适应RGBD深度融合的改进版本
    """
    # 使用RGBD编码器（支持动态深度融合）
    use_dynamic_depth = getattr(cfg.MODEL, 'USE_DYNAMIC_DEPTH', True)
    encoder = build_encoder_rgbd(cfg, use_dynamic_depth=use_dynamic_depth)
    
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, encoder)
    else:
        text_encoder = None
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    
    # 从配置中读取特征增强模块的开关
    use_cross_attention = getattr(cfg.MODEL, 'USE_CROSS_ATTENTION', True)
    use_cbam = getattr(cfg.MODEL, 'USE_CBAM', True)
    use_task_adaptive = getattr(cfg.MODEL, 'USE_TASK_ADAPTIVE', True)
    num_tasks = cfg.MODEL.TASK_NUM
    
    model = SUTRACK(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE,
        use_cross_attention=use_cross_attention,
        use_cbam=use_cbam,
        use_task_adaptive=use_task_adaptive,
        num_tasks=num_tasks
    )

    return model


# 保留原函数名以兼容旧代码
def build_sutrack_active(cfg):
    """兼容旧代码的别名"""
    return build_sutrack_activev1(cfg)


