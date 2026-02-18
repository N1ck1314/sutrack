"""
SUTRACK_ARV2 Model - Integrated with ARTrackV2 modules
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
from .clip import build_textencoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class SUTRACK_ARV2(nn.Module):
    """ SUTrack with ARTrackV2 integration for speedup and appearance evolution """
    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average",
                 use_artrackv2=False):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type
        self.use_artrackv2 = use_artrackv2

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
        
        # ARTrackV2状态：用于跨帧演化
        self.prev_trajectory_token = None
        self.prev_appearance_token = None


    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,
                text_src=None, task_index=None,
                feature=None, mode="encoder", 
                gt_bbox=None):
        if mode == "text":
            return self.forward_textencoder(text_data)
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
        elif mode == "decoder":
            return self.forward_decoder(feature, gt_bbox), self.forward_task_decoder(feature)
        else:
            raise ValueError

    def forward_textencoder(self, text_data):
        # Forward the encoder
        text_src = self.text_encoder(text_data)
        return text_src

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        # Forward the encoder
        xz, aux_dict = self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        return xz, aux_dict

    def forward_decoder(self, feature, gt_bbox=None):
        # 处理ARTrackV2的返回格式
        aux_dict = {}
        if isinstance(feature, tuple):
            feature, aux_dict = feature
        
        # 如果启用ARTrackV2的pure encoder，使用其预测
        if self.use_artrackv2 and aux_dict.get('use_artrackv2', False):
            pure_encoder = aux_dict.get('pure_encoder')
            if pure_encoder is not None:
                # 提取search特征
                xz = feature[0] if isinstance(feature, list) else feature
                
                if self.class_token:
                    search_feature = xz[:, 1:self.num_patch_x * self.num_frames+1]
                else:
                    search_feature = xz[:, 0:self.num_patch_x * self.num_frames]
                
                # 提取template特征（用于外观重建）
                if self.class_token:
                    template_feature = xz[:, self.num_patch_x * self.num_frames+1:]
                else:
                    template_feature = xz[:, self.num_patch_x * self.num_frames:]
                
                # 使用Pure Encoder进行预测
                bbox, confidence, arv2_aux = pure_encoder(
                    search_features=search_feature,
                    prev_trajectory=self.prev_trajectory_token,
                    prev_appearance=self.prev_appearance_token,
                    target_features=template_feature
                )
                
                # 更新状态用于下一帧
                self.prev_trajectory_token = arv2_aux.get('trajectory_token')
                self.prev_appearance_token = arv2_aux.get('appearance_token')
                
                # 计算IoU loss（训练时）
                if self.training and gt_bbox is not None:
                    iou_loss = pure_encoder.confidence_module.compute_iou_loss(
                        confidence, gt_bbox, bbox
                    )
                    arv2_aux['iou_loss'] = iou_loss
                
                # 构造输出（与原decoder格式兼容）
                bs = bbox.shape[0]
                outputs_coord = bbox  # 已经是 [B, 4] 格式
                outputs_coord_new = outputs_coord.view(bs, 1, 4)
                
                out = {
                    'pred_boxes': outputs_coord_new,
                    'confidence': confidence,
                    'arv2_aux': arv2_aux
                }
                return out
        
        # 使用原始decoder流程
        feature = feature[0] if isinstance(feature, list) else feature
        if self.class_token:
            feature = feature[:,1:self.num_patch_x * self.num_frames+1]
        else:
            feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
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
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, None)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, None)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_task_decoder(self, feature):
        # 处理元组格式
        if isinstance(feature, tuple):
            feature = feature[0]
        
        feature = feature[0] if isinstance(feature, list) else feature
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
    
    def reset_arv2_state(self):
        """重置ARTrackV2的跨帧状态（用于新序列）"""
        self.prev_trajectory_token = None
        self.prev_appearance_token = None

def build_sutrack_arv2(cfg):
    encoder = build_encoder(cfg)
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, encoder)
    else:
        text_encoder = None
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    
    # 检查是否启用ARTrackV2
    use_artrackv2 = cfg.MODEL.ARTRACKV2.ENABLE if hasattr(cfg.MODEL, 'ARTRACKV2') else False
    
    model = SUTRACK_ARV2(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE,
        use_artrackv2=use_artrackv2
    )

    return model

