"""
SUTrack Model with MLKA (Multi-scale Large Kernel Attention)
Integrates MLKA for enhanced multi-scale feature extraction in multi-modal tracking
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
from .clip import build_textencoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder
from .mlka import MLKA, MLKAFeatureEnhancement
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class SUTRACK(nn.Module):
    """ SUTrack with MLKA for enhanced multi-scale feature extraction """
    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average",
                 use_mlka=True, mlka_position="decoder", mlka_blocks=1):
        """ 
        Initializes the model.
        
        Args:
            use_mlka (bool): Whether to use MLKA enhancement
            mlka_position (str): Where to apply MLKA - "encoder", "decoder", or "both"
            mlka_blocks (int): Number of MLKA blocks to use
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type
        self.use_mlka = use_mlka
        self.mlka_position = mlka_position

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
        
        # MLKA Feature Enhancement Modules
        if self.use_mlka:
            encoder_dim = encoder.num_channels
            print(f"[MLKA] Initializing MLKA modules at position: {mlka_position}")
            print(f"[MLKA] Encoder dim: {encoder_dim}, Blocks: {mlka_blocks}")
            
            # MLKA after encoder (operates on search features before decoder)
            if mlka_position in ["encoder", "both"]:
                self.mlka_encoder = MLKAFeatureEnhancement(
                    dim=encoder_dim, 
                    num_blocks=mlka_blocks
                )
                print(f"[MLKA] Encoder enhancement initialized")
            
            # MLKA in decoder pathway (enhances decoder input features)
            if mlka_position in ["decoder", "both"]:
                self.mlka_decoder = MLKAFeatureEnhancement(
                    dim=encoder_dim,
                    num_blocks=mlka_blocks
                )
                print(f"[MLKA] Decoder enhancement initialized")


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
        # Forward the encoder
        text_src = self.text_encoder(text_data)
        return text_src

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        # Forward the encoder
        xz = self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        
        # Apply MLKA enhancement after encoder if enabled
        if self.use_mlka and self.mlka_position in ["encoder", "both"]:
            # Extract search region features for enhancement
            feature = xz[0]  # (B, N, C)
            
            if self.class_token:
                search_feat = feature[:, 1:self.num_patch_x * self.num_frames + 1]
            else:
                search_feat = feature[:, 0:self.num_patch_x * self.num_frames]
            
            bs, HW, C = search_feat.size()
            # Reshape to 2D feature map for MLKA
            search_feat_2d = search_feat.permute(0, 2, 1).contiguous().view(bs, C, self.fx_sz, self.fx_sz)
            
            # Apply MLKA enhancement
            search_feat_2d = self.mlka_encoder(search_feat_2d)
            
            # Reshape back to sequence
            search_feat = search_feat_2d.view(bs, C, -1).permute(0, 2, 1).contiguous()
            
            # Replace enhanced search features
            if self.class_token:
                feature[:, 1:self.num_patch_x * self.num_frames + 1] = search_feat
            else:
                feature[:, 0:self.num_patch_x * self.num_frames] = search_feat
            
            xz = (feature,) + xz[1:] if len(xz) > 1 else (feature,)
        
        return xz

    def forward_decoder(self, feature, gt_score_map=None):
        feature = feature[0]
        
        # Extract search features
        if self.class_token:
            feature = feature[:, 1:self.num_patch_x * self.num_frames + 1]
        else:
            feature = feature[:, 0:self.num_patch_x * self.num_frames]  # (B, HW, C)

        bs, HW, C = feature.size()
        
        # Reshape to 2D for decoder
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        
        # Apply MLKA enhancement before decoder if enabled
        if self.use_mlka and self.mlka_position in ["decoder", "both"]:
            feature = self.mlka_decoder(feature)
        
        # Run decoder
        if self.decoder_type == "CORNER":
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map}
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


def build_sutrack_mlka(cfg):
    """Build SUTrack with MLKA enhancement"""
    encoder = build_encoder(cfg)
    
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, encoder)
    else:
        text_encoder = None
    
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    
    # MLKA configuration
    use_mlka = cfg.MODEL.get('USE_MLKA', True)
    mlka_position = cfg.MODEL.get('MLKA_POSITION', 'decoder')  # "encoder", "decoder", "both"
    mlka_blocks = cfg.MODEL.get('MLKA_BLOCKS', 1)
    
    model = SUTRACK(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE,
        use_mlka=use_mlka,
        mlka_position=mlka_position,
        mlka_blocks=mlka_blocks
    )
    
    print("\n" + "="*60)
    print("SUTrack-MLKA Model Summary")
    print("="*60)
    print(f"MLKA Enabled: {use_mlka}")
    if use_mlka:
        print(f"MLKA Position: {mlka_position}")
        print(f"MLKA Blocks: {mlka_blocks}")
    print(f"Encoder: {cfg.MODEL.ENCODER.TYPE}")
    print(f"Decoder: {cfg.MODEL.DECODER.TYPE}")
    print(f"Total Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("="*60 + "\n")
    
    return model
