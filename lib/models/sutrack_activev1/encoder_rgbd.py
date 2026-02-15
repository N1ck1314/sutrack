"""
Encoder with Adaptive RGBD Fusion
支持动态深度选择的编码器 - 直接在encoder层级处理深度融合
不使用 Fast_iTPN_RGBD 包装器，而是在forward中手动调用原始模型的内部方法
"""
import inspect
import torch
from torch import nn
import torch.nn.functional as F
from lib.utils.misc import is_main_process
from . import fastitpn as fastitpn_module
from . import itpn as oriitpn_module
from .rgbd_dynamic_fusion import RGBDDynamicFusion, LayerwiseDepthGate


class EncoderBaseRGBD(nn.Module):
    """
    支持RGBD的编码器基类
    body 直接是原始 Fast_iTPN（不包装）
    """
    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
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
        forward_params = inspect.signature(self.body.forward).parameters
        self._supports_text_task = ("text_src" in forward_params) and ("task_index" in forward_params)

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        if self._supports_text_task:
            out = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        else:
            out = self.body(template_list, search_list, template_anno_list)

        if isinstance(out, tuple):
            xs, probs_active = out
        else:
            xs, probs_active = out, None
        return xs, probs_active


class EncoderRGBD(EncoderBaseRGBD):
    """
    支持自适应RGBD融合的编码器
    核心思路：
    1. 使用原始 Fast_iTPN 作为 body（不包装）
    2. 在 forward 中手动调用 body 的内部方法
    3. 在主blocks迭代中注入深度融合
    """
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None,
                 use_dynamic_depth=True):

        self.use_dynamic_depth = use_dynamic_depth

        if "fastitpn" in name.lower():
            # 构建原始编码器（不包装）
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
                patchembed_init=cfg.MODEL.ENCODER.PATCHEMBED_INIT
            )

            # 确定通道数
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

        super().__init__(encoder, train_encoder, open_layers, num_channels)

        # 添加深度融合模块（作为encoder的成员，不修改body）
        if use_dynamic_depth:
            embed_dim = encoder.embed_dim
            depth_dim = embed_dim // 2  # 深度特征维度 = embed_dim * depth_dim_ratio(0.5)

            # 深度特征提取器：3ch depth -> depth_dim tokens
            # 总stride需要匹配patch_embed的stride，使token数量一致
            # Fast_iTPN patch_embed: stride=16 -> search 224/16=14 -> 196 tokens
            self.depth_patch_embed = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, depth_dim, kernel_size=3, stride=4, padding=1),
                nn.BatchNorm2d(depth_dim),
                nn.ReLU(inplace=True),
            )

            # 每个主block一个深度融合模块
            num_main_blocks = encoder.num_main_blocks
            self.depth_fusion_layers = nn.ModuleList([
                RGBDDynamicFusion(embed_dim, num_heads=8, depth_dim_ratio=0.5)
                for _ in range(num_main_blocks)
            ])
            # 一个共享的层间门控（跨层传递隐藏状态）
            self.depth_gate = LayerwiseDepthGate(num_main_blocks, embed_dim)

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        """
        处理RGBD输入，支持动态深度融合
        输入格式: [B, 6, H, W] (RGB + Depth) 或 [B, 3, H, W] (RGB only)
        注意: patch_embed 使用 halfcopy 初始化，期望6通道(RGBD)输入
        """
        # 检查是否有深度通道
        has_depth = search_list[0].size(1) > 3

        if not self.use_dynamic_depth or not has_depth:
            # 无深度融合 - 直接传完整输入（包括6ch RGBD）
            return super().forward(template_list, search_list, template_anno_list, text_src, task_index)

        # === 有深度通道且启用动态融合 ===

        # 1. 单独提取深度通道用于融合模块
        search_depth = [s[:, 3:] for s in search_list]
        num_search = len(search_list)
        sd = torch.stack(search_depth, dim=1)  # (B, N_search, 3, H, W)
        B = sd.size(0)
        sd = sd.view(-1, *sd.size()[2:])  # (B*N_search, 3, H, W)
        depth_feat = self.depth_patch_embed(sd)  # (B*N_search, C_depth, H', W')
        depth_feat = depth_feat.flatten(2).transpose(1, 2)  # (B*N_search, L, C_depth)
        # 重组为 (B, N_search*L, C_depth)
        depth_feat = depth_feat.view(B, num_search, depth_feat.size(1), depth_feat.size(2))
        depth_feat = depth_feat.reshape(B, -1, depth_feat.size(-1))  # (B, N_search*L, C_depth)

        # 2. 使用原始模型的 prepare_tokens_with_masks 处理完整6ch RGBD输入
        #    patch_embed 使用 halfcopy 初始化，天然支持6通道输入
        xz = self.body.prepare_tokens_with_masks(
            template_list, search_list, template_anno_list, text_src, task_index
        )
        xz = self.body.pos_drop(xz)

        rel_pos_bias = self.body.rel_pos_bias() if self.body.rel_pos_bias is not None else None
        probs_active = []

        # 4. 确定search token在xz中的位置
        # xz = [cls_token?, search_tokens, template_tokens, text_tokens?]
        cls_offset = 1 if self.body.cls_token is not None else 0
        num_search_tokens = num_search * self.body.num_patches_search

        # 5. 主blocks迭代 + 深度融合
        main_blocks = self.body.blocks[-self.body.num_main_blocks:]
        hidden = None

        for i, blk in enumerate(main_blocks):
            # 原始block处理（包括动态激活）
            use_dynamic = (i >= 2)
            xz, prob_active = blk(xz, rel_pos_bias, dynamic_activation=use_dynamic)
            if prob_active is not None:
                probs_active.append(prob_active)

            # 深度融合 - 只对search tokens
            search_start = cls_offset
            search_end = cls_offset + num_search_tokens
            search_tokens = xz[:, search_start:search_end, :]  # (B, N_search_tokens, C)

            gate, hidden = self.depth_gate(i, search_tokens, hidden)
            
            # 深度融合 - 始终尝试融合（gate控制融合强度）
            try:
                fused, decision = self.depth_fusion_layers[i](
                    search_tokens, depth_feat, return_decision=True
                )
                gate_expanded = gate.unsqueeze(1)  # (B, 1, 1)
                search_tokens_fused = search_tokens * (1 - gate_expanded) + fused * gate_expanded

                # 拼回完整的xz
                xz = torch.cat([
                    xz[:, :search_start, :],
                    search_tokens_fused,
                    xz[:, search_end:, :]
                ], dim=1)
                
                # 记录深度使用情况（用于损失计算）
                if probs_active:
                    # 将深度使用概率附加到最后一个prob_active
                    depth_prob = torch.tensor([decision['use_depth_prob']], device=search_tokens.device).expand(search_tokens.size(0), 1)
                    probs_active[-1] = torch.cat([probs_active[-1], depth_prob], dim=1)
            except Exception as e:
                # 如果融合失败，保持原始tokens
                print(f"[WARN] Depth fusion failed at layer {i}: {e}")

        # 6. 归一化
        xz = self.body.norm(xz)
        if hasattr(self.body, 'fc_norm') and self.body.fc_norm is not None:
            xz = self.body.fc_norm(xz)

        return [xz], probs_active if len(probs_active) > 0 else None


def build_encoder_rgbd(cfg, use_dynamic_depth=True):
    """
    构建支持RGBD的编码器
    """
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = EncoderRGBD(
        cfg.MODEL.ENCODER.TYPE,
        train_encoder,
        cfg.DATA.SEARCH.SIZE,
        cfg.DATA.TEMPLATE.SIZE,
        cfg.TRAIN.ENCODER_OPEN,
        cfg,
        use_dynamic_depth=use_dynamic_depth
    )
    return encoder


# 兼容性：保留原函数名
def build_encoder(cfg):
    """默认使用RGBD编码器"""
    use_dynamic_depth = getattr(cfg.MODEL, 'USE_DYNAMIC_DEPTH', True)
    return build_encoder_rgbd(cfg, use_dynamic_depth)
