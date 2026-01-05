"""
Encoder modules with MFEblock: we use ITPN for the encoder and add MFEblock modules for multi-scale feature extraction.
Based on SHISRCNet paper: Multi-scale feature extraction with attention-based fusion for enhanced tracking.
"""

import torch
from torch import nn
from lib.utils.misc import is_main_process
from lib.models.sutrack import fastitpn as fastitpn_module
from lib.models.sutrack import itpn as oriitpn_module


class oneConv(nn.Module):
    """卷积层加激活函数的封装"""
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations, bias=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ASPPConv(nn.Sequential):
    """ASPP卷积模块，包含卷积、批归一化和ReLU激活"""
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class MFEblock(nn.Module):
    """
    Multi-scale Feature Extraction Block (MFEblock)
    基于SHISRCNet论文的多尺度特征提取模块
    
    核心功能：
    1. 多尺度感受野建模（使用不同dilation rate的空洞卷积）
    2. 多尺度特征的选择性融合（MSF模块）
    3. 残差连接保证稳定训练
    
    参数:
        in_channels: 输入特征通道数
        atrous_rates: 空洞卷积的膨胀率列表 [rate1, rate2, rate3]
    """
    def __init__(self, in_channels, atrous_rates):
        super(MFEblock, self).__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = tuple(atrous_rates)

        # 多尺度特征提取分支
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        self.layer4 = ASPPConv(in_channels, out_channels, rate3)

        # 投影层
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
        # 全局平均池化用于特征聚合
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        # 通道注意力模块 (Multi-scale Selective Fusion - MSF)
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE3 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE4 = oneConv(in_channels, in_channels, 1, 0, 1)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            增强后的特征图 (B, C, H, W)
        """
        # 多尺度特征提取（逐级累加输入，构建特征金字塔）
        y0 = self.layer1(x)  # 基础特征 (dilation=1)
        y1 = self.layer2(y0 + x)  # 中等感受野 (dilation=rate1)
        y2 = self.layer3(y1 + x)  # 较大感受野 (dilation=rate2)
        y3 = self.layer4(y2 + x)  # 最大感受野 (dilation=rate3)

        # 计算每个尺度的注意力权重
        y0_weight = self.SE1(self.gap(y0))  # (B, C, 1, 1)
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        y3_weight = self.SE4(self.gap(y3))

        # 拼接所有权重并归一化 (Multi-scale Selective Fusion)
        weight = torch.cat([y0_weight, y1_weight, y2_weight, y3_weight], 2)  # (B, C, 4, 1)
        weight = self.softmax(self.sigmoid(weight))  # 竞争式注意力

        # 分离每个尺度的权重
        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)  # (B, C, 1, 1)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 2], 2)
        y3_weight = torch.unsqueeze(weight[:, :, 3], 2)

        # 加权融合多尺度特征
        x_att = y0_weight * y0 + y1_weight * y1 + y2_weight * y2 + y3_weight * y3

        # 投影并添加残差连接
        return self.project(x_att + x)


class EncoderBase(nn.Module):
    """
    SUTrack编码器基类，集成MFEblock多尺度特征增强
    """
    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, 
                 num_channels: int, use_mfe: bool = True, mfe_atrous_rates: list = [2, 4, 8]):
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
                    parameter.requires_grad_(False)

        self.body = encoder
        self.num_channels = num_channels
        
        # 添加MFEblock模块
        self.use_mfe = use_mfe
        if self.use_mfe:
            self.mfe_module = MFEblock(
                in_channels=num_channels,
                atrous_rates=mfe_atrous_rates
            )
            print(f"[MFE Encoder] MFEblock initialized with {num_channels} channels, atrous_rates={mfe_atrous_rates}")

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        """
        前向传播
        在encoder输出后应用MFEblock进行多尺度特征增强
        主要增强search region特征（检测目标区域）
        """
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        
        # 应用MFEblock增强search region特征
        if self.use_mfe:
            feature = xs[0]  # (B, N, C) where N = 1(cls) + num_template + num_search
            
            # 获取搜索区域的patch数量和尺寸
            num_patches_search = self.body.num_patches_search
            num_frames = 1  # 默认搜索帧数
            fx_sz = int(num_patches_search ** 0.5)
            
            # 提取search region特征
            if self.body.cls_token is not None:
                # 有class token: [cls, template_patches, search_patches]
                search_feat = feature[:, 1 + self.body.num_patches_template:
                                     1 + self.body.num_patches_template + num_patches_search * num_frames]
            else:
                # 无class token: [template_patches, search_patches]
                search_feat = feature[:, self.body.num_patches_template:
                                     self.body.num_patches_template + num_patches_search * num_frames]
            
            bs, HW, C = search_feat.size()
            
            # 重塑为2D特征图用于MFEblock处理
            search_feat_2d = search_feat.permute(0, 2, 1).contiguous().view(bs, C, fx_sz, fx_sz)
            
            # 应用MFEblock多尺度特征增强
            search_feat_2d = self.mfe_module(search_feat_2d)
            
            # 重塑回序列格式
            search_feat = search_feat_2d.view(bs, C, -1).permute(0, 2, 1).contiguous()
            
            # 替换增强后的search特征
            if self.body.cls_token is not None:
                feature[:, 1 + self.body.num_patches_template:
                       1 + self.body.num_patches_template + num_patches_search * num_frames] = search_feat
            else:
                feature[:, self.body.num_patches_template:
                       self.body.num_patches_template + num_patches_search * num_frames] = search_feat
            
            xs = (feature,) + xs[1:] if len(xs) > 1 else (feature,)
                
        return xs


class Encoder(EncoderBase):
    """ViT encoder with MFEblock."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None,
                 use_mfe: bool = True):
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
        
        # 从配置获取MFE参数
        mfe_atrous_rates = getattr(cfg.MODEL, 'MFE_ATROUS_RATES', [2, 4, 8])
        super().__init__(encoder, train_encoder, open_layers, num_channels, use_mfe, mfe_atrous_rates)


def build_encoder(cfg):
    """构建带MFEblock的编码器"""
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    use_mfe = getattr(cfg.MODEL, 'USE_MFE', True)  # 默认使用MFE
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg, use_mfe)
    return encoder
