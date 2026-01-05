import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.sutrack import build_sutrack
from lib.models.sutrack_active import build_sutrack_active
from lib.models.sutrack_rewight import build_sutrack_rewight
from lib.models.sutrack_patch import build_sutrack_patch
from lib.models.sutrack_scale import build_sutrack_scale
from lib.models.sutrack_STAtten import build_sutrack_statten
from lib.models.sutrack_S4F import build_sutrack_s4f
from lib.models.sutrack_CMA import build_sutrack_cma
from lib.models.sutrack_RMT import build_sutrack_rmt
from lib.models.sutrack_MLKA import build_sutrack_mlka
from lib.models.sutrack_MFE import build_sutrack as build_sutrack_mfe
from lib.models.sutrack_ASSA import build_sutrack as build_sutrack_assa
from lib.models.sutrack_CPAM import build_sutrack as build_sutrack_cpam
from lib.models.sutrack_DynRes import build_sutrack as build_sutrack_dynres
from lib.models.sutrack_SparseViT import build_sutrack as build_sutrack_sparsevit
from lib.models.sutrack_Mamba import build_sutrack as build_sutrack_mamba


from lib.train.actors import SUTrack_Actor
from lib.train.actors import SUTrack_active_Actor
from lib.utils.focal_loss import FocalLoss
# for import modules
import importlib


def run(settings):
    settings.description = 'Training script for Goku series'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg # generate cfg from lib.config
    config_module.update_config_from_file(settings.cfg_file) #update cfg from experiments
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal DATA LOADER")


    # Create network
    if settings.script_name == "sutrack":
        net = build_sutrack(cfg)
    elif settings.script_name == "sutrack_active":
        net = build_sutrack_active(cfg)
    elif settings.script_name == "sutrack_rewight":
        net = build_sutrack_rewight(cfg)
    elif settings.script_name == "sutrack_patch":
        net = build_sutrack_patch(cfg)
    elif settings.script_name == "sutrack_scale":
        net = build_sutrack_scale(cfg)
    elif settings.script_name == "sutrack_STAtten":
        net = build_sutrack_statten(cfg)
    elif settings.script_name == "sutrack_S4F":
        net = build_sutrack_s4f(cfg)
    elif settings.script_name == "sutrack_CMA":
        net = build_sutrack_cma(cfg)
    elif settings.script_name == "sutrack_RMT":
        net = build_sutrack_rmt(cfg)
    elif settings.script_name == "sutrack_MLKA":
        net = build_sutrack_mlka(cfg)
    elif settings.script_name == "sutrack_MFE":
        net = build_sutrack_mfe(cfg)
    elif settings.script_name == "sutrack_ASSA":
        net = build_sutrack_assa(cfg)
    elif settings.script_name == "sutrack_CPAM":
        net = build_sutrack_cpam(cfg)
    elif settings.script_name == "sutrack_DynRes":
        net = build_sutrack_dynres(cfg)
    elif settings.script_name == "sutrack_SparseViT":
        net = build_sutrack_sparsevit(cfg)
    elif settings.script_name == "sutrack_Mamba":
        net = build_sutrack_mamba(cfg)

    else:
        raise ValueError("illegal script name")
    
    # 打印模块配置确认信息（在配置加载后）
    if settings.local_rank in [-1, 0]:
        if settings.script_name == "sutrack_STAtten":
            print("\n" + "="*60)
            print("🔍 STAtten模块配置确认")
            print("="*60)
            use_statten = cfg.MODEL.ENCODER.get('USE_STATTEN', False)
            statten_mode = cfg.MODEL.ENCODER.get('STATTEN_MODE', 'STAtten')
            use_snn = cfg.MODEL.ENCODER.get('USE_SNN', False)
            print(f"✓ STAtten启用状态: {'🟢 已启用' if use_statten else '🔴 未启用'}")
            if use_statten:
                print(f"✓ 注意力模式: {statten_mode}")
                print(f"✓ 脉冲神经网络(SNN): {'🟢 启用' if use_snn else '🔴 禁用'}")
                print("✓ 注意力机制: 时空注意力 (替代标准自注意力)")
            else:
                print("⚠️  警告: STAtten未启用，将使用标准的Transformer注意力")
            print("="*60 + "\n")
        elif settings.script_name == "sutrack_S4F":
            print("\n" + "="*60)
            print("🔍 CMSA模块配置确认")
            print("="*60)
            use_cmsa = cfg.MODEL.ENCODER.get('USE_CMSA', False)
            cmsa_mode = cfg.MODEL.ENCODER.get('CMSA_MODE', 'cmsa')
            use_ssm = cfg.MODEL.ENCODER.get('USE_SSM', True)
            print(f"✓ CMSA启用状态: {'🟢 已启用' if use_cmsa else '🔴 未启用'}")
            if use_cmsa:
                print(f"✓ CMSA融合模式: {cmsa_mode}")
                print(f"✓ 状态空间模型(SSM): {'🟢 启用' if use_ssm else '🔴 禁用'}")
                print("✓ 多模态融合策略: 跨模态空间感知 (替代简单拼接)")
            else:
                print("⚠️  警告: CMSA未启用，将使用原始的简单拼接融合")
            print("="*60 + "\n")
        elif settings.script_name == "sutrack_CMA":
            print("\n" + "="*60)
            print("🔍 CMA模块配置确认")
            print("="*60)
            use_cma = cfg.MODEL.get('USE_CMA', True)  # 修复：从MODEL而不是ENCODER获取
            hidden_ratio = cfg.MODEL.CMA.get('HIDDEN_RATIO', 0.5) if hasattr(cfg.MODEL, 'CMA') else 0.5
            print(f"✓ CMA启用状态: {'🟢 已启用' if use_cma else '🔴 未启用'}")
            if use_cma:
                print(f"✓ 隐藏层通道比例: {hidden_ratio}")
                print("✓ 融合机制: 跨模态注意力 (RGB空间域 ↔ 频域)")
                print("✓ 增强范围: Search Region特征增强")
                print("✓ 适用场景: 多尺度特征提取、全局建模增强")
            else:
                print("⚠️  警告: CMA未启用，将使用标准特征流")
            print("="*60 + "\n")
        elif settings.script_name == "sutrack_RMT":
            print("\n" + "="*60)
            print("🔍 RMT模块配置确认")
            print("="*60)
            use_rmt = cfg.MODEL.ENCODER.get('USE_RMT', False)
            rmt_layers = cfg.MODEL.ENCODER.get('RMT_LAYERS', [])
            rmt_num_heads = cfg.MODEL.ENCODER.get('RMT_NUM_HEADS', 8)
            print(f"✓ RMT启用状态: {'🟢 已启用' if use_rmt else '🔴 未启用'}")
            if use_rmt:
                print(f"✓ RMT层索引: {rmt_layers}")
                print(f"✓ 注意力头数: {rmt_num_heads}")
                print("✓ 注意力机制: Retentive Multi-scale Attention (替代标准自注意力)")
                print("✓ 优势: 更长的记忆保持、线性复杂度、全局上下文建模")
            else:
                print("⚠️  警告: RMT未启用，将使用标准的Transformer注意力")
            print("="*60 + "\n")
        elif settings.script_name == "sutrack_MLKA":
            print("\n" + "="*60)
            print("🔍 MLKA模块配置确认")
            print("="*60)
            use_mlka = cfg.MODEL.get('USE_MLKA', False)
            mlka_position = cfg.MODEL.get('MLKA_POSITION', 'decoder')
            mlka_blocks = cfg.MODEL.get('MLKA_BLOCKS', 1)
            print(f"✓ MLKA启用状态: {'🟢 已启用' if use_mlka else '🔴 未启用'}")
            if use_mlka:
                print(f"✓ MLKA位置: {mlka_position}")
                print(f"✓ MLKA块数: {mlka_blocks}")
                print("✓ 多尺度核: 3x3, 5x5, 7x7 (配合空洞卷积)")
                print("✓ 注意力机制: 大核注意力 (增强感受野)")
                position_desc = {
                    'decoder': '解码器前增强 - 提升定位精度',
                    'encoder': '编码器后增强 - 提升特征表达',
                    'both': '双重增强 - 最强效果'
                }
                print(f"✓ 增强策略: {position_desc.get(mlka_position, '自定义位置')}")
            else:
                print("⚠️  警告: MLKA未启用，将使用标准的特征流")
            print("="*60 + "\n")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=True) # modify the find_unused_parameters to False to skip a runtime error of twice variable ready
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    
    # 验证模块是否被实际初始化
    if settings.local_rank in [-1, 0]:
        if settings.script_name == "sutrack_STAtten":
            print("\n🔍 验证STAtten模块实际初始化状态...")
            # 获取encoder
            encoder = net.module.encoder.body if hasattr(net, 'module') else net.encoder.body
            # 检查blocks中是否使用了STAtten
            if hasattr(encoder, 'blocks') and len(encoder.blocks) > 0:
                # 检查最后的main blocks中的attention类型
                last_block = encoder.blocks[-1]
                if hasattr(last_block, 'attn'):
                    attn_type = type(last_block.attn).__name__
                    if 'STAtten' in attn_type:
                        print("✅ STAtten模块已成功初始化！")
                        print(f"   - Attention类型: {attn_type}")
                    else:
                        print(f"⚠️  使用的是标准注意力: {attn_type}")
                else:
                    print("⚠️  无法检测attention模块")
            else:
                print("⚠️  无法检测encoder blocks")
            print()
        elif settings.script_name == "sutrack_S4F":
            print("\n🔍 验证CMSA模块实际初始化状态...")
            # 获取encoder
            encoder = net.module.encoder.body if hasattr(net, 'module') else net.encoder.body
            if hasattr(encoder, 'cmsa_search') and encoder.cmsa_search is not None:
                print("✅ CMSA模块已成功初始化！")
                print(f"   - cmsa_search: {type(encoder.cmsa_search).__name__}")
                print(f"   - cmsa_template: {type(encoder.cmsa_template).__name__}")
            else:
                print("⚠️  CMSA模块未初始化（可能配置中USE_CMSA=False）")
            print()
        elif settings.script_name == "sutrack_CMA":
            print("\n🔍 验证CMA模块实际初始化状态...")
            # 获取encoder（注意：CMA在encoder wrapper中，不是在body中）
            encoder = net.module.encoder if hasattr(net, 'module') else net.encoder
            if hasattr(encoder, 'cma_module') and encoder.cma_module is not None:
                print("✅ CMA模块已成功初始化！")
                print(f"   - cma_module: {type(encoder.cma_module).__name__}")
                if hasattr(encoder.cma_module, 'freq_filter'):
                    print(f"   - freq_filter: {type(encoder.cma_module.freq_filter).__name__}")
                if hasattr(encoder.cma_module, 'cma_block'):
                    print(f"   - cma_block: {type(encoder.cma_module.cma_block).__name__}")
                print(f"   - 增强状态: ✅ 启用 (use_cma={encoder.use_cma})")
            else:
                print("⚠️  CMA模块未初始化（可能配置中USE_CMA=False）")
                print(f"   - encoder.use_cma: {getattr(encoder, 'use_cma', 'N/A')}")
            print()
        elif settings.script_name == "sutrack_RMT":
            print("\n🔍 验证RMT模块实际初始化状态...")
            # 获取encoder
            encoder = net.module.encoder.body if hasattr(net, 'module') else net.encoder.body
            if hasattr(encoder, 'rmt_rel_pos_encoder') and encoder.rmt_rel_pos_encoder is not None:
                print("✅ RMT模块已成功初始化！")
                print(f"   - rmt_rel_pos_encoder: {type(encoder.rmt_rel_pos_encoder).__name__}")
                print(f"   - RMT层数: {len(encoder.rmt_layers) if hasattr(encoder, 'rmt_layers') else 0}")
            else:
                print("⚠️  RMT模块未初始化（可能配置中USE_RMT=False）")
            print()
        elif settings.script_name == "sutrack_MLKA":
            print("\n🔍 验证MLKA模块实际初始化状态...")
            # 获取主模型
            model = net.module if hasattr(net, 'module') else net
            if hasattr(model, 'mlka_decoder') and model.mlka_decoder is not None:
                print("✅ MLKA模块已成功初始化！")
                print(f"   - mlka_decoder: {type(model.mlka_decoder).__name__}")
                if hasattr(model, 'mlka_encoder') and model.mlka_encoder is not None:
                    print(f"   - mlka_encoder: {type(model.mlka_encoder).__name__}")
                    print("   - 增强模式: 双重增强 (encoder + decoder)")
                else:
                    print("   - 增强模式: decoder增强")
            elif hasattr(model, 'mlka_encoder') and model.mlka_encoder is not None:
                print("✅ MLKA模块已成功初始化！")
                print(f"   - mlka_encoder: {type(model.mlka_encoder).__name__}")
                print("   - 增强模式: encoder增强")
            else:
                print("⚠️  MLKA模块未初始化（可能配置中USE_MLKA=False）")
            print()
        elif settings.script_name == "sutrack_MFE":
            print("\n🔍 验证MFEblock模块实际初始化状态...")
            # 获取encoder（注意：MFE在encoder wrapper中）
            encoder = net.module.encoder if hasattr(net, 'module') else net.encoder
            if hasattr(encoder, 'mfe_module') and encoder.mfe_module is not None:
                print("✅ MFEblock模块已成功初始化！")
                print(f"   - mfe_module: {type(encoder.mfe_module).__name__}")
                print(f"   - 增强状态: ✅ 启用 (use_mfe={encoder.use_mfe})")
                # 统计MFE参数量
                mfe_params = sum(p.numel() for p in encoder.mfe_module.parameters())
                print(f"   - MFE参数量: {mfe_params / 1e6:.2f}M")
            else:
                print("⚠️  MFEblock模块未初始化（可能配置中USE_MFE=False）")
                print(f"   - encoder.use_mfe: {getattr(encoder, 'use_mfe', 'N/A')}")
            print()
        elif settings.script_name == "sutrack_ASSA":
            print("\n🔍 验证ASSA模块实际初始化状态...")
            # 获取encoder（注意：ASSA在encoder wrapper中）
            encoder = net.module.encoder if hasattr(net, 'module') else net.encoder
            if hasattr(encoder, 'assa_blocks') and encoder.assa_blocks is not None:
                print("✅ ASSA模块已成功初始化！")
                print(f"   - ASSA块数量: {len(encoder.assa_blocks)}")
                print(f"   - 增强状态: ✅ 启用 (use_assa={encoder.use_assa})")
                # 统计ASSA参数量
                assa_params = sum(p.numel() for p in encoder.assa_blocks.parameters())
                print(f"   - ASSA参数量: {assa_params / 1e6:.2f}M")
            else:
                print("⚠️  ASSA模块未初始化（可能配置中USE_ASSA=False）")
                print(f"   - encoder.use_assa: {getattr(encoder, 'use_assa', 'N/A')}")
            print()
        elif settings.script_name == "sutrack_CPAM":
            print("\n🔍 验证CPAM模块实际初始化状态...")
            # 获取encoder（注意：CPAM在encoder wrapper中）
            encoder = net.module.encoder if hasattr(net, 'module') else net.encoder
            if hasattr(encoder, 'cpam_module') and encoder.cpam_module is not None:
                print("✅ CPAM模块已成功初始化！")
                print(f"   - cpam_module: {type(encoder.cpam_module).__name__}")
                print(f"   - 增强状态: ✅ 启用 (use_cpam={encoder.use_cpam})")
                # 统计CPAM参数量
                cpam_params = sum(p.numel() for p in encoder.cpam_module.parameters())
                print(f"   - CPAM参数量: {cpam_params / 1e6:.2f}M")
                print("   - 注意力机制: 通道注意力 + 位置注意力")
            else:
                print("⚠️  CPAM模块未初始化（可能配置中USE_CPAM=False）")
                print(f"   - encoder.use_cpam: {getattr(encoder, 'use_cpam', 'N/A')}")
            print()
        elif settings.script_name == "sutrack_DynRes":
            print("\n🔍 验证DynRes模块实际初始化状态...")
            # 获取encoder（注意：DynRes在encoder wrapper中）
            encoder = net.module.encoder if hasattr(net, 'module') else net.encoder
            if hasattr(encoder, 'dynres_module') and encoder.dynres_module is not None:
                print("✅ DynRes模块已成功初始化！")
                print(f"   - dynres_module: {type(encoder.dynres_module).__name__}")
                print(f"   - 增强状态: ✅ 启用 (use_dynres={encoder.use_dynres})")
                # 统计DynRes参数量
                dynres_params = sum(p.numel() for p in encoder.dynres_module.parameters())
                print(f"   - DynRes参数量: {dynres_params / 1e6:.2f}M")
                print("   - 核心机制: 动态分辨率 + 多视图融合 + 区域对齐")
            else:
                print("⚠️  DynRes模块未初始化（可能配置中USE_DYNRES=False）")
                print(f"   - encoder.use_dynres: {getattr(encoder, 'use_dynres', 'N/A')}")
            print()
        elif settings.script_name == "sutrack_SparseViT":
            print("\n🔍 验证SparseViT模块实际初始化状态...")
            # 获取encoder（注意：SparseViT在encoder wrapper中）
            encoder = net.module.encoder if hasattr(net, 'module') else net.encoder
            if hasattr(encoder, 'sparsevit_module') and encoder.sparsevit_module is not None:
                print("✅ SparseViT模块已成功初始化！")
                print(f"   - sparsevit_module: {type(encoder.sparsevit_module).__name__}")
                print(f"   - 增强状态: ✅ 启用 (use_sparsevit={encoder.use_sparsevit})")
                # 统计SparseViT参数量
                sparsevit_params = sum(p.numel() for p in encoder.sparsevit_module.parameters())
                print(f"   - SparseViT参数量: {sparsevit_params / 1e6:.2f}M")
                print(f"   - SABlock数量: {len(encoder.sparsevit_module.blocks)}")
                print("   - 核心机制: 稀疏自注意力 + 层级稀疏结构")
            else:
                print("⚠️  SparseViT模块未初始化（可能配置中USE_SPARSEVIT=False）")
                print(f"   - encoder.use_sparsevit: {getattr(encoder, 'use_sparsevit', 'N/A')}")
            print()
        elif settings.script_name == "sutrack_Mamba":
            print("\n🔍 验证Mamba模块实际初始化状态...")
            # 获取主模型
            model = net.module if hasattr(net, 'module') else net
            if hasattr(model, 'mamba_fusion') and model.mamba_fusion is not None:
                print("✅ Mamba模块已成功初始化！")
                print(f"   - mamba_fusion: {type(model.mamba_fusion).__name__}")
                print(f"   - 层数: {len(model.mamba_fusion.mamba_layers)}")
                print(f"   - 增强状态: ✅ 启用 (use_mamba={model.use_mamba})")
                # 统计Mamba参数量
                mamba_params = sum(p.numel() for p in model.mamba_fusion.parameters())
                print(f"   - Mamba参数量: {mamba_params / 1e6:.2f}M")
                print("   - 核心机制: 选择性状态空间模型 (SSM) + 线性复杂度")
            else:
                print("⚠️  Mamba模块未初始化（可能配置中USE_MAMBA=False）")
                print(f"   - model.use_mamba: {getattr(model, 'use_mamba', 'N/A')}")
            print()
    # Loss functions and Actors
    if settings.script_name == "sutrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_rewight":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_patch":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_scale":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_STAtten":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_S4F":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_CMA":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_RMT":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_MLKA":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_MFE":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_ASSA":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_CPAM":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_DynRes":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_SparseViT":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_Mamba":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "sutrack_active":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss(),
                     'task_cls': CrossEntropyLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': cfg.TRAIN.CE_WEIGHT,
                       'task_cls': cfg.TRAIN.TASK_CE_WEIGHT}
        actor = SUTrack_active_Actor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
