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
            use_cma = cfg.MODEL.ENCODER.get('USE_CMA', False)
            cma_mode = cfg.MODEL.ENCODER.get('CMA_MODE', 'cma')
            print(f"✓ CMA启用状态: {'🟢 已启用' if use_cma else '🔴 未启用'}")
            if use_cma:
                print(f"✓ CMA融合模式: {cma_mode}")
                print("✓ 融合机制: 跨模态注意力 (替代简单拼接)")
                print("✓ 适用场景: 多模态融合、语义引导的视觉注意力")
            else:
                print("⚠️  警告: CMA未启用，将使用简单的特征拼接")
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
            # 获取encoder
            encoder = net.module.encoder.body if hasattr(net, 'module') else net.encoder.body
            if hasattr(encoder, 'cma_fusion') and encoder.cma_fusion is not None:
                print("✅ CMA模块已成功初始化！")
                print(f"   - cma_fusion: {type(encoder.cma_fusion).__name__}")
            else:
                print("⚠️  CMA模块未初始化（可能配置中USE_CMA=False）")
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
