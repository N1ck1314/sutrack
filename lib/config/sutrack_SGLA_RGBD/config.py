from easydict import EasyDict as edict
import yaml

'''
SUTrack SGLA-RGBD: SGLA-inspired Multi-Modal RGBD Fusion
基于SGLA思想的多模态RGBD跟踪器
'''

cfg = edict()

# MODEL
cfg.MODEL = edict()

# TASK_INDEX
cfg.MODEL.TASK_NUM = 5
cfg.MODEL.TASK_INDEX = edict()
cfg.MODEL.TASK_INDEX.VASTTRACK = 0
cfg.MODEL.TASK_INDEX.LASOT = 0
cfg.MODEL.TASK_INDEX.TRACKINGNET = 0
cfg.MODEL.TASK_INDEX.GOT10K = 0
cfg.MODEL.TASK_INDEX.COCO = 0
cfg.MODEL.TASK_INDEX.TNL2K = 1
cfg.MODEL.TASK_INDEX.DEPTHTRACK = 2
cfg.MODEL.TASK_INDEX.LASHER = 3
cfg.MODEL.TASK_INDEX.VISEVENT = 4

# MODEL.TEXT_ENCODER
cfg.MODEL.TEXT_ENCODER = edict()
cfg.MODEL.TEXT_ENCODER.TYPE = 'ViT-L/14'

# MODEL.ENCODER
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = "fastitpnt"
cfg.MODEL.ENCODER.DROP_PATH = 0.1
cfg.MODEL.ENCODER.PRETRAIN_TYPE = "pretrained/itpn/fast_itpn_tiny_1600e_1k.pt"
cfg.MODEL.ENCODER.PATCHEMBED_INIT = "halfcopy"
cfg.MODEL.ENCODER.USE_CHECKPOINT = False
cfg.MODEL.ENCODER.STRIDE = 16
cfg.MODEL.ENCODER.POS_TYPE = 'index'
cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE = True
cfg.MODEL.ENCODER.CLASS_TOKEN = True

# SGLA Configuration (保留原有SGLA功能)
cfg.MODEL.ENCODER.USE_SGLA = False  # 关闭原SGLA，使用SGLA-RGBD
cfg.MODEL.ENCODER.SGLA_LOSS_WEIGHT = 0.1

# SGLA-RGBD Configuration (新增)
cfg.MODEL.ENCODER.USE_SGLA_RGBD = True
cfg.MODEL.ENCODER.SGLA_RGBD = edict()
cfg.MODEL.ENCODER.SGLA_RGBD.USE_MODAL_SELECTION = True  # 模态选择
cfg.MODEL.ENCODER.SGLA_RGBD.USE_LAYERWISE_FUSION = True  # 逐层融合
cfg.MODEL.ENCODER.SGLA_RGBD.USE_SELECTIVE_DEPTH = True  # 选择性深度
cfg.MODEL.ENCODER.SGLA_RGBD.USE_COMPLEMENTARITY_LOSS = True  # 互补性损失
cfg.MODEL.ENCODER.SGLA_RGBD.COMPLEMENTARITY_LOSS_WEIGHT = 0.1
cfg.MODEL.ENCODER.SGLA_RGBD.MODAL_BALANCE_WEIGHT = 0.05

# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.TYPE = "CENTER"
cfg.MODEL.DECODER.NUM_CHANNELS = 256
cfg.MODEL.DECODER.CONV_TYPE = "normal"
cfg.MODEL.DECODER.XAVIER_INIT = True

# MODEL.TASK_DECODER
cfg.MODEL.TASK_DECODER = edict()
cfg.MODEL.TASK_DECODER.NUM_CHANNELS = 256
cfg.MODEL.TASK_DECODER.FEATURE_TYPE = "average"

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 180
cfg.TRAIN.LR_DROP_EPOCH = 144
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.NUM_WORKER = 10
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1
cfg.TRAIN.FREEZE_ENCODER = False
cfg.TRAIN.ENCODER_OPEN = []
cfg.TRAIN.CE_WEIGHT = 1.0
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.TASK_CE_WEIGHT = 1.0
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.FIX_BN = False

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.TYPE = "text_frozen"
cfg.TRAIN.PRETRAINED_PATH = None

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 400
cfg.DATA.SAMPLER_MODE = "order"
cfg.DATA.LOADER = "tracking"
cfg.DATA.MULTI_MODAL_VISION = True  # 启用多模态(RGB+Depth)
cfg.DATA.MULTI_MODAL_LANGUAGE = False
cfg.DATA.USE_NLP = edict()
cfg.DATA.USE_NLP.LASOT = False
cfg.DATA.USE_NLP.GOT10K = False
cfg.DATA.USE_NLP.COCO = False
cfg.DATA.USE_NLP.TRACKINGNET = False
cfg.DATA.USE_NLP.VASTTRACK = False
cfg.DATA.USE_NLP.REFCOCOG = False
cfg.DATA.USE_NLP.TNL2K = False
cfg.DATA.USE_NLP.OTB99 = False
cfg.DATA.USE_NLP.DEPTHTRACK = False
cfg.DATA.USE_NLP.LASHER = False
cfg.DATA.USE_NLP.VISEVENT = False

# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["GOT10K_vottrain", "DepthTrack_train"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000

# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.SIZE = 224
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5

# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 112
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 112
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 224
cfg.TEST.EPOCH = 180
cfg.TEST.WINDOW = True
cfg.TEST.NUM_TEMPLATES = 1

cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.DEFAULT = 999999

cfg.TEST.UPDATE_THRESHOLD = edict()
cfg.TEST.UPDATE_THRESHOLD.DEFAULT = 1.0

cfg.TEST.MULTI_MODAL_VISION = edict()
cfg.TEST.MULTI_MODAL_VISION.DEFAULT = True

cfg.TEST.MULTI_MODAL_LANGUAGE = edict()
cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT = False

cfg.TEST.USE_NLP = edict()
cfg.TEST.USE_NLP.DEFAULT = False
cfg.TEST.USE_NLP.LASOT = False
cfg.TEST.USE_NLP.GOT10K = False
cfg.TEST.USE_NLP.COCO = False
cfg.TEST.USE_NLP.TRACKINGNET = False
cfg.TEST.USE_NLP.VASTTRACK = False
cfg.TEST.USE_NLP.TNL2K = False
cfg.TEST.USE_NLP.OTB99 = False
cfg.TEST.USE_NLP.DEPTHTRACK = False
cfg.TEST.USE_NLP.LASHER = False
cfg.TEST.USE_NLP.VISEVENT = False


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
