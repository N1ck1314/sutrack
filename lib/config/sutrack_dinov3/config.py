from easydict import EasyDict as edict
import yaml

'''
SUTrack with DINOv3 ConvNeXt-Tiny Encoder
使用 facebook/dinov3-convnext-tiny 作为 backbone，速度优先
'''

cfg = edict()

# MODEL
cfg.MODEL = edict()

# TAKS_INDEX
cfg.MODEL.TASK_NUM=5 #should be the largest index number + 1
cfg.MODEL.TASK_INDEX = edict() # index for tasks
cfg.MODEL.TASK_INDEX.VASTTRACK = 0
cfg.MODEL.TASK_INDEX.LASOT = 0
cfg.MODEL.TASK_INDEX.TRACKINGNET = 0
cfg.MODEL.TASK_INDEX.GOT10K = 0
cfg.MODEL.TASK_INDEX.COCO = 0
cfg.MODEL.TASK_INDEX.TNL2K = 1
cfg.MODEL.TASK_INDEX.DEPTHTRACK = 2
cfg.MODEL.TASK_INDEX.LASHER = 3
cfg.MODEL.TASK_INDEX.VISEVENT = 4


# MODEL.LANGUAGE
cfg.MODEL.TEXT_ENCODER = edict()
cfg.MODEL.TEXT_ENCODER.TYPE = 'ViT-L/14' # clip: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px

# MODEL.ENCODER - 关键修改
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = "dinov3_convnext_tiny"  # 使用 DINOv3 ConvNeXt-Tiny
cfg.MODEL.ENCODER.DROP_PATH = 0
cfg.MODEL.ENCODER.PRETRAIN_TYPE = ""  # DINOv3 从 HuggingFace 自动加载，不需要本地权重
cfg.MODEL.ENCODER.PATCHEMBED_INIT = "copy"  # 保留以兼容接口，但 DINOv3 不会用到
cfg.MODEL.ENCODER.USE_CHECKPOINT = False # to save the memory.
cfg.MODEL.ENCODER.STRIDE = 32  # DINOv3 ConvNeXt-Tiny 的输出 stride
cfg.MODEL.ENCODER.POS_TYPE = 'interpolate' # 位置编码插值方式
cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE = False  # 暂时关闭 token_type（简化版）
cfg.MODEL.ENCODER.CLASS_TOKEN = False  # 暂时关闭 cls_token（简化版）

# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.TYPE = "CENTER" # MLP, CORNER, CENTER
cfg.MODEL.DECODER.NUM_CHANNELS = 256
cfg.MODEL.DECODER.CONV_TYPE = "normal" # normal: 3*3 conv, small: 1*1 conv, only for the center head for now.
cfg.MODEL.DECODER.XAVIER_INIT = True

# MODEL.TASK_DECODER
cfg.MODEL.TASK_DECODER = edict()
cfg.MODEL.TASK_DECODER.NUM_CHANNELS = 256
cfg.MODEL.TASK_DECODER.FEATURE_TYPE = "average" # class: using class token, average: average the feature, text: using the text token

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 180
cfg.TRAIN.LR_DROP_EPOCH = 144
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1  # encoder's LR = this factor * LR
cfg.TRAIN.FREEZE_ENCODER = False # 是否冻结 encoder（建议先 True 测试，再 False 微调）
cfg.TRAIN.ENCODER_OPEN = [] # only for debug, open some layers of encoder when FREEZE_ENCODER is True
cfg.TRAIN.CE_WEIGHT = 1.0 # weight for cross-entropy loss
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.TASK_CE_WEIGHT = 1.0
cfg.TRAIN.PRINT_INTERVAL = 50 # interval to print the training log
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.FIX_BN = False
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.TYPE = "normal" # normal, peft, fft, text_frozen
cfg.TRAIN.PRETRAINED_PATH = None

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = "order"
cfg.DATA.LOADER = "tracking"
cfg.DATA.MULTI_MODAL_VISION = True # vision multi-modal
cfg.DATA.MULTI_MODAL_LANGUAGE = False  # 暂时关闭文本模态（简化版）
cfg.DATA.USE_NLP = edict() # using the text of the dataset
cfg.DATA.USE_NLP.LASOT = False
cfg.DATA.USE_NLP.GOT10K = False
cfg.DATA.USE_NLP.COCO = False
cfg.DATA.USE_NLP.TRACKINGNET = False
cfg.DATA.USE_NLP.VASTTRACK = False
cfg.DATA.USE_NLP.REFCOCOG = False
cfg.DATA.USE_NLP.TNL2K = False
cfg.DATA.USE_NLP.OTB99 = False
cfg.DATA.USE_NLP.DEPTHTRACK = False  # 添加 DepthTrack
cfg.DATA.USE_NLP.LASHER = False       # 添加 LasHeR
cfg.DATA.USE_NLP.VISEVENT = False     # 添加 VisEvent

# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 112
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.SIZE = 224
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3
cfg.DATA.SEARCH.SCALE_JITTER = 0.25

# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 112
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 224
cfg.TEST.EPOCH = 180
cfg.TEST.WINDOW = True  # 添加 WINDOW 字段
cfg.TEST.MULTI_MODAL_VISION = edict()
cfg.TEST.MULTI_MODAL_VISION.DEFAULT = True
cfg.TEST.MULTI_MODAL_LANGUAGE = edict()
cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT = False
cfg.TEST.USE_NLP = edict()
cfg.TEST.USE_NLP.DEFAULT = False
cfg.TEST.USE_NLP.TNL2K = False


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


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
