from easydict import EasyDict as edict
import yaml

'''
SUTrack Select (Selective Depth Integration)
选择性深度集成 - 借鉴SGLA层跳过机制，实现深度特征的智能选择使用
'''

cfg = edict()

# MODEL
cfg.MODEL = edict()

# TASK_INDEX
cfg.MODEL.TASK_NUM = 5  # should be the largest index number + 1
cfg.MODEL.TASK_INDEX = edict()  # index for tasks
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
cfg.MODEL.TEXT_ENCODER.TYPE = 'ViT-L/14'  # clip: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px

# MODEL.ENCODER
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = "fastitpnt"  # encoder model (using tiny)
cfg.MODEL.ENCODER.DROP_PATH = 0.1
cfg.MODEL.ENCODER.STRIDE = 16
cfg.MODEL.ENCODER.PRETRAIN_TYPE = "pretrained/itpn/fast_itpn_tiny_1600e_1k.pt"
cfg.MODEL.ENCODER.PATCHEMBED_INIT = "halfcopy"  # copy, halfcopy, random
cfg.MODEL.ENCODER.USE_CHECKPOINT = False  # to save the memory
cfg.MODEL.ENCODER.POS_TYPE = 'index'  # type of loading the positional encoding: "interpolate" or "index"
cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE = True  # add a token_type_embedding to indicate the search, template_foreground, template_background
cfg.MODEL.ENCODER.CLASS_TOKEN = True  # class token

# Selective Depth Integration Configuration
cfg.MODEL.ENCODER.USE_SELECTIVE_DEPTH = True  # Enable selective depth integration
cfg.MODEL.ENCODER.SELECTIVE_DEPTH_THRESHOLD = 0.5  # Threshold for inference (0-1)
cfg.MODEL.ENCODER.SELECTION_LOSS_WEIGHT = 0.01  # Weight for depth selection loss
cfg.MODEL.ENCODER.SELECTIVE_DEPTH_REDUCTION = 4  # Reduction ratio in predictor
cfg.MODEL.ENCODER.SELECTIVE_DEPTH_DROPOUT = 0.1  # Dropout rate
cfg.MODEL.ENCODER.USE_GUMBEL_SOFTMAX = False  # Use Gumbel-Softmax sampling

# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.TYPE = "CENTER"  # MLP, CORNER, CENTER
cfg.MODEL.DECODER.NUM_CHANNELS = 256
cfg.MODEL.DECODER.NUM_LAYERS = 3
cfg.MODEL.DECODER.KERNEL_SIZE = 5
cfg.MODEL.DECODER.CONV_TYPE = "normal"  # normal: 3*3 conv, small: 1*1 conv
cfg.MODEL.DECODER.XAVIER_INIT = True

# MODEL.TASK_DECODER
cfg.MODEL.TASK_DECODER = edict()
cfg.MODEL.TASK_DECODER.TYPE = "MLP"
cfg.MODEL.TASK_DECODER.NUM_CHANNELS = 256  # Increased from 128 to match S4F
cfg.MODEL.TASK_DECODER.NUM_LAYERS = 2
cfg.MODEL.TASK_DECODER.FEATURE_TYPE = 'average'  # class, text, average

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.TYPE = 'text_frozen'
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 180  # Reduced from 300 to match S4F
cfg.TRAIN.TASK_CE_WEIGHT = 0.3
cfg.TRAIN.LR_DROP_EPOCH = 144  # Adjusted for 180 epochs (0.8 * 180)
cfg.TRAIN.BATCH_SIZE = 32  # Increased from 16 to match S4F
cfg.TRAIN.NUM_WORKER = 10
cfg.TRAIN.OPTIMIZER = 'ADAMW'
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.CE_WEIGHT = 1.0
cfg.TRAIN.FREEZE_ENCODER = False
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False  # automatic mixed precision

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = 'step'
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# TRAIN.ENCODER_OPEN
cfg.TRAIN.ENCODER_OPEN = ['patch_embed.proj.weight', 'patch_embed.proj.bias',
                          'blocks.5', 'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11']

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 400  # Increased from 200 to match S4F
cfg.DATA.MULTI_MODAL_LANGUAGE = False
cfg.DATA.MULTI_MODAL_VISION = True  # Enable depth/multi-modal support

# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["GOT10K_vottrain", "DepthTrack_train"]  # Focus on GOT10K and DepthTrack
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 80000  # Increased from 60000

# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 224
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5  # Increased from 3 to match S4F
cfg.DATA.SEARCH.SCALE_JITTER = 0.5  # Increased from 0.25 to match S4F
cfg.DATA.SEARCH.NUMBER = 1

# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.SIZE = 112
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0
cfg.DATA.TEMPLATE.NUMBER = 1

# DATA.USE_NLP
cfg.DATA.USE_NLP = edict()
cfg.DATA.USE_NLP.LASOT = False
cfg.DATA.USE_NLP.GOT10K = False
cfg.DATA.USE_NLP.TRACKINGNET = False
cfg.DATA.USE_NLP.COCO = False
cfg.DATA.USE_NLP.DEPTHTRACK = False
cfg.DATA.USE_NLP.VASTTRACK = False
cfg.DATA.USE_NLP.TNL2K = False
cfg.DATA.USE_NLP.LASHER = False
cfg.DATA.USE_NLP.VISEVENT = False

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 112
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 224
cfg.TEST.EPOCH = 180  # Changed from 300
cfg.TEST.WINDOW = True

# TEST.MULTI_MODAL_VISION
cfg.TEST.MULTI_MODAL_VISION = edict()
cfg.TEST.MULTI_MODAL_VISION.DEFAULT = True

# TEST.MULTI_MODAL_LANGUAGE
cfg.TEST.MULTI_MODAL_LANGUAGE = edict()
cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT = False

# TEST.USE_NLP
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


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
