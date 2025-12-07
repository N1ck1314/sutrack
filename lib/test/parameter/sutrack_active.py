from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.sutrack_active.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    """
    Parameters for sutrack_active.
    Example: yaml_name='sutrack_t224' -> experiments/sutrack_active/sutrack_t224.yaml
    """
    params = TrackerParams()
    settings = env_settings()

    # Load yaml config
    yaml_file = os.path.join(settings.prj_dir, f'experiments/sutrack_active/{yaml_name}.yaml')
    update_config_from_file(yaml_file)
    params.cfg = cfg

    # Basic test sizes
    params.yaml_name = yaml_name
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path (active)
    # checkpoints/train/sutrack_active/<yaml_name>/SUTRACK_epXXXX.pth.tar
    params.checkpoint = os.path.join(
        settings.save_dir,
        f"checkpoints/train/sutrack_active/{yaml_name}/SUTRACK_ep{cfg.TEST.EPOCH:04d}.pth.tar"
    )

    # Optional flags
    params.save_all_boxes = False

    return params