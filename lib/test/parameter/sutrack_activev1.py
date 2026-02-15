from lib.config.sutrack_activev1.config import cfg, update_config_from_file
from lib.utils.load import load_yaml
import os


def parameters(yaml_name='sutrack_activev1_t224'):
    params = load_yaml(os.path.join(os.path.dirname(__file__), '../../../experiments/sutrack_activev1', yaml_name + '.yaml'))
    update_config_from_file(params)
    cfg.TEST.EPOCH = params.TEST.EPOCH
    return cfg


if __name__ == '__main__':
    p = parameters()
    print(p)
