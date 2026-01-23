"""
快速测试SCSA配置是否正确加载
"""
import sys
sys.path.insert(0, '/home/nick/code/code.sutrack/SUTrack')

import importlib

# 加载配置
config_module = importlib.import_module("lib.config.sutrack_SCSA.config")
cfg = config_module.cfg
config_module.update_config_from_file('/home/nick/code/code.sutrack/SUTrack/experiments/sutrack_SCSA/sutrack_scsa_t224.yaml')

print("="*60)
print("SCSA 配置检查")
print("="*60)
print(f"USE_SCSA: {cfg.MODEL.ENCODER.get('USE_SCSA', 'NOT FOUND')}")
print(f"SCSA_REDUCTION_RATIO: {cfg.MODEL.ENCODER.get('SCSA_REDUCTION_RATIO', 'NOT FOUND')}")
print(f"SCSA_GATE_LAYER: {cfg.MODEL.ENCODER.get('SCSA_GATE_LAYER', 'NOT FOUND')}")
print(f"ENCODER TYPE: {cfg.MODEL.ENCODER.TYPE}")
print("="*60)

if cfg.MODEL.ENCODER.get('USE_SCSA', False):
    print("✅ SCSA 配置已启用")
else:
    print("❌ SCSA 配置未启用")
