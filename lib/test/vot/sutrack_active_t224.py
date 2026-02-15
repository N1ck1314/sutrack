"""
SUTrack Active with Dynamic Activation - VOT RGBD Wrapper
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from lib.test.vot.sutrack_class import run_vot_exp

# Run VOT experiment with RGBD channel type
run_vot_exp('sutrack_active', 'sutrack_active_t224', vis=False, out_conf=True, channel_type='rgbd')
