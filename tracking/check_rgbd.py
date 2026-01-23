#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUTrack RGBD 模式检测工具

用于检查当前SUTrack模型配置是否支持RGBD输入

运行方式：
  1) conda activate sutrack
  2) export PYTHONPATH=/绝对路径/SUTrack:$PYTHONPATH
  3) python check_rgbd.py sutrack_b224
"""

import os
import sys
import argparse

# 添加路径
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker


def check_rgbd_configuration(tracker_param):
    """检查RGBD配置"""
    print("\n" + "="*80)
    print("SUTrack RGBD Configuration Checker")
    print("="*80)
    
    try:
        # 创建跟踪器以获取配置
        tracker_wrapper = Tracker("sutrack", tracker_param, "demo", run_id=None)
        params = tracker_wrapper.get_parameters()
        cfg = params.cfg
        
        print(f"Configuration: {tracker_param}")
        print("-" * 80)
        
        # 显示原始配置
        print("\n🔍 ORIGINAL CONFIGURATION (Before Modification):")
        original_score = analyze_config(cfg, "Original")
        
        # 模拟我们的修改过程
        print("\n🔧 APPLYING RGBD MODIFICATIONS...")
        modified_cfg = apply_rgbd_modifications(cfg)
        
        # 显示修改后的配置
        print("\n🔍 MODIFIED CONFIGURATION (After RGBD Enhancement):")
        modified_score = analyze_config(modified_cfg, "Modified")
        
        # 对比总结
        print("\n📊 COMPARISON SUMMARY:")
        print(f"   Original Score:  {original_score[0]}/{original_score[1]} - {get_score_description(original_score[0], original_score[1])}")
        print(f"   Modified Score:  {modified_score[0]}/{modified_score[1]} - {get_score_description(modified_score[0], modified_score[1])}")
        
        if modified_score[0] > original_score[0]:
            print("   ✅ IMPROVEMENT: RGBD modifications successfully enhanced the configuration!")
        else:
            print("   ⚠️  NO IMPROVEMENT: Modifications may not have been applied correctly.")
        
        print("\n💡 WHAT HAPPENS IN YOUR DEMO:")
        print("   Your demo_realsense.py and mydemo.py automatically apply these modifications")
        print("   when initializing the tracker, enabling full RGBD support.")
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to check configuration")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80 + "\n")


def analyze_config(cfg, config_name):
    """分析配置并返回RGBD支持分数"""
    print(f"\n--- {config_name} Configuration Analysis ---")
    
    # 1. 检查模型配置
    print("\n🔍 MODEL CONFIGURATION:")
    
    if hasattr(cfg.MODEL, 'ENCODER'):
        encoder_cfg = cfg.MODEL.ENCODER
        print(f"   Encoder Type: {getattr(encoder_cfg, 'TYPE', 'Not specified')}")
        
        # 检查输入通道
        input_channels = None
        for attr in ['in_chans', 'IN_CHANS', 'INPUT_CHANNELS']:
            if hasattr(encoder_cfg, attr):
                input_channels = getattr(encoder_cfg, attr)
                print(f"   Input Channels: {input_channels} (from {attr})")
                break
        
        if input_channels is None:
            print("   Input Channels: Not explicitly specified")
    
    # 2. 检查数据配置
    print("\n🔍 DATA CONFIGURATION:")
    
    if hasattr(cfg.DATA, 'MULTI_MODAL_VISION'):
        print(f"   Multi-modal Vision: {cfg.DATA.MULTI_MODAL_VISION}")
    
    if hasattr(cfg.DATA, 'MEAN') and hasattr(cfg.DATA, 'STD'):
        mean = cfg.DATA.MEAN
        std = cfg.DATA.STD
        print(f"   Normalization MEAN: {mean} ({len(mean)} channels)")
        print(f"   Normalization STD:  {std} ({len(std)} channels)")
    
    # 3. 计算得分
    rgbd_indicators = []
    
    # 检查各种RGBD指标
    if hasattr(cfg.MODEL, 'ENCODER'):
        encoder_cfg = cfg.MODEL.ENCODER
        for attr in ['in_chans', 'IN_CHANS', 'INPUT_CHANNELS']:
            if hasattr(encoder_cfg, attr) and getattr(encoder_cfg, attr) == 6:
                rgbd_indicators.append("✅ 6-channel encoder input")
                break
        else:
            rgbd_indicators.append("⚠️  Non-6-channel encoder input")
    
    if hasattr(cfg.DATA, 'MEAN') and len(cfg.DATA.MEAN) == 6:
        rgbd_indicators.append("✅ 6-channel data normalization")
    else:
        rgbd_indicators.append("⚠️  Non-6-channel data normalization")
    
    if hasattr(cfg.DATA, 'MULTI_MODAL_VISION') and cfg.DATA.MULTI_MODAL_VISION:
        rgbd_indicators.append("✅ Multi-modal vision enabled")
    else:
        rgbd_indicators.append("⚠️  Multi-modal vision disabled")
    
    print("\n📋 INDICATORS:")
    for indicator in rgbd_indicators:
        print(f"   {indicator}")
    
    # 计算得分
    rgbd_count = sum(1 for ind in rgbd_indicators if ind.startswith("✅"))
    total_count = len(rgbd_indicators)
    
    print(f"\n🎯 RGBD SUPPORT SCORE: {rgbd_count}/{total_count}")
    print(f"   {get_score_description(rgbd_count, total_count)}")
    
    return rgbd_count, total_count


def apply_rgbd_modifications(cfg):
    """应用RGBD修改（模拟demo中的修改）"""
    import copy
    modified_cfg = copy.deepcopy(cfg)
    
    print("   🔧 Setting encoder input channels to 6...")
    # 1. 设置编码器输入通道为6
    if hasattr(modified_cfg.MODEL, 'ENCODER'):
        modified_cfg.MODEL.ENCODER.IN_CHANS = 6
        if hasattr(modified_cfg.MODEL.ENCODER, 'in_chans'):
            modified_cfg.MODEL.ENCODER.in_chans = 6
    
    print("   🔧 Extending normalization parameters to 6 channels...")
    # 2. 扩展数据归一化参数到6通道
    if hasattr(modified_cfg.DATA, 'MEAN') and len(modified_cfg.DATA.MEAN) == 3:
        rgb_mean = list(modified_cfg.DATA.MEAN)
        rgb_std = list(modified_cfg.DATA.STD)
        
        # 为深度通道添加归一化参数
        depth_mean = [0.5, 0.5, 0.5]  # 深度通道使用0.5作为均值
        depth_std = [0.5, 0.5, 0.5]   # 深度通道使用0.5作为标准差
        
        # 扩展到6通道：RGB + Depth
        modified_cfg.DATA.MEAN = rgb_mean + depth_mean
        modified_cfg.DATA.STD = rgb_std + depth_std
    
    print("   🔧 Ensuring multi-modal vision is enabled...")
    # 3. 确保多模态视觉开启
    if hasattr(modified_cfg.DATA, 'MULTI_MODAL_VISION'):
        modified_cfg.DATA.MULTI_MODAL_VISION = True
    
    print("   ✅ Modifications applied successfully!")
    
    return modified_cfg


def get_score_description(rgbd_count, total_count):
    """获取得分描述"""
    if rgbd_count == total_count:
        return "🟢 FULL RGBD SUPPORT - Model fully configured for RGB-D input"
    elif rgbd_count >= total_count * 0.6:
        return "🟡 PARTIAL RGBD SUPPORT - Model may work with RGBD but not optimal"
    else:
        return "🔴 LIMITED RGBD SUPPORT - Model primarily designed for RGB-only"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Check SUTrack RGBD configuration.')
    parser.add_argument('tracker_param', type=str, help='Tracker parameter name (e.g., sutrack_b224)')
    
    args = parser.parse_args()
    
    check_rgbd_configuration(args.tracker_param)


if __name__ == '__main__':
    main()
