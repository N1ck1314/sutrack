#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 D435i 做 SUTrack 在线 RGB-Depth 跟踪 Demo

运行方式：
  1) conda activate sutrack
  2) 确保：export PYTHONPATH=/绝对路径/SUTrack:$PYTHONPATH
  3) python mydemo.py

操作说明：
  - 按 's' 键：选取初始目标（用鼠标框选）
  - 按 ESC：退出
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import os
import sys

# 添加路径以导入 SUTrack 模块
env_path = os.path.join(os.path.dirname(__file__), '..')
if (env_path not in sys.path):
    sys.path.append(env_path)

# ========= 1. RealSense 相机部分 =========

def create_realsense_pipeline():
    """创建并启动 RealSense pipeline，并对齐深度到彩色坐标系。"""
    pipeline = rs.pipeline()
    config = rs.config()

    # 根据自己需要调分辨率 / FPS（建议和 GPU 带宽权衡）
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # 深度对齐到彩色
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 深度 scale（单位：米 / depth_unit）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print(f"[INFO] RealSense started. depth_scale = {depth_scale:.6f} m/unit")

    return pipeline, align, depth_scale


def grab_rgbd(pipeline, align, timeout_ms=5000, max_retries=3):
    """从 RealSense 获取一帧对齐后的 RGB + Depth。"""
    for retry in range(max_retries):
        try:
            frames = pipeline.wait_for_frames(timeout_ms)
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                if retry < max_retries - 1:
                    print(f"[WARNING] Invalid frames, retry {retry + 1}/{max_retries}")
                    continue
                return None, None

            color_image = np.asanyarray(color_frame.get_data())    # (H,W,3) BGR uint8
            depth_image = np.asanyarray(depth_frame.get_data())    # (H,W)   uint16

            return color_image, depth_image
        
        except RuntimeError as e:
            if retry < max_retries - 1:
                print(f"[WARNING] Frame timeout, retry {retry + 1}/{max_retries}: {e}")
                time.sleep(0.1)  # 短暂等待后重试
            else:
                print(f"[ERROR] Failed to get frames after {max_retries} retries: {e}")
                raise
    
    return None, None


# ========= 2. 预处理：RGB + Depth =========

def preprocess_rgb_depth(color_bgr, depth_raw, depth_scale, max_dist_m=5.0):
    """
    color_bgr: (H,W,3) uint8, BGR
    depth_raw: (H,W)   uint16, raw depth
    depth_scale: D435i 深度单位到米的比例（一般 ~0.001）
    返回：
      color_rgb_uint8: (H,W,3) uint8 [0,255]  # RGB 3通道 - 供SUTrack使用
      depth_3ch_uint8: (H,W,3) uint8 [0,255]  # 深度信息复制成3通道 - 供SUTrack使用
      color_rgb_float: (H,W,3) float32 [0,1]  # RGB float版本 - 供可视化使用
      depth_3ch_float: (H,W,3) float32 [0,1]  # 深度float版本 - 供可视化使用
    
    SUTrack 为什么需要6通道？
    - 通道 0-2: RGB 彩色信息 (红、绿、蓝)
    - 通道 3-5: 深度信息 (复制3次，保持与RGB相同的维度结构)
    
    ⚠️ 重要：SUTrack的Preprocessor期望uint8 [0-255]输入，内部会除以255归一化
    """
    # 1) BGR -> RGB, uint8 [0,255] for SUTrack
    color_rgb_uint8 = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # 2) raw depth -> meters -> uint8 [0,255]
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    depth_m = np.clip(depth_m, 0.0, max_dist_m)
    depth_norm = depth_m / max_dist_m  # [0,1]
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)

    # 3) 扩展成3通道 uint8版本 for SUTrack
    depth_3ch_uint8 = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=2)
    
    # 4) 同时准备float版本供可视化使用（如果需要）
    color_rgb_float = color_rgb_uint8.astype(np.float32) / 255.0
    depth_3ch_float = np.stack([depth_norm, depth_norm, depth_norm], axis=2)

    return color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float


# ========= 3. SUTrack 接口封装（使用标准 Tracker 类） =========

class SUTrackOnlineTracker:
    """
    在线跟踪封装：
      - 使用标准的 lib.test.evaluation.tracker.Tracker 类
      - 与 test.py 中的实现方式完全一致
    """

    def __init__(self, tracker_name="sutrack", tracker_param="sutrack_b224", dataset_name='demo', 
                 checkpoint_path=None):
        """
        使用标准 Tracker 类创建跟踪器
        
        Args:
            tracker_name: 跟踪器名称
            tracker_param: 参数配置名
            dataset_name: 数据集名称
            checkpoint_path: 权重文件路径（如果不指定，使用默认路径）
        """
        print(f"[INFO] Creating tracker: {tracker_name} with param: {tracker_param}")
        if checkpoint_path:
            print(f"[INFO] Using checkpoint: {checkpoint_path}")
        
        try:
            # 导入标准的 Tracker 类（与 test.py 相同）
            from lib.test.evaluation.tracker import Tracker
            
            # 创建 Tracker 实例（这只是个包装器）
            tracker_wrapper = Tracker(tracker_name, tracker_param, dataset_name, run_id=None)
            
            # 获取参数
            params = tracker_wrapper.get_parameters()
            
            # 如果指定了 checkpoint 路径，覆盖默认配置
            if checkpoint_path:
                params.checkpoint = checkpoint_path
                print(f"[INFO] Overriding checkpoint path to: {checkpoint_path}")
            
            # 添加缺失的参数（避免 AttributeError）
            if not hasattr(params, 'debug'):
                params.debug = 0  # 0 = 不调试, 1 = 显示调试信息
            
            # 🔧 禁用文本模态和CLIP
            print(f"[INFO] Disabling text modality and CLIP...")
            self._disable_text_modality(params)
            
            # 🔍 检查模型输入通道数
            print(f"[INFO] Checking model input configuration...")
            self._check_model_channels(params)
            
            # 🎯 优化：启用模板更新以提高跟踪精度
            print("[INFO] Applying tracking optimizations...")
            # 覆盖配置文件中的参数，启用模板更新
            params.cfg.TEST.UPDATE_INTERVALS.DEFAULT = 25       # 每25帧更新一次模板
            params.cfg.TEST.UPDATE_THRESHOLD.DEFAULT = 0.85     # 🔒 提高阈值到0.85，只在非常确信时更新
            params.cfg.TEST.NUM_TEMPLATES = 2                   # 使用2个模板（当前帧+历史帧）
            print(f"[INFO] Template update enabled: interval=25, threshold=0.85 (strict), num_templates=2")
            print(f"[INFO] ⚠️  Conservative update: Only update when confidence > 0.85 to prevent drift")
            
            # 启用调试模式以显示模板更新信息
            params.debug = 0  # 保持为0，我们会在wrapper中添加调试
            
            # 创建真正的跟踪器实例（与 run_sequence 中的逻辑一致）
            self.tracker = tracker_wrapper.create_tracker(params)
            
            self.initialized = False
            self.last_bbox = None
            self.last_confidence = 0.0  # 用于显示跟踪置信度
            
            print("[INFO] Tracker created successfully using standard Tracker class.")
            
        except Exception as e:
            print(f"[ERROR] 创建 Tracker 失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _disable_text_modality(self, params):
        """禁用文本模态和CLIP相关功能"""
        cfg = params.cfg
        
        print("[INFO] 🔧 Configuring vision-only mode...")
        
        # 禁用文本编码器
        if hasattr(cfg.MODEL, 'TEXT_ENCODER'):
            print("[INFO]    - Disabling text encoder...")
            cfg.MODEL.TEXT_ENCODER = None  # 完全禁用文本编码器
        
        # 禁用多模态语言功能
        if hasattr(cfg.DATA, 'MULTI_MODAL_LANGUAGE'):
            print("[INFO]    - Disabling multi-modal language in DATA...")
            cfg.DATA.MULTI_MODAL_LANGUAGE = False
        
        # 禁用所有数据集的NLP功能
        if hasattr(cfg.DATA, 'USE_NLP'):
            print("[INFO]    - Disabling NLP for all datasets...")
            for dataset_key in cfg.DATA.USE_NLP:
                cfg.DATA.USE_NLP[dataset_key] = False
        
        # 训练时也禁用语言模态（即使这是推理）
        if hasattr(cfg.TRAIN, 'TYPE') and 'text' in cfg.TRAIN.TYPE:
            print("[INFO]    - Adjusting training type to vision-only...")
            cfg.TRAIN.TYPE = 'vision_only'
        
        # 测试时禁用语言模态
        if hasattr(cfg.TEST, 'MULTI_MODAL_LANGUAGE'):
            print("[INFO]    - Disabling language modality in TEST...")
            if isinstance(cfg.TEST.MULTI_MODAL_LANGUAGE, dict):
                for key in cfg.TEST.MULTI_MODAL_LANGUAGE:
                    cfg.TEST.MULTI_MODAL_LANGUAGE[key] = False
            else:
                cfg.TEST.MULTI_MODAL_LANGUAGE = False
        
        # 测试时禁用NLP
        if hasattr(cfg.TEST, 'USE_NLP'):
            print("[INFO]    - Disabling NLP in TEST...")
            if isinstance(cfg.TEST.USE_NLP, dict):
                for key in cfg.TEST.USE_NLP:
                    cfg.TEST.USE_NLP[key] = False
            else:
                cfg.TEST.USE_NLP = False
        
        # 确保只使用视觉模态
        if hasattr(cfg.TEST, 'MULTI_MODAL_VISION'):
            print("[INFO]    - Ensuring vision modality is enabled...")
            if isinstance(cfg.TEST.MULTI_MODAL_VISION, dict):
                for key in cfg.TEST.MULTI_MODAL_VISION:
                    cfg.TEST.MULTI_MODAL_VISION[key] = True
            else:
                cfg.TEST.MULTI_MODAL_VISION = True
        
        # 如果有数据预处理相关的设置
        if hasattr(cfg.DATA, 'MULTI_MODAL_VISION'):
            print("[INFO]    - Enabling vision processing...")
            cfg.DATA.MULTI_MODAL_VISION = True
        
        # 🔧 强制启用RGBD支持
        print("[INFO]    - Configuring RGBD support...")
        self._force_rgbd_support(cfg)
        
        print("[SUCCESS] ✅ Vision-only mode configured - Text/CLIP bypassed")
    
    def _force_rgbd_support(self, cfg):
        """强制配置RGBD支持"""
        
        # 1. 设置编码器输入通道为6
        if hasattr(cfg.MODEL, 'ENCODER'):
            print("[INFO]       - Setting encoder input channels to 6...")
            cfg.MODEL.ENCODER.IN_CHANS = 6  # 强制设置为6通道
            if hasattr(cfg.MODEL.ENCODER, 'in_chans'):
                cfg.MODEL.ENCODER.in_chans = 6
        
        # 2. 扩展数据归一化参数到6通道
        if hasattr(cfg.DATA, 'MEAN') and len(cfg.DATA.MEAN) == 3:
            print("[INFO]       - Extending normalization parameters to 6 channels...")
            # RGB通道的归一化参数
            rgb_mean = list(cfg.DATA.MEAN)  # [0.485, 0.456, 0.406]
            rgb_std = list(cfg.DATA.STD)    # [0.229, 0.224, 0.225]
            
            # 为深度通道添加归一化参数（使用适合深度的参数）
            # 深度通道使用不同的归一化参数，因为深度值分布与RGB不同
            depth_mean = [0.5, 0.5, 0.5]   # 深度归一化到[0,1]，所以均值用0.5
            depth_std = [0.5, 0.5, 0.5]    # 深度标准差用0.5
            
            # 扩展到6通道：RGB + Depth
            cfg.DATA.MEAN = rgb_mean + depth_mean
            cfg.DATA.STD = rgb_std + depth_std
            
            print(f"[INFO]       - New MEAN (RGB+Depth): {cfg.DATA.MEAN}")
            print(f"[INFO]       - New STD (RGB+Depth): {cfg.DATA.STD}")
        
        # 3. 确保多模态视觉开启
        if hasattr(cfg.DATA, 'MULTI_MODAL_VISION'):
            cfg.DATA.MULTI_MODAL_VISION = True
        
        print("[INFO]       - RGBD support configured successfully")

    def _check_model_channels(self, params):
        """检查模型输入通道配置"""
        cfg = params.cfg
        
        print("\n" + "="*60)
        print("[🔍 MODEL CHANNEL ANALYSIS]")
        print("="*60)
        
        # 检查编码器输入通道数
        if hasattr(cfg.MODEL, 'ENCODER'):
            encoder_cfg = cfg.MODEL.ENCODER
            print(f"[INFO] Encoder type: {getattr(encoder_cfg, 'TYPE', 'Unknown')}")
            
            # 检查是否有输入通道配置
            input_channels = None
            for attr in ['IN_CHANS', 'INPUT_CHANNELS', 'in_chans']:
                if hasattr(encoder_cfg, attr):
                    input_channels = getattr(encoder_cfg, attr)
                    print(f"[INFO] Found input channels config: {attr} = {input_channels}")
                    break
            
            if input_channels is None:
                print("[WARNING] No explicit input channel configuration found")
                print("[INFO] Will check default model behavior...")
            elif input_channels == 6:
                print("[SUCCESS] ✅ Model configured for RGBD input (6 channels)")
                print("[INFO]    - Channels 0-2: RGB")
                print("[INFO]    - Channels 3-5: Depth (replicated)")
            elif input_channels == 3:
                print("[WARNING] ⚠️  Model configured for RGB-only input (3 channels)")
                print("[INFO] This may not utilize depth information properly")
            else:
                print(f"[WARNING] ⚠️  Unexpected channel count: {input_channels}")
        
        # 检查数据配置
        if hasattr(cfg, 'DATA'):
            data_cfg = cfg.DATA
            multi_modal = getattr(data_cfg, 'MULTI_MODAL_VISION', 'Unknown')
            print(f"[INFO] Multi-modal vision enabled: {multi_modal}")
            
            # 检查数据预处理配置
            if hasattr(data_cfg, 'MEAN') and hasattr(data_cfg, 'STD'):
                mean = data_cfg.MEAN
                std = data_cfg.STD
                print(f"[INFO] Data normalization - MEAN: {mean}")
                print(f"[INFO] Data normalization - STD: {std}")
                
                if len(mean) == 6 and len(std) == 6:
                    print("[SUCCESS] ✅ Normalization configured for 6 channels (RGBD)")
                elif len(mean) == 3 and len(std) == 3:
                    print("[INFO] Normalization configured for 3 channels (RGB)")
                    print("[WARNING] ⚠️  This suggests RGB-only processing")
        
        print("="*60 + "\n")

    def initialize(self, color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, init_bbox):
        """
        初始化跟踪器。

        color_rgb_uint8: (H,W,3) uint8 [0,255]  # RGB 3通道 - 供SUTrack使用
        depth_3ch_uint8: (H,W,3) uint8 [0,255]  # 深度 3通道 - 供SUTrack使用
        color_rgb_float: (H,W,3) float32 [0,1]  # RGB float版本（暂未使用）
        depth_3ch_float: (H,W,3) float32 [0,1]  # 深度float版本（暂未使用）
        init_bbox: [x, y, w, h]，图像坐标
        
        最终输入给SUTrack的是6通道uint8图像：
        - 前3通道：RGB彩色信息 [0-255]
        - 后3通道：深度信息（重复3次）[0-255]
        """
        print("\n" + "="*60)
        print("[INFO] Starting tracker initialization...")
        print("="*60)
        
        try:
            # 合并 RGB(3通道) 和 Depth(3通道) = 总共6通道 uint8
            rgbd_image = np.concatenate([color_rgb_uint8, depth_3ch_uint8], axis=2)  # (H,W,6) uint8
            
            # 🔍 详细分析输入数据
            self._analyze_input_data(color_rgb_uint8, depth_3ch_uint8, rgbd_image)
            
            print(f"[DEBUG] RGBD image shape: {rgbd_image.shape}, dtype: {rgbd_image.dtype}")
            print(f"[DEBUG] Value range: [{rgbd_image.min()}, {rgbd_image.max()}]")
            print(f"[DEBUG] Init bbox: {init_bbox}")
            
            # 准备初始化信息
            init_info = {
                'init_bbox': init_bbox,  # [x, y, w, h]
            }
            
            print(f"[DEBUG] Calling tracker.initialize()...")
            
            # 调用跟踪器初始化（使用 tracker_impl）
            out = self.tracker.initialize(rgbd_image, init_info)
            
            print(f"[DEBUG] Tracker.initialize() returned: {type(out)}")
            
            self.initialized = True
            self.last_bbox = init_bbox
            
            print("\n" + "="*60)
            print(f"[SUCCESS] ✓ Tracker initialized successfully!")
            print(f"[SUCCESS] ✓ Init bbox: {init_bbox}")
            print("="*60 + "\n")
            
        except Exception as e:
            print("\n" + "="*60)
            print(f"[ERROR] ✗ Tracker initialization FAILED!")
            print(f"[ERROR] ✗ Error: {e}")
            print("="*60)
            import traceback
            traceback.print_exc()
            print("="*60 + "\n")
            self.initialized = False

    def _analyze_input_data(self, color_rgb, depth_3ch, rgbd_combined):
        """分析输入数据特征"""
        print("\n" + "="*60)
        print("[🔍 INPUT DATA ANALYSIS]")
        print("="*60)
        
        # RGB分析
        print(f"[RGB] Shape: {color_rgb.shape}, dtype: {color_rgb.dtype}")
        print(f"[RGB] Value range: [{color_rgb.min():.1f}, {color_rgb.max():.1f}]")
        rgb_mean = color_rgb.mean(axis=(0,1))
        print(f"[RGB] Channel means: R={rgb_mean[0]:.1f}, G={rgb_mean[1]:.1f}, B={rgb_mean[2]:.1f}")
        
        # 深度分析
        print(f"[DEPTH] Shape: {depth_3ch.shape}, dtype: {depth_3ch.dtype}")
        print(f"[DEPTH] Value range: [{depth_3ch.min():.1f}, {depth_3ch.max():.1f}]")
        depth_mean = depth_3ch.mean(axis=(0,1))
        print(f"[DEPTH] Channel means: D1={depth_mean[0]:.1f}, D2={depth_mean[1]:.1f}, D3={depth_mean[2]:.1f}")
        
        # 检查深度是否有效（非全零）
        depth_nonzero_ratio = np.count_nonzero(depth_3ch) / depth_3ch.size
        print(f"[DEPTH] Non-zero ratio: {depth_nonzero_ratio:.3f}")
        
        if depth_nonzero_ratio < 0.1:
            print("[WARNING] ⚠️  Very low depth data! Possible issues:")
            print("          - Depth sensor not working properly")
            print("          - Target too far (>5m)")
            print("          - Depth alignment issues")
        elif depth_nonzero_ratio > 0.5:
            print("[SUCCESS] ✅ Good depth coverage")
        else:
            print("[INFO] Moderate depth coverage")
        
        # RGBD组合分析
        print(f"[RGBD] Combined shape: {rgbd_combined.shape}")
        print(f"[RGBD] Total channels: {rgbd_combined.shape[2]}")
        
        if rgbd_combined.shape[2] == 6:
            print("[SUCCESS] ✅ 6-channel RGBD input prepared correctly")
            print("[INFO]    - Channels 0-2: RGB color information")
            print("[INFO]    - Channels 3-5: Depth information (replicated)")
            
            # 检查RGB和深度的区别
            rgb_part = rgbd_combined[:,:,:3]
            depth_part = rgbd_combined[:,:,3:]
            
            # 计算RGB和深度部分的差异
            if not np.array_equal(rgb_part, depth_part):
                print("[SUCCESS] ✅ RGB and depth channels contain different information")
            else:
                print("[ERROR] ❌ RGB and depth channels are identical! Check preprocessing.")
        else:
            print(f"[ERROR] ❌ Unexpected channel count: {rgbd_combined.shape[2]}")
        
        print("="*60 + "\n")

    def track(self, color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, frame_id):
        """
        进行单帧跟踪，返回 bbox = [x, y, w, h]。
        
        同样将RGB(3通道) + Depth(3通道) = 6通道uint8输入给网络
        """
        if not self.initialized:
            print("[WARNING] Tracker not initialized. Returning default bbox.")
            return [0, 0, 50, 50]

        try:
            # 合并为6通道uint8输入
            rgbd_image = np.concatenate([color_rgb_uint8, depth_3ch_uint8], axis=2)  # (H,W,6) uint8
            
            # 🔍 定期检查输入数据质量
            if frame_id % 100 == 0:  # 每100帧检查一次
                self._monitor_input_quality(rgbd_image, frame_id)
            
            # 调用跟踪方法（使用 tracker_impl）
            out = self.tracker.track(rgbd_image)
            
            # 打印输出结构（仅第一次）
            if not hasattr(self, '_track_debug_done'):
                print(f"[DEBUG] Track output type: {type(out)}")
                if isinstance(out, dict):
                    print(f"[DEBUG] Track output keys: {list(out.keys())}")
                self._track_debug_done = True
            
            # 从输出中提取 bbox 和置信度
            bbox, confidence = self._extract_bbox_from_output(out)
            
            # 保存置信度供显示使用
            self.last_confidence = confidence
            
            # 🔒 安全检查：检测跟踪漂移并阻止错误模板更新
            if not hasattr(self, '_confidence_history'):
                self._confidence_history = []
            self._confidence_history.append(confidence)
            if len(self._confidence_history) > 10:
                self._confidence_history.pop(0)
            
            # 计算近10帧的平均置信度
            avg_confidence = sum(self._confidence_history) / len(self._confidence_history)
            
            # 🚨 漂移检测：如果置信度持续下降或波动大，可能已经漂移
            is_drifting = False
            if len(self._confidence_history) >= 5:
                recent_5 = self._confidence_history[-5:]
                # 检测1: 最近5帧平均置信度 < 0.6
                if sum(recent_5) / 5 < 0.6:
                    is_drifting = True
                # 检测2: 置信度波动过大（标准差 > 0.2）
                mean_conf = sum(recent_5) / 5
                variance = sum((x - mean_conf) ** 2 for x in recent_5) / 5
                std_dev = variance ** 0.5
                if std_dev > 0.2:
                    is_drifting = True
            
            # 🔍 检测模板是否更新（通过比较模板列表长度变化）
            if hasattr(self.tracker, 'template_list'):
                current_template_count = len(self.tracker.template_list)
                if not hasattr(self, '_last_template_count'):
                    self._last_template_count = current_template_count
                elif current_template_count != self._last_template_count:
                    status = "✅ SAFE" if not is_drifting else "⚠️  RISKY"
                    print(f"\n[🔄 TEMPLATE UPDATE] Frame {self.tracker.frame_id}: "
                          f"Conf={confidence:.3f}, AvgConf={avg_confidence:.3f}, Status={status}")
                    if is_drifting:
                        print(f"   ⚠️  Warning: Possible drift detected! Consider re-initialization.")
                    self._last_template_count = current_template_count
            
            # 🔍 定期打印跟踪状态
            if hasattr(self.tracker, 'frame_id'):
                frame = self.tracker.frame_id
                if frame % 25 == 0:  # 每25帧打印一次（对应更新间隔）
                    update_interval = self.tracker.update_intervals
                    update_threshold = self.tracker.update_threshold
                    will_check = "Will check" if frame % update_interval == 0 else "No check"
                    passed_threshold = "✓" if confidence > update_threshold else "✗"
                    drift_status = "⚠️ DRIFTING" if is_drifting else "✅ Stable"
                    print(f"[📊 Status] Frame {frame}: Conf={confidence:.3f} {passed_threshold}, "
                          f"AvgConf={avg_confidence:.3f}, {drift_status}, "
                          f"Update: {will_check} (threshold={update_threshold:.2f})")
            
            if bbox is not None and len(bbox) == 4:
                # 验证 bbox 是否合理
                if all(isinstance(v, (int, float)) for v in bbox) and bbox[2] > 0 and bbox[3] > 0:
                    self.last_bbox = bbox
                else:
                    print(f"[WARNING] Invalid bbox values: {bbox}, using last bbox")
                    bbox = self.last_bbox if self.last_bbox is not None else [0, 0, 50, 50]
            else:
                print(f"[WARNING] Invalid bbox format: {bbox}, using last bbox")
                bbox = self.last_bbox if self.last_bbox is not None else [0, 0, 50, 50]
            
            return bbox
            
        except Exception as e:
            print(f"[WARNING] Tracking failed: {e}")
            import traceback
            traceback.print_exc()
            # 兜底方案：返回上一帧的 bbox
            return self.last_bbox if self.last_bbox is not None else [0, 0, 50, 50]

    def _monitor_input_quality(self, rgbd_image, frame_id):
        """监控输入数据质量"""
        rgb_part = rgbd_image[:,:,:3]
        depth_part = rgbd_image[:,:,3:]
        
        # 检查深度数据质量
        depth_nonzero = np.count_nonzero(depth_part) / depth_part.size
        rgb_var = np.var(rgb_part)
        depth_var = np.var(depth_part)
        
        print(f"\n[📊 QUALITY CHECK - Frame {frame_id}]")
        print(f"   Depth coverage: {depth_nonzero:.3f}")
        print(f"   RGB variance: {rgb_var:.1f}")
        print(f"   Depth variance: {depth_var:.1f}")
        
        if depth_nonzero < 0.2:
            print("   ⚠️  Warning: Low depth coverage")
        if depth_var < 100:
            print("   ⚠️  Warning: Low depth variation")

    def _extract_bbox_from_output(self, output):
        """从跟踪器输出中提取bbox和置信度"""
        bbox = None
        confidence = 0.0
        
        if isinstance(output, dict):
            # 提取 bbox
            for key in ['target_bbox', 'bbox', 'pred_bbox', 'box']:
                if key in output:
                    bbox = output[key]
                    break
            else:
                print(f"[DEBUG] Output dict keys: {list(output.keys())}")
                return self.last_bbox, 0.0
            
            # 提取置信度
            for key in ['best_score', 'confidence', 'score', 'conf']:
                if key in output:
                    confidence = output[key]
                    if isinstance(confidence, torch.Tensor):
                        confidence = confidence.item()
                    break
                    
        elif hasattr(output, 'target_bbox'):
            bbox = output.target_bbox
            confidence = getattr(output, 'best_score', 0.0)
        elif hasattr(output, 'bbox'):
            bbox = output.bbox
            confidence = getattr(output, 'score', 0.0)
        elif isinstance(output, (list, tuple, np.ndarray)) and len(output) == 4:
            bbox = output
        else:
            print(f"[DEBUG] Unexpected output type: {type(output)}")
            return self.last_bbox, 0.0
        
        # 确保bbox是正确的格式
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().cpu().numpy()
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        elif not isinstance(bbox, list):
            bbox = list(bbox) if bbox is not None else self.last_bbox
        
        return bbox, confidence


# 备选实现：如果上述方法失败，使用简化版本
class SimpleSUTracker:
    """简化版跟踪器实现"""
    
    def __init__(self):
        print("[INFO] Using simplified tracker (fallback)")
        self.initialized = False
        self.last_bbox = None
        
    def initialize(self, color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, init_bbox):
        self.last_bbox = init_bbox
        self.initialized = True
        print(f"[INFO] Simple tracker initialized with bbox={init_bbox}")
        
    def track(self, color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, frame_id):
        if not self.initialized:
            return [0, 0, 50, 50]
        
        # 简单的跟踪：保持固定位置（仅作为兜底）
        return self.last_bbox if self.last_bbox is not None else [0, 0, 50, 50]


# ========= 4. 主循环：从 D435i 拉流 + SUTrack 在线跟踪 =========
def main():
    print("[INFO] Initializing RealSense camera...")
    pipeline, align, depth_scale = create_realsense_pipeline()
    
    print("[INFO] Waiting for camera to stabilize...")
    # 等待相机稳定，丢弃前几帧
    for _ in range(30):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
        except:
            pass
    
    print("[INFO] Initializing SUTrack tracker...")
    
    # 🔧 修复：尝试多个配置，优先使用有checkpoint的
    tracker = None
    
    # 尝试的配置列表（按优先级排序）
    config_attempts = [
        ("sutrack", "sutrack_b224", "/home/nick/code/code.sutrack/SUTrack/checkpoints_backup/train/sutrack/sutrack_b224/SUTRACK_ep0180.pth.tar"),
        ("sutrack", "sutrack_t224", "/home/nick/code/code.sutrack/SUTrack/checkpoints_backup/train/sutrack/sutrack_t224/SUTRACK_ep0180.pth.tar"), 
        ("sutrack", "sutrack_b224", None),  # 不指定checkpoint路径
        ("sutrack", "sutrack_t224", None),  # 不指定checkpoint路径
    ]
    
    for tracker_name, tracker_param, checkpoint_path in config_attempts:
        if checkpoint_path and not os.path.exists(checkpoint_path):
            print(f"[INFO] Checkpoint not found: {os.path.basename(checkpoint_path)}, skipping...")
            continue
            
        try:
            print(f"\n[INFO] Attempting to create tracker: {tracker_name} + {tracker_param}")
            if checkpoint_path:
                print(f"[INFO] Using checkpoint: {os.path.basename(checkpoint_path)}")
            else:
                print(f"[INFO] Using default checkpoint path")
            
            tracker = SUTrackOnlineTracker(
                tracker_name=tracker_name,
                tracker_param=tracker_param,
                dataset_name='demo',
                checkpoint_path=checkpoint_path
            )
            
            print(f"[SUCCESS] ✅ Tracker created successfully with {tracker_param}!")
            break
            
        except FileNotFoundError as e:
            if "pretrained" in str(e):
                print(f"[WARNING] ⚠️  Missing pretrained encoder file: {e}")
                print(f"[INFO] 🔧 This config requires pretrained encoder, trying next config...")
            else:
                print(f"[WARNING] ⚠️  File not found: {e}")
            continue
        except Exception as e:
            print(f"[WARNING] ⚠️  Failed with {tracker_param}: {e}")
            continue
    
    # 如果所有标准方法都失败，使用简化版本
    if tracker is None:
        print("\n" + "="*60)
        print("[WARNING] ⚠️  All standard tracker creation methods failed!")
        print("[INFO] 🔧 Falling back to simplified tracker...")
        print("="*60)
        tracker = SimpleSUTracker()
    
    frame_id = 0
    init_bbox = None

    win_name = "SUTrack RGB-D Online Tracking (D435i)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print("[INFO] Starting tracking loop...")
    try:
        while True:
            try:
                color_bgr, depth_raw = grab_rgbd(pipeline, align, timeout_ms=2000, max_retries=2)
            except RuntimeError as e:
                print(f"[ERROR] Camera error: {e}")
                print("[INFO] Trying to reinitialize camera...")
                pipeline.stop()
                time.sleep(1)
                pipeline, align, depth_scale = create_realsense_pipeline()
                continue
            
            if color_bgr is None:
                print("[WARNING] Failed to get frame, skipping...")
                continue

            vis = color_bgr.copy()

            # 第一次：等待用户按 's' 选框
            if init_bbox is None:
                cv2.putText(vis, "Press 's' to select ROI, ESC to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                cv2.imshow(win_name, vis)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break

                if key == ord('s'):
                    # 暂停当前画面，让你用鼠标框出 ROI
                    print("\n[INFO] Please select ROI...")
                    roi = cv2.selectROI(win_name, vis, fromCenter=False, showCrosshair=True)
                    x, y, w, h = roi
                    print(f"[INFO] ROI selected: x={x}, y={y}, w={w}, h={h}")
                    
                    if (w > 0 and h > 0):
                        init_bbox = [float(x), float(y), float(w), float(h)]
                        # 做一次预处理并初始化 tracker
                        print("[INFO] Preprocessing image...")
                        color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float = preprocess_rgb_depth(
                            color_bgr, depth_raw, depth_scale
                        )
                        print(f"[INFO] Image preprocessed. Shape: {color_rgb_uint8.shape}")
                        
                        # 初始化跟踪器
                        tracker.initialize(color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, init_bbox)
                        
                        # 检查是否初始化成功
                        if tracker.initialized:
                            print(f"\n[SUCCESS] ✓✓✓ Tracking started! Target bbox: {init_bbox}\n")
                        else:
                            print(f"\n[ERROR] ✗✗✗ Initialization failed! Please check errors above.\n")
                            init_bbox = None
                    else:
                        init_bbox = None
                        print("[WARNING] Invalid ROI selection (w or h is 0), please try again")
                continue

            # 之后：正常跟踪

            # 之后：正常跟踪
            t0 = time.time()
            color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float = preprocess_rgb_depth(
                color_bgr, depth_raw, depth_scale
            )
            bbox = tracker.track(color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, frame_id)
            t1 = time.time()

            x, y, w, h = bbox
            
            # 每50帧打印一次bbox信息用于调试
            if frame_id % 50 == 0:
                print(f"[DEBUG] Frame {frame_id}: bbox=[{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}], FPS={1.0/(t1-t0):.1f}")
            
            # 🔒 检测到漂移时，修改bbox颜色为红色警告
            box_color = (0, 255, 0)  # 默认绿色
            if hasattr(tracker, '_confidence_history') and len(tracker._confidence_history) >= 5:
                recent_avg = sum(tracker._confidence_history[-5:]) / 5
                if recent_avg < 0.6:
                    box_color = (0, 0, 255)  # 漂移时显示红色框
            
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(vis, p1, p2, box_color, 2)

            fps = 1.0 / max(1e-6, (t1 - t0))
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示帧号
            cv2.putText(vis, f"Frame: {frame_id}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示跟踪置信度（帮助诊断跟踪质量）
            if hasattr(tracker, 'last_confidence'):
                conf = tracker.last_confidence
                # 置信度颜色：高(绿) -> 中(黄) -> 低(红)
                if conf > 0.85:
                    conf_color = (0, 255, 0)  # 绿色 - 很稳定
                elif conf > 0.7:
                    conf_color = (0, 255, 255)  # 黄色 - 一般
                elif conf > 0.5:
                    conf_color = (0, 165, 255)  # 橙色 - 不稳定
                else:
                    conf_color = (0, 0, 255)  # 红色 - 可能漂移
                cv2.putText(vis, f"Conf: {conf:.3f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
            
            # 显示漂移警告
            if hasattr(tracker, '_confidence_history') and len(tracker._confidence_history) >= 5:
                recent_avg = sum(tracker._confidence_history[-5:]) / 5
                if recent_avg < 0.6:
                    cv2.putText(vis, "WARNING: Possible Drift!", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(vis, "Press 's' to re-select", (10, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 显示深度图（用于验证深度信息）
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            cv2.rectangle(depth_colormap, p1, p2, (0, 255, 0), 2)
            cv2.putText(depth_colormap, "Depth", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(win_name, vis)
            cv2.imshow("Depth View", depth_colormap)  # 显示深度图窗口

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # 🔄 跟踪过程中按's'重新初始化
                print("\n" + "="*60)
                print("[INFO] 🔄 Re-initialization requested...")
                print("[INFO] Please select new target ROI...")
                print("="*60)
                
                # 暂停当前画面，让用户重新选择
                roi = cv2.selectROI(win_name, vis, fromCenter=False, showCrosshair=True)
                x_new, y_new, w_new, h_new = roi
                
                if w_new > 0 and h_new > 0:
                    print(f"[INFO] New ROI selected: x={x_new}, y={y_new}, w={w_new}, h={h_new}")
                    init_bbox = [float(x_new), float(y_new), float(w_new), float(h_new)]
                    
                    # 重新初始化跟踪器
                    print("[INFO] Re-initializing tracker...")
                    color_rgb_uint8_new, depth_3ch_uint8_new, color_rgb_float_new, depth_3ch_float_new = preprocess_rgb_depth(
                        color_bgr, depth_raw, depth_scale
                    )
                    tracker.initialize(color_rgb_uint8_new, depth_3ch_uint8_new, color_rgb_float_new, depth_3ch_float_new, init_bbox)
                    
                    if tracker.initialized:
                        print("\n" + "="*60)
                        print(f"[SUCCESS] ✅ Tracker re-initialized successfully!")
                        print(f"[SUCCESS] ✅ New target bbox: {init_bbox}")
                        print("="*60 + "\n")
                        frame_id = 0  # 重置帧计数
                    else:
                        print("[ERROR] Re-initialization failed!")
                else:
                    print("[WARNING] Invalid ROI, keeping current tracking...")
                
                continue  # 继续跟踪循环

            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Cleaning up...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()

