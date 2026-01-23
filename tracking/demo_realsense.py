#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 test.py 的 D435i 实时跟踪 Demo

模仿 test.py 的标准用法，使用 Tracker 类进行实时RGB-D跟踪

运行方式：
  1) conda activate sutrack
  2) 确保：export PYTHONPATH=/绝对路径/SUTrack:$PYTHONPATH
  3) python demo_realsense.py sutrack sutrack_b224 --debug 1

操作说明：
  - 按 's' 键：选取初始目标（用鼠标框选）
  - 按 'r' 键：重新初始化
  - 按 ESC：退出
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import pyrealsense2 as rs

# 添加路径以导入 SUTrack 模块
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker


class RealSenseSequence:
    """
    模拟数据集接口的 RealSense 相机序列
    让 Tracker 以为在处理标准数据集
    """
    
    def __init__(self):
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.frame_id = 0
        self.current_frame = None
        self.ground_truth = None  # 实时跟踪没有GT
        
        print("[INFO] Initializing RealSense camera...")
        self._init_camera()
        
    def _init_camera(self):
        """初始化 D435i 相机"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # 配置流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = self.pipeline.start(config)
        
        # 深度对齐到彩色
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        # 获取深度比例
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        print(f"[INFO] RealSense started. depth_scale = {self.depth_scale:.6f} m/unit")
        
        # 等待相机稳定
        print("[INFO] Waiting for camera to stabilize...")
        for _ in range(30):
            try:
                self.pipeline.wait_for_frames(timeout_ms=1000)
            except:
                pass
        
    def get_frame(self, timeout_ms=5000):
        """获取一帧数据，返回 6 通道 RGBD 图像"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None
                
            # 获取原始数据
            color_bgr = np.asanyarray(color_frame.get_data())  # (H,W,3) BGR uint8
            depth_raw = np.asanyarray(depth_frame.get_data())  # (H,W) uint16
            
            # 预处理为6通道 RGBD
            rgbd_image = self._preprocess_rgbd(color_bgr, depth_raw)
            
            self.current_frame = rgbd_image
            self.frame_id += 1
            
            return rgbd_image, color_bgr  # 返回处理后的RGBD和原始BGR(用于显示)
            
        except Exception as e:
            print(f"[ERROR] Failed to get frame: {e}")
            return None
    
    def _preprocess_rgbd(self, color_bgr, depth_raw, max_dist_m=5.0):
        """
        预处理为6通道RGBD图像
        
        Returns:
            rgbd_image: (H,W,6) uint8 [0-255]
            前3通道: RGB, 后3通道: Depth(重复3次)
        """
        # BGR -> RGB
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        
        # 深度处理: raw -> meters -> [0,1] -> [0,255]
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        depth_m = np.clip(depth_m, 0.0, max_dist_m)
        depth_norm = depth_m / max_dist_m  # [0,1]
        depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
        
        # 深度扩展为3通道
        depth_3ch = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=2)
        
        # 合并为6通道
        rgbd_image = np.concatenate([color_rgb, depth_3ch], axis=2)
        
        return rgbd_image
    
    def cleanup(self):
        """清理资源"""
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()


class RealtimeTracker:
    """
    实时跟踪器包装类
    模仿 test.py 中 run_dataset 的逻辑
    """
    
    def __init__(self, tracker_name, tracker_param, debug=0):
        self.tracker_name = tracker_name
        self.tracker_param = tracker_param
        self.debug = debug
        
        # 创建序列（相机接口）
        self.sequence = RealSenseSequence()
        
        # 创建跟踪器（模仿 test.py）
        print(f"[INFO] Creating tracker: {tracker_name} with param: {tracker_param}")
        self.tracker = Tracker(tracker_name, tracker_param, 'demo', run_id=None)
        
        # 获取参数并创建跟踪器实例
        params = self.tracker.get_parameters()
        params.debug = debug
        
        # 🔧 禁用文本模态和CLIP
        print(f"[INFO] Disabling text modality and CLIP...")
        self._disable_text_modality(params)
        
        # 修复预训练权重路径问题
        print(f"[INFO] Checking and fixing pretrained paths...")
        self._fix_pretrained_paths(params)
        
        # 创建跟踪器实例（模仿 run_sequence）
        self.tracker_impl = self.tracker.create_tracker(params)
        
        print("[INFO] Tracker created successfully.")
        
        # 初始化状态
        self.initialized = False
        self.init_bbox = None
    
    def _disable_text_modality(self, params):
        """禁用文本模态和CLIP相关功能"""
        cfg = params.cfg
        
        # 禁用文本编码器
        if hasattr(cfg.MODEL, 'TEXT_ENCODER'):
            print("[INFO] 🔧 Disabling text encoder...")
            cfg.MODEL.TEXT_ENCODER.TYPE = None  # 禁用文本编码器
        
        # 禁用多模态语言功能
        if hasattr(cfg.DATA, 'MULTI_MODAL_LANGUAGE'):
            print("[INFO] 🔧 Disabling multi-modal language...")
            cfg.DATA.MULTI_MODAL_LANGUAGE = False
        
        # 禁用所有数据集的NLP功能
        if hasattr(cfg.DATA, 'USE_NLP'):
            print("[INFO] 🔧 Disabling NLP for all datasets...")
            for dataset_key in cfg.DATA.USE_NLP:
                cfg.DATA.USE_NLP[dataset_key] = False
        
        # 测试时禁用语言模态
        if hasattr(cfg.TEST, 'MULTI_MODAL_LANGUAGE'):
            print("[INFO] 🔧 Disabling language modality in TEST...")
            if hasattr(cfg.TEST.MULTI_MODAL_LANGUAGE, 'DEFAULT'):
                cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT = False
            else:
                cfg.TEST.MULTI_MODAL_LANGUAGE = {'DEFAULT': False}
        
        # 测试时禁用NLP
        if hasattr(cfg.TEST, 'USE_NLP'):
            print("[INFO] 🔧 Disabling NLP in TEST...")
            if hasattr(cfg.TEST.USE_NLP, 'DEFAULT'):
                cfg.TEST.USE_NLP.DEFAULT = False
            else:
                cfg.TEST.USE_NLP = {'DEFAULT': False}
        
        # 确保只使用视觉模态
        if hasattr(cfg.TEST, 'MULTI_MODAL_VISION'):
            print("[INFO] 🔧 Enabling vision-only mode...")
            if hasattr(cfg.TEST.MULTI_MODAL_VISION, 'DEFAULT'):
                cfg.TEST.MULTI_MODAL_VISION.DEFAULT = True
            else:
                cfg.TEST.MULTI_MODAL_VISION = {'DEFAULT': True}
        
        # 🔧 修复RGBD支持：强制设置6通道输入
        print("[INFO] 🔧 Configuring RGBD support...")
        self._force_rgbd_support(cfg)
        
        # 如果有任务相关的设置，也禁用语言任务
        if hasattr(cfg.MODEL, 'TASK_INDEX'):
            print("[INFO] 🔧 Adjusting task settings for vision-only...")
            # 保持视觉任务，禁用需要语言的任务
            # 这里不修改TASK_INDEX，因为模型架构可能依赖它
        
        print("[SUCCESS] ✅ Text modality and CLIP disabled - Vision-only mode enabled")
        
        # 🔍 检查模型是否支持RGBD
        self._check_rgbd_support(params)
    
    def _force_rgbd_support(self, cfg):
        """强制配置RGBD支持"""
        print("[INFO]    - Forcing RGBD configuration...")
        
        # 1. 设置编码器输入通道为6
        if hasattr(cfg.MODEL, 'ENCODER'):
            print("[INFO]    - Setting encoder input channels to 6...")
            cfg.MODEL.ENCODER.IN_CHANS = 6  # 强制设置为6通道
            if hasattr(cfg.MODEL.ENCODER, 'in_chans'):
                cfg.MODEL.ENCODER.in_chans = 6
        
        # 2. 扩展数据归一化参数到6通道
        if hasattr(cfg.DATA, 'MEAN') and len(cfg.DATA.MEAN) == 3:
            print("[INFO]    - Extending normalization parameters to 6 channels...")
            # RGB通道的归一化参数
            rgb_mean = cfg.DATA.MEAN
            rgb_std = cfg.DATA.STD
            
            # 为深度通道添加归一化参数（使用ImageNet的均值和方差）
            depth_mean = [0.485, 0.456, 0.406]  # 复用RGB的参数
            depth_std = [0.229, 0.224, 0.225]   # 复用RGB的参数
            
            # 扩展到6通道：RGB + Depth
            cfg.DATA.MEAN = rgb_mean + depth_mean
            cfg.DATA.STD = rgb_std + depth_std
            
            print(f"[INFO]    - New MEAN: {cfg.DATA.MEAN}")
            print(f"[INFO]    - New STD: {cfg.DATA.STD}")
        
        # 3. 确保多模态视觉开启
        if hasattr(cfg.DATA, 'MULTI_MODAL_VISION'):
            cfg.DATA.MULTI_MODAL_VISION = True
        
        print("[INFO]    - RGBD support configured successfully")
    
    def _check_rgbd_support(self, params):
        """检查模型RGBD支持情况"""
        cfg = params.cfg
        
        print("\n" + "="*50)
        print("[🔍 RGBD SUPPORT CHECK]")
        print("="*50)
        
        # 检查编码器配置
        if hasattr(cfg.MODEL, 'ENCODER'):
            encoder_type = getattr(cfg.MODEL.ENCODER, 'TYPE', 'Unknown')
            print(f"Encoder type: {encoder_type}")
            
            # 检查输入通道数
            input_channels = None
            for attr in ['in_chans', 'IN_CHANS', 'INPUT_CHANNELS']:
                if hasattr(cfg.MODEL.ENCODER, attr):
                    input_channels = getattr(cfg.MODEL.ENCODER, attr)
                    break
            
            if input_channels == 6:
                print("✅ Model supports 6-channel RGBD input")
            elif input_channels == 3:
                print("⚠️  Model configured for 3-channel RGB input")
                print("   Depth information may not be fully utilized")
            else:
                print(f"❓ Input channels: {input_channels}")
        
        # 检查预处理配置
        if hasattr(cfg.DATA, 'MEAN') and hasattr(cfg.DATA, 'STD'):
            mean_channels = len(cfg.DATA.MEAN)
            std_channels = len(cfg.DATA.STD)
            print(f"Normalization channels: MEAN={mean_channels}, STD={std_channels}")
            
            if mean_channels == 6 and std_channels == 6:
                print("✅ Preprocessing configured for RGBD")
            else:
                print("⚠️  Preprocessing may be RGB-only")
        
        print("="*50 + "\n")
    
    def _fix_pretrained_paths(self, params):
        """修复预训练权重路径问题"""
        cfg = params.cfg
        
        # 检查编码器预训练路径
        if hasattr(cfg.MODEL.ENCODER, 'PRETRAIN_TYPE'):
            pretrain_path = cfg.MODEL.ENCODER.PRETRAIN_TYPE
            print(f"[DEBUG] Original pretrain path: {pretrain_path}")
            
            # 如果是相对路径且文件不存在
            if not os.path.isabs(pretrain_path) and not os.path.exists(pretrain_path):
                # 尝试几个可能的位置
                possible_paths = [
                    f"/home/nick/code/code.sutrack/SUTrack/{pretrain_path}",
                    f"/home/nick/code/code.sutrack/SUTrack/pretrained/{os.path.basename(pretrain_path)}",
                    f"/home/nick/code/code.sutrack/SUTrack/checkpoints/{os.path.basename(pretrain_path)}",
                    f"/home/nick/code/code.sutrack/SUTrack/checkpoints_backup/{os.path.basename(pretrain_path)}",
                ]
                
                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        print(f"[INFO] ✅ Found pretrained file at: {path}")
                        cfg.MODEL.ENCODER.PRETRAIN_TYPE = path
                        found = True
                        break
                
                if not found:
                    print(f"[WARNING] ⚠️  Pretrained file not found: {pretrain_path}")
                    print(f"[INFO] 🔧 Disabling encoder pretraining...")
                    # 方案1: 设置为None禁用预训练
                    cfg.MODEL.ENCODER.PRETRAIN_TYPE = None
                    # 方案2: 或者设置一个空字符串
                    # cfg.MODEL.ENCODER.PRETRAIN_TYPE = ""
                    print(f"[INFO] ✅ Encoder will be initialized randomly")
            else:
                print(f"[INFO] ✅ Pretrained path exists: {pretrain_path}")
        
    def run_interactive(self):
        """运行交互式跟踪"""
        print("\n" + "="*60)
        print("Interactive Tracking Started!")
        print("Controls:")
        print("  's' - Select target (mouse selection)")
        print("  'r' - Re-initialize tracking") 
        print("  ESC - Exit")
        print("="*60 + "\n")
        
        win_name = f"SUTrack Real-time Tracking ({self.tracker_name})"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        
        frame_id = 0
        
        try:
            while True:
                # 获取帧
                result = self.sequence.get_frame()
                if result is None:
                    print("[WARNING] Failed to get frame, skipping...")
                    continue
                    
                rgbd_image, color_bgr = result
                vis = color_bgr.copy()
                
                # 第一次或未初始化：等待用户选择目标
                if not self.initialized:
                    cv2.putText(vis, "Press 's' to select target, ESC to quit",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
                    cv2.imshow(win_name, vis)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == ord('s'):
                        self._select_target(vis, rgbd_image, win_name)
                    continue
                
                # 正常跟踪
                t_start = time.time()
                
                # 调用跟踪器（模仿 run_sequence 的逻辑）
                output = self.tracker_impl.track(rgbd_image)
                
                t_end = time.time()
                
                # 提取结果
                bbox, confidence = self._extract_results(output)
                
                # 可视化
                self._visualize_results(vis, bbox, confidence, t_end - t_start, frame_id)
                
                cv2.imshow(win_name, vis)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r'):  # 重新初始化
                    print("\n[INFO] Re-initialization requested...")
                    self._select_target(vis, rgbd_image, win_name)
                elif key == ord('s'):  # 选择新目标
                    print("\n[INFO] New target selection requested...")
                    self._select_target(vis, rgbd_image, win_name)
                
                frame_id += 1
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] Tracking error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.sequence.cleanup()
            print("[INFO] Tracking finished.")
    
    def _select_target(self, vis, rgbd_image, win_name):
        """选择跟踪目标"""
        print("[INFO] Please select ROI...")
        roi = cv2.selectROI(win_name, vis, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi
        
        if w > 0 and h > 0:
            self.init_bbox = [float(x), float(y), float(w), float(h)]
            
            # 初始化跟踪器（模仿 run_sequence 的初始化逻辑）
            init_info = {'init_bbox': self.init_bbox}
            
            print(f"[INFO] Initializing tracker with bbox: {self.init_bbox}")
            
            try:
                out = self.tracker_impl.initialize(rgbd_image, init_info)
                self.initialized = True
                print(f"[SUCCESS] Tracker initialized successfully!")
                
                if self.debug:
                    print(f"[DEBUG] Init output: {type(out)}")
                    
            except Exception as e:
                print(f"[ERROR] Tracker initialization failed: {e}")
                import traceback
                traceback.print_exc()
                self.initialized = False
        else:
            print("[WARNING] Invalid ROI selection")
    
    def _extract_results(self, output):
        """提取跟踪结果"""
        bbox = None
        confidence = 0.0
        
        if isinstance(output, dict):
            # 提取bbox
            for key in ['target_bbox', 'bbox', 'pred_bbox']:
                if key in output:
                    bbox = output[key]
                    break
            
            # 提取confidence  
            for key in ['best_score', 'confidence', 'score']:
                if key in output:
                    confidence = output[key]
                    if hasattr(confidence, 'item'):  # tensor
                        confidence = confidence.item()
                    break
                    
        elif hasattr(output, 'target_bbox'):
            bbox = output.target_bbox
            if hasattr(output, 'best_score'):
                confidence = output.best_score
                
        # 转换bbox格式
        if bbox is not None:
            if hasattr(bbox, 'detach'):  # tensor
                bbox = bbox.detach().cpu().numpy()
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
        else:
            bbox = self.init_bbox if self.init_bbox else [0, 0, 50, 50]
            
        return bbox, confidence
    
    def _visualize_results(self, vis, bbox, confidence, elapsed_time, frame_id):
        """可视化跟踪结果"""
        x, y, w, h = bbox
        
        # 根据置信度选择颜色
        if confidence > 0.8:
            color = (0, 255, 0)  # 绿色 - 高置信度
        elif confidence > 0.6:
            color = (0, 255, 255)  # 黄色 - 中等置信度
        else:
            color = (0, 0, 255)  # 红色 - 低置信度
            
        # 绘制bbox
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(vis, p1, p2, color, 2)
        
        # 显示信息
        fps = 1.0 / max(elapsed_time, 1e-6)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Frame: {frame_id}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Conf: {confidence:.3f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis, f"Box: [{x:.1f},{y:.1f},{w:.1f},{h:.1f}]", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 低置信度警告
        if confidence < 0.5:
            cv2.putText(vis, "WARNING: Low Confidence", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(vis, "Press 'r' to re-init", (10, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def main():
    """主函数 - 模仿 test.py 的参数解析"""
    parser = argparse.ArgumentParser(description='Run SUTrack on RealSense D435i camera.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method (e.g., sutrack)')
    parser.add_argument('tracker_param', type=str, help='Name of config file (e.g., sutrack_b224)')
    parser.add_argument('--debug', type=int, default=0, help='Debug level (0=none, 1=basic, 2=verbose)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"SUTrack Real-time Demo")
    print(f"Tracker: {args.tracker_name}")
    print(f"Config: {args.tracker_param}")
    print(f"Debug: {args.debug}")
    print("="*80 + "\n")
    
    # 预检查：验证基本路径
    print("[INFO] Pre-checking SUTrack installation...")
    
    # 检查主要权重文件
    checkpoint_paths = [
        "/home/nick/code/code.sutrack/SUTrack/checkpoints/sutrack_t224.pth",
    ]
    
    found_checkpoints = []
    for path in checkpoint_paths:
        if os.path.exists(path):
            found_checkpoints.append(path)
            print(f"[INFO] ✅ Found checkpoint: {os.path.basename(path)}")
    
    if not found_checkpoints:
        print("[WARNING] ⚠️  No main checkpoints found! Tracker may not work properly.")
    
    # 检查预训练编码器
    pretrain_paths = [
        "/home/nick/code/code.sutrack/SUTrack/pretrained/itpn/fast_itpn_tiny_1600e_1k.pt",
        "/home/nick/code/code.sutrack/SUTrack/pretrained/fast_itpn_tiny_1600e_1k.pt",
        "/home/nick/code/code.sutrack/SUTrack/checkpoints/fast_itpn_tiny_1600e_1k.pt",
    ]
    
    found_pretrains = []
    for path in pretrain_paths:
        if os.path.exists(path):
            found_pretrains.append(path)
            print(f"[INFO] ✅ Found pretrained encoder: {os.path.basename(path)}")
    
    if not found_pretrains:
        print("[WARNING] ⚠️  No pretrained encoder found. Will use random initialization.")
        print("[INFO] 💡 This is OK - the main checkpoint contains trained encoder weights.")
    
    print("="*80 + "\n")
    
    try:
        # 创建实时跟踪器
        tracker = RealtimeTracker(
            tracker_name=args.tracker_name,
            tracker_param=args.tracker_param,
            debug=args.debug
        )
        
        # 运行交互式跟踪
        tracker.run_interactive()
        
    except Exception as e:
        print(f"[ERROR] Failed to start tracking: {e}")
        import traceback
        traceback.print_exc()
        
        # 提供解决方案提示
        print("\n" + "="*80)
        print("💡 TROUBLESHOOTING TIPS:")
        print("="*80)
        
        if "No such file or directory" in str(e) and "pretrained" in str(e):
            print("❌ Missing pretrained files detected!")
            print("🔧 SOLUTIONS:")
            print("   1. Download missing pretrained files from the official repo")
            print("   2. Or try using sutrack_b224 instead of sutrack_t224:")
            print("      python demo_realsense.py sutrack sutrack_b224 --debug 1")
            print("   3. Or modify config to skip encoder pretraining")
        elif "checkpoint" in str(e).lower():
            print("❌ Checkpoint loading error!")
            print("🔧 SOLUTIONS:")
            print("   1. Check if checkpoint files exist in checkpoints_backup/")
            print("   2. Verify checkpoint format and pytorch version compatibility")
            print("   3. Try re-downloading checkpoints")
        elif "CUDA" in str(e) or "GPU" in str(e):
            print("❌ GPU/CUDA error!")
            print("🔧 SOLUTIONS:")
            print("   1. Check if CUDA is properly installed: nvidia-smi")
            print("   2. Check pytorch CUDA version: python -c 'import torch; print(torch.cuda.is_available())'")
            print("   3. Try CPU mode by modifying code")
        else:
            print("❌ General error occurred!")
            print("🔧 GENERAL SOLUTIONS:")
            print("   1. Check SUTrack installation and dependencies")
            print("   2. Verify PYTHONPATH includes SUTrack directory")
            print("   3. Check if all required packages are installed")
        
        print("="*80 + "\n")
        
        return 1
    
    print("\n[INFO] Demo completed successfully.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
