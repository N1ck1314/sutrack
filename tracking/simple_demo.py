#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 SUTrack D435i 实时跟踪 - 纯视觉模态

专注于RGB-D视觉跟踪，完全绕过文本模态和CLIP

运行方式：
  1) conda activate sutrack
  2) 确保：export PYTHONPATH=/绝对路径/SUTrack:$PYTHONPATH  
  3) python simple_demo.py

特点：
  - 纯视觉模态，无文本依赖
  - 简化的配置和错误处理
  - 专为实时RGB-D跟踪优化
"""

import os
import sys
import time
import cv2
import numpy as np
import pyrealsense2 as rs

# 添加路径以导入 SUTrack 模块
env_path = os.path.join(os.path.dirname(__file__), '..')
if (env_path not in sys.path):
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker


class VisionOnlyRealSenseTracker:
    """纯视觉实时跟踪器"""
    
    def __init__(self, tracker_param="sutrack_t224"):
        self.tracker_param = tracker_param
        print(f"[INFO] Initializing Vision-Only SUTrack with config: {tracker_param}")
        
        # 初始化相机
        self._init_camera()
        
        # 创建跟踪器
        self._init_tracker()
        
        # 跟踪状态
        self.initialized = False
        self.frame_count = 0
    
    def _init_camera(self):
        """初始化D435i相机"""
        print("[INFO] Initializing RealSense camera...")
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # 配置流 - 使用较小分辨率以提高帧率
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = self.pipeline.start(config)
        
        # 深度对齐
        self.align = rs.align(rs.stream.color)
        
        # 深度比例
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        print(f"[INFO] Camera ready. Depth scale: {self.depth_scale:.6f}")
        
        # 稳定化
        for _ in range(30):
            try:
                self.pipeline.wait_for_frames(1000)
            except:
                pass
    
    def _init_tracker(self):
        """初始化纯视觉跟踪器"""
        print("[INFO] Creating vision-only tracker...")
        
        try:
            # 创建跟踪器
            tracker_wrapper = Tracker("sutrack", self.tracker_param, "demo", run_id=None)
            params = tracker_wrapper.get_parameters()
            
            # 🔧 修复预训练权重路径问题
            print("[INFO] Fixing pretrained paths...")
            self._fix_pretrained_paths(params)
            
            # 强制设置为纯视觉模式
            print("[INFO] Forcing vision-only configuration...")
            self._force_vision_only(params)
            
            # 创建跟踪器实例
            self.tracker = tracker_wrapper.create_tracker(params)
            
            print("[SUCCESS] ✅ Vision-only tracker ready!")
            
        except Exception as e:
            print(f"[ERROR] Failed to create tracker: {e}")
            raise
    
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
                    cfg.MODEL.ENCODER.PRETRAIN_TYPE = None
                    print(f"[INFO] ✅ Encoder will be initialized randomly")
            else:
                print(f"[INFO] ✅ Pretrained path exists: {pretrain_path}")
    
    def _force_vision_only(self, params):
        """强制设置为纯视觉模式"""
        cfg = params.cfg
        
        print("[INFO] Configuring pure vision mode...")
        
        # 完全移除文本编码器
        if hasattr(cfg.MODEL, 'TEXT_ENCODER'):
            print("[INFO]    - Removing text encoder...")
            delattr(cfg.MODEL, 'TEXT_ENCODER')
        
        # 🔧 强制启用RGBD支持
        print("[INFO]    - Configuring RGBD support...")
        self._force_rgbd_support(cfg)
        
        # 设置所有语言相关为False
        language_configs = [
            ('DATA', 'MULTI_MODAL_LANGUAGE'),
            ('TEST', 'MULTI_MODAL_LANGUAGE'), 
            ('DATA', 'USE_NLP'),
            ('TEST', 'USE_NLP')
        ]
        
        for section, key in language_configs:
            if hasattr(getattr(cfg, section), key):
                print(f"[INFO]    - Disabling {section}.{key}...")
                attr = getattr(getattr(cfg, section), key)
                if isinstance(attr, dict):
                    for k in attr:
                        attr[k] = False
                else:
                    setattr(getattr(cfg, section), key, False)
        
        # 确保视觉模态开启
        vision_configs = [
            ('DATA', 'MULTI_MODAL_VISION'),
            ('TEST', 'MULTI_MODAL_VISION')
        ]
        
        for section, key in vision_configs:
            if hasattr(getattr(cfg, section), key):
                print(f"[INFO]    - Enabling {section}.{key}...")
                attr = getattr(getattr(cfg, section), key)
                if isinstance(attr, dict):
                    for k in attr:
                        attr[k] = True
                else:
                    setattr(getattr(cfg, section), key, True)
        
        # 简化任务设置
        if hasattr(cfg.MODEL, 'TASK_NUM'):
            original_tasks = cfg.MODEL.TASK_NUM
            cfg.MODEL.TASK_NUM = min(3, original_tasks)  # 保留前3个任务（视觉相关）
            print(f"[INFO]    - Simplified tasks: {original_tasks} -> {cfg.MODEL.TASK_NUM}")
        
        print("[SUCCESS] ✅ Pure vision mode configured successfully")
    
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
            
            # 为深度通道添加归一化参数
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
    
    def get_frame(self):
        """获取预处理后的帧"""
        try:
            frames = self.pipeline.wait_for_frames(5000)
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            # 原始数据
            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            
            # 预处理为6通道RGBD
            rgbd = self._preprocess_rgbd(color_bgr, depth_raw)
            
            return rgbd, color_bgr
            
        except Exception as e:
            print(f"[WARNING] Frame capture failed: {e}")
            return None, None
    
    def _preprocess_rgbd(self, color_bgr, depth_raw, max_depth=5.0):
        """预处理为6通道输入"""
        # BGR -> RGB
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        
        # 深度处理
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        depth_m = np.clip(depth_m, 0.0, max_depth)
        depth_norm = (depth_m / max_depth * 255.0).astype(np.uint8)
        
        # 扩展深度为3通道
        depth_3ch = np.stack([depth_norm] * 3, axis=2)
        
        # 合并为6通道
        rgbd = np.concatenate([color_rgb, depth_3ch], axis=2)
        
        return rgbd
    
    def initialize_tracking(self, rgbd_image, bbox):
        """初始化跟踪"""
        try:
            init_info = {'init_bbox': bbox}
            self.tracker.initialize(rgbd_image, init_info)
            self.initialized = True
            self.frame_count = 0
            print(f"[SUCCESS] Tracking initialized: {bbox}")
            return True
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            return False
    
    def track_frame(self, rgbd_image):
        """跟踪单帧"""
        if not self.initialized:
            return None
        
        try:
            output = self.tracker.track(rgbd_image)
            self.frame_count += 1
            
            # 提取结果
            if isinstance(output, dict) and 'target_bbox' in output:
                bbox = output['target_bbox']
                confidence = output.get('best_score', 0.0)
            else:
                return None
            
            # 转换格式
            if hasattr(bbox, 'detach'):
                bbox = bbox.detach().cpu().numpy()
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            
            if hasattr(confidence, 'item'):
                confidence = confidence.item()
            
            return {
                'bbox': bbox,
                'confidence': float(confidence),
                'frame_id': self.frame_count
            }
            
        except Exception as e:
            print(f"[WARNING] Tracking failed: {e}")
            return None
    
    def run_interactive(self):
        """运行交互式跟踪"""
        print("\n" + "="*50)
        print("Vision-Only SUTrack Demo")
        print("Controls: 's' - select, ESC - exit")
        print("="*50)
        
        win_name = "Vision-Only SUTrack"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                frame_result = self.get_frame()
                if frame_result[0] is None:
                    continue
                
                rgbd_image, color_bgr = frame_result
                vis = color_bgr.copy()
                
                # 等待初始化
                if not self.initialized:
                    cv2.putText(vis, "Press 's' to select target", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(win_name, vis)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == ord('s'):
                        # 选择目标
                        roi = cv2.selectROI(win_name, vis, False, True)
                        if roi[2] > 0 and roi[3] > 0:
                            bbox = [float(x) for x in roi]
                            self.initialize_tracking(rgbd_image, bbox)
                    continue
                
                # 正常跟踪
                t0 = time.time()
                result = self.track_frame(rgbd_image)
                t1 = time.time()
                
                if result:
                    # 可视化
                    bbox = result['bbox']
                    conf = result['confidence']
                    x, y, w, h = bbox
                    
                    # 绘制框
                    color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.5 else (0, 0, 255)
                    cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                    
                    # 显示信息
                    fps = 1.0 / max(t1 - t0, 1e-6)
                    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis, f"Conf: {conf:.3f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(vis, f"Frame: {result['frame_id']}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(win_name, vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r'):  # 重新初始化
                    self.initialized = False
                    
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Cleaned up")


def main():
    """主函数"""
    print("Starting Vision-Only SUTrack Demo...")
    
    # 🔍 预检查：显示可用的配置和权重
    print("\n[INFO] Pre-checking available configurations...")
    
    # 检查预训练权重
    pretrain_files = {
        'sutrack_t224': 'pretrained/itpn/fast_itpn_tiny_1600e_1k.pt'
    }
    
    available_configs = []
    for config, pretrain_file in pretrain_files.items():
        # 检查多个可能的位置
        search_paths = [
            f"/home/nick/code/code.sutrack/SUTrack/{pretrain_file}",
            f"/home/nick/code/code.sutrack/SUTrack/pretrained/{os.path.basename(pretrain_file)}",
            f"/home/nick/code/code.sutrack/SUTrack/checkpoints/{os.path.basename(pretrain_file)}",
        ]
        
        found = False
        for path in search_paths:
            if os.path.exists(path):
                print(f"[INFO] ✅ {config}: Found pretrained file at {path}")
                available_configs.append(config)
                found = True
                break
        
        if not found:
            print(f"[INFO] ⚠️  {config}: Pretrained file not found, will use random initialization")
            available_configs.append(config)  # 仍然可以尝试，只是不用预训练权重
    
    # 按优先级排序配置
    configs_to_try = []
    if 'sutrack_t224' in available_configs:
        configs_to_try.append('sutrack_t224')  # 优先尝试tiny版本
    if 'sutrack_b224' in available_configs:
        configs_to_try.append('sutrack_b224')  # 然后尝试base版本
    
    if not configs_to_try:
        print("[ERROR] No valid configurations found!")
        return 1
    
    print(f"[INFO] Will try configurations in order: {configs_to_try}")
    print("="*60 + "\n")
    
    try:
        tracker = None
        for config in configs_to_try:
            try:
                print(f"\n[INFO] Attempting to create tracker with: {config}")
                tracker = VisionOnlyRealSenseTracker(config)
                print(f"[SUCCESS] ✅ Successfully created tracker with {config}")
                break
            except Exception as e:
                print(f"[WARNING] ⚠️  Failed with {config}: {str(e)[:100]}...")
                # 如果是权重文件问题，继续尝试下一个配置
                if "No such file or directory" in str(e) and "pretrained" in str(e):
                    print(f"[INFO] 🔄 Pretrained file issue with {config}, trying next config...")
                    continue
                else:
                    print(f"[ERROR] Unexpected error with {config}, stopping...")
                    break
        
        if tracker is None:
            print("\n" + "="*60)
            print("[ERROR] ❌ Failed to create tracker with any configuration!")
            print("="*60)
            print("💡 TROUBLESHOOTING:")
            print("1. Check if SUTrack is properly installed")
            print("2. Verify PYTHONPATH includes SUTrack directory")
            print("3. Check CUDA/GPU availability")
            print("4. Try downloading missing pretrained files")
            print("="*60)
            return 1
        
        # 运行跟踪
        print(f"\n🚀 Starting interactive tracking with {tracker.tracker_param}...")
        tracker.run_interactive()
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n[INFO] Demo completed successfully! 🎉")
    return 0


if __name__ == '__main__':
    sys.exit(main())
