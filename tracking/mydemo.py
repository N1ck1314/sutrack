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
      color_rgb: (H,W,3) float32 [0,1]  # RGB 3个通道
      depth_3ch: (H,W,3) float32 [0,1]  # 深度信息复制成3个通道 (为了与RGB维度匹配)
    
    SUTrack 为什么需要6通道？
    - 通道 0-2: RGB 彩色信息 (红、绿、蓝)
    - 通道 3-5: 深度信息 (复制3次，保持与RGB相同的维度结构)
    
    这样设计的原因：
    1. SUTrack 是RGB-D跟踪器，需要同时利用颜色和深度信息
    2. 深度信息提供物体的3D几何结构，有助于更准确的跟踪
    3. 将深度复制成3通道是为了保持与RGB相同的维度，便于网络处理
    """
    # 1) BGR -> RGB, [0,1]
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_rgb = color_rgb.astype(np.float32) / 255.0

    # 2) raw depth -> meters
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)  # (H,W) in meters
    depth_m = np.clip(depth_m, 0.0, max_dist_m)
    depth_norm = depth_m / max_dist_m  # [0,1]

    # 3) 将单通道深度扩展成3通道，与RGB维度匹配
    # 这样最终的RGBD图像就是6通道：RGB(3) + Depth(3) = 6
    depth_3ch = np.stack([depth_norm, depth_norm, depth_norm], axis=2)

    return color_rgb, depth_3ch


# ========= 3. SUTrack 接口封装（直接使用跟踪器实现） =========

class SUTrackOnlineTracker:
    """
    在线跟踪封装：
      - 直接使用 SUTrack 跟踪器实现
      - 绕过测试框架，直接调用核心跟踪器
    """

    def __init__(self, tracker_name="sutrack", tracker_param="sutrack_b224"):
        """
        直接创建跟踪器实现
        """
        print(f"[INFO] Creating tracker: {tracker_name} with param: {tracker_param}")
        
        try:
            # 直接导入跟踪器实现
            from lib.test.tracker.sutrack import SUTrack
            
            # 创建跟踪器参数
            params = self._get_tracker_params(tracker_param)
            
            # 创建跟踪器实例
            self.tracker = SUTrack(params)
            self.initialized = False
            self.last_bbox = None
            
            print("[INFO] SUTrack tracker created successfully.")
            
        except ImportError as e:
            print(f"[ERROR] Failed to import SUTrack: {e}")
            # 备选方案：尝试其他导入路径
            try:
                from lib.test.tracker import sutrack_tracker
                self.tracker = sutrack_tracker.get_tracker_class()(tracker_param)
                print("[INFO] SUTrack tracker created using alternative method.")
            except Exception as e2:
                print(f"[ERROR] All tracker creation methods failed: {e2}")
                raise
        except Exception as e:
            print(f"[ERROR] 创建 SUTrack tracker 失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _get_tracker_params(self, tracker_param):
        """获取跟踪器参数"""
        from lib.test.parameter.sutrack import parameters
        
        # 尝试获取参数
        if hasattr(parameters, tracker_param):
            return getattr(parameters, tracker_param)()
        else:
            # 使用默认参数
            print(f"[WARNING] Parameter {tracker_param} not found, using default")
            return parameters.sutrack_b224()

    def initialize(self, color_rgb, depth_3ch, init_bbox):
        """
        初始化跟踪器。

        color_rgb: (H,W,3) float32 [0,1]  # RGB 3通道
        depth_3ch: (H,W,3) float32 [0,1]  # 深度 3通道
        init_bbox: [x, y, w, h]，图像坐标
        
        最终输入给SUTrack的是6通道图像：
        - 前3通道：RGB彩色信息
        - 后3通道：深度信息（重复3次）
        """
        try:
            # 合并 RGB(3通道) 和 Depth(3通道) = 总共6通道
            H, W = color_rgb.shape[:2]
            rgbd_image = np.concatenate([color_rgb, depth_3ch], axis=2)  # (H,W,6)
            
            print(f"[DEBUG] RGBD image shape: {rgbd_image.shape}")  # 应该是 (H,W,6)
            print(f"[DEBUG] Channels breakdown: RGB(0-2) + Depth(3-5) = 6 total")
            
            # 转换为张量格式 (C, H, W)
            rgbd_tensor = torch.from_numpy(rgbd_image).permute(2, 0, 1).float()
            
            # 准备初始化信息
            init_info = {
                'init_bbox': init_bbox,  # [x, y, w, h]
            }
            
            # 调用跟踪器初始化
            if hasattr(self.tracker, 'initialize'):
                self.tracker.initialize(rgbd_tensor, init_info)
            elif hasattr(self.tracker, 'init'):
                self.tracker.init(rgbd_tensor, init_bbox)
            else:
                print(f"[ERROR] Tracker has no initialize/init method")
                print(f"[DEBUG] Available methods: {[m for m in dir(self.tracker) if not m.startswith('_')]}")
                return
            
            self.initialized = True
            self.last_bbox = init_bbox
            
            print(f"[INFO] Tracker initialized with bbox={init_bbox}")
            
        except Exception as e:
            print(f"[ERROR] Tracker initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.initialized = False

    def track(self, color_rgb, depth_3ch, frame_id):
        """
        进行单帧跟踪，返回 bbox = [x, y, w, h]。
        
        同样将RGB(3通道) + Depth(3通道) = 6通道输入给网络
        """
        if not self.initialized:
            print("[WARNING] Tracker not initialized. Returning default bbox.")
            return [0, 0, 50, 50]

        try:
            # 合并为6通道输入
            rgbd_image = np.concatenate([color_rgb, depth_3ch], axis=2)  # (H,W,6)
            
            # 转换为张量格式 (C, H, W)
            rgbd_tensor = torch.from_numpy(rgbd_image).permute(2, 0, 1).float()
            
            # 调用跟踪方法
            if hasattr(self.tracker, 'track'):
                output = self.tracker.track(rgbd_tensor)
            else:
                print("[ERROR] No track method found")
                print(f"[DEBUG] Available methods: {[m for m in dir(self.tracker) if not m.startswith('_')]}")
                return self.last_bbox if self.last_bbox is not None else [0, 0, 50, 50]
            
            # 从输出中提取 bbox
            bbox = self._extract_bbox_from_output(output)
            
            if bbox is not None and len(bbox) == 4:
                self.last_bbox = bbox
            else:
                print(f"[WARNING] Invalid bbox: {bbox}, using last bbox")
                bbox = self.last_bbox if self.last_bbox is not None else [0, 0, 50, 50]
            
            return bbox
            
        except Exception as e:
            print(f"[WARNING] Tracking failed: {e}")
            import traceback
            traceback.print_exc()
            # 兜底方案：返回上一帧的 bbox
            return self.last_bbox if self.last_bbox is not None else [0, 0, 50, 50]

    def _extract_bbox_from_output(self, output):
        """从跟踪器输出中提取bbox"""
        if isinstance(output, dict):
            # 尝试常见的字段名
            for key in ['target_bbox', 'bbox', 'pred_bbox', 'box']:
                if key in output:
                    bbox = output[key]
                    break
            else:
                print(f"[DEBUG] Output dict keys: {list(output.keys())}")
                return self.last_bbox
        elif hasattr(output, 'target_bbox'):
            bbox = output.target_bbox
        elif hasattr(output, 'bbox'):
            bbox = output.bbox
        elif isinstance(output, (list, tuple, np.ndarray)) and len(output) == 4:
            bbox = output
        else:
            print(f"[DEBUG] Unexpected output type: {type(output)}")
            return self.last_bbox
        
        # 确保bbox是正确的格式
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().cpu().numpy()
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        elif not isinstance(bbox, list):
            bbox = list(bbox) if bbox is not None else self.last_bbox
        
        return bbox


# 备选实现：如果上述方法失败，使用简化版本
class SimpleSUTracker:
    """简化版跟踪器实现"""
    
    def __init__(self):
        print("[INFO] Using simplified tracker (fallback)")
        self.initialized = False
        self.last_bbox = None
        
    def initialize(self, color_rgb, depth_3ch, init_bbox):
        self.last_bbox = init_bbox
        self.initialized = True
        print(f"[INFO] Simple tracker initialized with bbox={init_bbox}")
        
    def track(self, color_rgb, depth_3ch, frame_id):
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
    # 尝试创建跟踪器
    tracker = None
    for tracker_param in ["sutrack_b224", "sutrack"]:
        try:
            tracker = SUTrackOnlineTracker(
                tracker_name="sutrack", 
                tracker_param=tracker_param
            )
            break
        except Exception as e:
            print(f"[WARNING] Failed to create tracker with {tracker_param}: {e}")
            continue
    
    # 如果所有方法都失败，使用简化版本
    if tracker is None:
        print("[WARNING] All tracker creation methods failed, using simplified tracker")
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
                    roi = cv2.selectROI(win_name, vis, fromCenter=False, showCrosshair=True)
                    x, y, w, h = roi
                    if (w > 0 and h > 0):
                        init_bbox = [float(x), float(y), float(w), float(h)]
                        # 做一次预处理并初始化 tracker
                        color_rgb, depth_3ch = preprocess_rgb_depth(color_bgr, depth_raw, depth_scale)
                        tracker.initialize(color_rgb, depth_3ch, init_bbox)
                        print(f"[INFO] Tracking initialized with bbox: {init_bbox}")
                    else:
                        init_bbox = None
                        print("[WARNING] Invalid ROI selection, please try again")
                continue

            # 之后：正常跟踪
            t0 = time.time()
            color_rgb, depth_3ch = preprocess_rgb_depth(color_bgr, depth_raw, depth_scale)
            bbox = tracker.track(color_rgb, depth_3ch, frame_id)
            t1 = time.time()

            x, y, w, h = bbox
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)

            fps = 1.0 / max(1e-6, (t1 - t0))
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示帧号
            cv2.putText(vis, f"Frame: {frame_id}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(win_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

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

