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
      color_rgb_uint8: (H,W,3) uint8 [0,255]  # RGB 3个通道，供SUTRACK使用
      depth_3ch_uint8: (H,W,3) uint8 [0,255]  # 深度信息复制成3个通道
      color_rgb_float: (H,W,3) float32 [0,1]  # RGB float版本，供遮挡检测使用
      depth_3ch_float: (H,W,3) float32 [0,1]  # 深度float版本，供遮挡检测使用
    
    SUTrack 为什么需要6通道？
    - 通道 0-2: RGB 彩色信息 (红、绿、蓝)
    - 通道 3-5: 深度信息 (复制3次，保持与RGB相同的维度结构)
    
    这样设计的原因：
    1. SUTrack 是RGB-D跟踪器，需要同时利用颜色和深度信息
    2. 深度信息提供物体的3D几何结构，有助于更准确的跟踪
    3. 将深度复制成3通道是为了保持与RGB相同的维度，便于网络处理
    """
    # 1) BGR -> RGB, uint8 [0,255] for SUTRACK
    color_rgb_uint8 = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # 2) raw depth -> meters -> uint8 [0,255]
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)  # (H,W) in meters
    depth_m = np.clip(depth_m, 0.0, max_dist_m)
    depth_norm = depth_m / max_dist_m  # [0,1]
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)  # [0,255]

    # 3) 将单通道深度扩展成3通道 uint8版本
    depth_3ch_uint8 = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=2)
    
    # 4) 同时准备float版本供遮挡检测使用
    color_rgb_float = color_rgb_uint8.astype(np.float32) / 255.0
    depth_3ch_float = np.stack([depth_norm, depth_norm, depth_norm], axis=2)

    return color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float


# ========= 2.1 遮挡检测与重检测辅助 =========
def compute_depth_stats(depth_3ch, bbox):
    x, y, w, h = map(int, bbox)
    H, W = depth_3ch.shape[:2]
    x = max(0, x); y = max(0, y)
    x2 = min(W, x + w); y2 = min(H, y + h)
    if x2 <= x or y2 <= y:
        return None, None
    d = depth_3ch[y:y2, x:x2, 0]
    d_flat = d.reshape(-1)
    d_valid = d_flat[np.isfinite(d_flat)]
    if d_valid.size == 0:
        return None, None
    return float(d_valid.mean()), float(d_valid.std())

def re_detect_person(color_bgr, depth_3ch, prev_mean, max_dist_diff=0.1):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(color_bgr, winStride=(8, 8), padding=(8, 8), scale=1.05)
    best = None
    best_score = 1e9
    for (x, y, w, h) in rects:
        mean, _ = compute_depth_stats(depth_3ch, [x, y, w, h])
        if mean is None:
            continue
        diff = abs(mean - float(prev_mean)) if prev_mean is not None else 0.0
        if diff <= max_dist_diff and diff < best_score:
            best = [float(x), float(y), float(w), float(h)]
            best_score = diff
    return best

def compute_adaptive_search_factor(depth_mean, min_factor=2.5, max_factor=5.5, near_depth=0.5, far_depth=4.0):
    """
    深度引导的搜索区域自适应策略
    
    参数：
        depth_mean: 当前目标的平均深度（归一化，0-1范围）
        min_factor: 最小搜索因子（目标很近时）
        max_factor: 最大搜索因子（目标很远时）
        near_depth: 近距离阈值（归一化深度）
        far_depth: 远距离阈值（归一化深度）
    
    返回：
        search_factor: 搜索区域相对于bbox的倍数
    
    核心思想：
    - 目标近 → 图像占比大，运动幅度小 → 小搜索区（省算力）
    - 目标远 → 图像占比小，UAV运动导致位移大 → 大搜索区（防丢）
    """
    if depth_mean is None or not np.isfinite(depth_mean):
        return (min_factor + max_factor) / 2.0  # 默认中等值
    
    # 将深度从归一化空间映射回米（假设max_dist_m=5.0）
    depth_m = depth_mean * 5.0
    
    # 线性插值：S(d) = a*d + b
    # near_depth_m 对应 min_factor, far_depth_m 对应 max_factor
    a = (max_factor - min_factor) / (far_depth - near_depth)
    b = min_factor - a * near_depth
    
    search_factor = a * depth_m + b
    search_factor = np.clip(search_factor, min_factor, max_factor)
    
    return float(search_factor)

# ========= 3. SUTrack 接口封装（直接使用跟踪器实现） =========

class SUTrackOnlineTracker:
    """
    在线跟踪封装：
      - 直接使用 SUTrack 跟踪器实现
      - 绕过测试框架，直接调用核心跟踪器
      - 支持深度自适应搜索区域
    """

    def __init__(self, tracker_name="sutrack", tracker_param="sutrack_b224", enable_adaptive_search=True):
        """
        直接创建跟踪器实现
        """
        print(f"[INFO] Creating tracker: {tracker_name} with param: {tracker_param}")
        
        try:
            # 直接导入跟踪器实现
            from lib.test.tracker.sutrack import SUTRACK
            
            # 创建跟踪器参数
            params = self._get_tracker_params(tracker_param)
            
            # 创建跟踪器实例（注意这里dataset_name参数）
            self.tracker = SUTRACK(params, dataset_name='demo')
            self.initialized = False
            self.last_bbox = None
            self.enable_adaptive_search = enable_adaptive_search
            self.base_search_factor = params.search_factor  # 保存原始搜索因子
            self.base_template_factor = params.template_factor  # 保存原始模板因子
            
            print(f"[INFO] SUTrack tracker created successfully. Adaptive search: {enable_adaptive_search}")
            
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

    def initialize(self, color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, init_bbox):
        """
        初始化跟踪器。

        color_rgb_uint8: (H,W,3) uint8 [0,255]  # RGB 3通道
        depth_3ch_uint8: (H,W,3) uint8 [0,255]  # 深度 3通道
        color_rgb_float: (H,W,3) float32 [0,1]  # RGB float版本
        depth_3ch_float: (H,W,3) float32 [0,1]  # 深度 float版本
        init_bbox: [x, y, w, h]，图像坐标
        
        最终输入给SUTrack的是6通道uint8图像：
        - 前3通道：RGB彩色信息
        - 后3通道：深度信息（重复3次）
        """
        try:
            # 合并 RGB(3通道) 和 Depth(3通道) = 总共6通道 uint8 [0,255]
            H, W = color_rgb_uint8.shape[:2]
            rgbd_image = np.concatenate([color_rgb_uint8, depth_3ch_uint8], axis=2)  # (H,W,6) uint8
            
            print(f"[DEBUG] RGBD image shape: {rgbd_image.shape}, dtype: {rgbd_image.dtype}")  # 应该是 (H,W,6) uint8
            print(f"[DEBUG] Channels breakdown: RGB(0-2) + Depth(3-5) = 6 total")
            
            # SUTRACK期期输入是numpy数组(H,W,C) uint8格式
            # 准备初始化信息
            init_info = {
                'init_bbox': init_bbox,  # [x, y, w, h]
            }
            
            # 调用跟踪器初始化
            if hasattr(self.tracker, 'initialize'):
                self.tracker.initialize(rgbd_image, init_info)
            elif hasattr(self.tracker, 'init'):
                self.tracker.init(rgbd_image, init_bbox)
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

    def track(self, color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, frame_id, current_depth_mean=None):
        """
        进行单帧跟踪，返回 bbox = [x, y, w, h]。
        
        同样将RGB(3通道) + Depth(3通道) = 6通道uint8输入给网络
        支持深度自适应搜索区域调整
        """
        if not self.initialized:
            print("[WARNING] Tracker not initialized. Returning default bbox.")
            return [0, 0, 50, 50]

        try:
            # 深度自适应搜索区域调整
            if self.enable_adaptive_search and current_depth_mean is not None:
                adaptive_factor = compute_adaptive_search_factor(current_depth_mean)
                # 动态调整跟踪器参数
                if hasattr(self.tracker, 'params'):
                    self.tracker.params.search_factor = adaptive_factor
                    # 可选：同时调整模板因子（保持比例）
                    # self.tracker.params.template_factor = adaptive_factor / 2.0
            
            # 合并为6通道输入 (H,W,6) uint8 [0,255]
            rgbd_image = np.concatenate([color_rgb_uint8, depth_3ch_uint8], axis=2)
            
            # SUTRACK期望输入是numpy数组(H,W,C) uint8格式
            # 调用跟踪方法
            if hasattr(self.tracker, 'track'):
                output = self.tracker.track(rgbd_image)
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
            
            # 恢复原始搜索因子（避免影响下一帧初始状态）
            if self.enable_adaptive_search and hasattr(self.tracker, 'params'):
                self.tracker.params.search_factor = self.base_search_factor
            
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
        
    def initialize(self, color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, init_bbox):
        self.last_bbox = init_bbox
        self.initialized = True
        print(f"[INFO] Simple tracker initialized with bbox={init_bbox}")
        
    def track(self, color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, frame_id, current_depth_mean=None):
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
                tracker_param=tracker_param,
                enable_adaptive_search=True  # 启用深度自适应搜索
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
    prev_depth_mean = None
    prev_depth_std = None
    occluded = False
    current_search_factor = 4.0  # 记录当前搜索因子

    win_name = "SUTrack RGB-D Adaptive Search (D435i)"
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
                        color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float = preprocess_rgb_depth(color_bgr, depth_raw, depth_scale)
                        tracker.initialize(color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, init_bbox)
                        mean, std = compute_depth_stats(depth_3ch_float, init_bbox)
                        prev_depth_mean, prev_depth_std = mean, std
                        occluded = False
                        print(f"[INFO] Tracking initialized with bbox: {init_bbox}")
                        if mean is not None:
                            print(f"[INFO] Initial depth: {mean*5.0:.2f}m")
                    else:
                        init_bbox = None
                        print("[WARNING] Invalid ROI selection, please try again")
                continue

            # 之后：正常跟踪
            t0 = time.time()
            color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float = preprocess_rgb_depth(color_bgr, depth_raw, depth_scale)
            
            # 计算当前深度用于自适应搜索
            cur_depth_mean_for_search, _ = compute_depth_stats(depth_3ch_float, tracker.last_bbox if tracker.last_bbox else [0,0,100,100])
            current_search_factor = compute_adaptive_search_factor(cur_depth_mean_for_search) if cur_depth_mean_for_search else 4.0
            
            bbox = tracker.track(color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, frame_id, current_depth_mean=cur_depth_mean_for_search)
            t1 = time.time()

            # 深度遮挡检测
            cur_mean, cur_std = compute_depth_stats(depth_3ch_float, bbox)
            near_obstacle = (prev_depth_mean is not None and cur_mean is not None and (cur_mean < prev_depth_mean - 0.10))
            var_spike = (prev_depth_std is not None and cur_std is not None and (cur_std > prev_depth_std * 2.0 + 0.05))
            if near_obstacle or var_spike:
                occluded = True
                # 重检测：HOG 人体检测 + 深度一致性约束
                new_bbox = re_detect_person(color_bgr, depth_3ch_float, prev_depth_mean, max_dist_diff=0.10)
                if new_bbox is not None:
                    tracker.initialize(color_rgb_uint8, depth_3ch_uint8, color_rgb_float, depth_3ch_float, new_bbox)
                    bbox = new_bbox
                    cur_mean, cur_std = compute_depth_stats(depth_3ch_float, bbox)
                    prev_depth_mean, prev_depth_std = cur_mean, cur_std
                    occluded = False
            else:
                occluded = False
                if cur_mean is not None and cur_std is not None:
                    prev_depth_mean, prev_depth_std = cur_mean, cur_std
            
            x, y, w, h = bbox
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(vis, p1, p2, (0, 255, 0) if not occluded else (0, 0, 255), 2)

            fps = 1.0 / max(1e-6, (t1 - t0))
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示帧号
            cv2.putText(vis, f"Frame: {frame_id}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 显示深度与自适应搜索因子
            if cur_mean is not None:
                depth_m = cur_mean * 5.0
                cv2.putText(vis, f"Depth: {depth_m:.2f}m | Search: {current_search_factor:.1f}x", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if occluded:
                cv2.putText(vis, "Occluded: re-detecting...", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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

