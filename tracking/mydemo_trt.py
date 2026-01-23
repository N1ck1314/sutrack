#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 D435i + TensorRT 做 SUTrack 在线 RGB-Depth 跟踪 Demo

运行方式（Jetson Orin NX）：
  1) 确保已经用 trtexec 生成 sutrack_tiny_rgbd_fp16.engine
  2) conda activate py38_sutrack_t224_onnx  # 或你的环境
  3) python mydemo_trt.py --engine /path/to/sutrack_tiny_rgbd_fp16.engine

操作说明：
  - 按 's' 键：选取初始目标（用鼠标框选）
  - 按 ESC：退出
"""

import time
import cv2
import numpy as np
# 补丁：解决 NumPy 1.24+ 移除 np.bool 导致 TensorRT 报错的问题
if not hasattr(np, "bool_"):
    np.bool_ = bool
if not hasattr(np, "bool"):
    np.bool = np.bool_
import pyrealsense2 as rs
import argparse
import math

# TensorRT Python API
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # 自动初始化 CUDA context
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError as e:
    print(f"[ERROR] 无法导入 TensorRT/PyCUDA: {e}")
    print("请先安装：pip install tensorrt pycuda")
    exit(1)


# ========= 1. RealSense 相机部分（和原版一样）=========

def create_realsense_pipeline():
    """创建并启动 RealSense pipeline，并对齐深度到彩色坐标系。"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
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
            color_image = np.asanyarray(color_frame.get_data())  # BGR uint8
            depth_image = np.asanyarray(depth_frame.get_data())  # uint16
            return color_image, depth_image
        except RuntimeError as e:
            if retry < max_retries - 1:
                print(f"[WARNING] Frame timeout, retry {retry + 1}/{max_retries}: {e}")
                time.sleep(0.1)
            else:
                print(f"[ERROR] Failed to get frames after {max_retries} retries: {e}")
                raise
    return None, None


def preprocess_rgb_depth(color_bgr, depth_raw, depth_scale, max_dist_m=5.0):
    """
    预处理：BGR -> RGB + Depth -> 3ch uint8
    返回：color_rgb_uint8 (H,W,3), depth_3ch_uint8 (H,W,3)
    """
    color_rgb_uint8 = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    depth_m = np.clip(depth_m, 0.0, max_dist_m)
    depth_norm = depth_m / max_dist_m
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
    depth_3ch_uint8 = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=2)
    return color_rgb_uint8, depth_3ch_uint8


# ========= 2. TensorRT 推理引擎封装 =========

class TRTInferenceEngine:
    """
    TensorRT 推理引擎封装，支持：
      - 加载 .engine 文件
      - 输入：template(1,6,112,112), search(1,6,224,224), template_anno(1,4)
      - 输出：pred_boxes(1,1,4), score_map(1,1,fx,fx)
    """

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
        self.bindings = []
        self.stream = cuda.Stream()

        # 加载引擎
        self._load_engine()
        # 分配内存
        self._allocate_buffers()

    def _load_engine(self):
        """加载 TensorRT 引擎"""
        print(f"[TRT] Loading engine from: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            self.runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        self.context = self.engine.create_execution_context()
        print(f"[TRT] Engine loaded successfully. Num bindings: {self.engine.num_bindings}")

    def _allocate_buffers(self):
        """为输入/输出分配 GPU 内存，并建立名称映射"""
        self.inputs = {}
        self.outputs = {}
        self.bindings = [None] * self.engine.num_bindings

        for i in range(self.engine.num_bindings):
            # 兼容新旧 TensorRT API
            if hasattr(self.engine, 'get_tensor_name'):
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = self.engine.get_tensor_shape(name)
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            else:
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)
            
            size = trt.volume(shape)
            
            # 分配 GPU 内存
            device_mem = cuda.mem_alloc(size * dtype().itemsize)
            self.bindings[i] = int(device_mem)

            binding_info = {
                'index': i,
                'name': name,
                'dtype': dtype,
                'shape': shape,
                'device_mem': device_mem,
                'size': size
            }

            if is_input:
                self.inputs[name] = binding_info
                print(f"[TRT]   Input  {i}: {name}, shape={shape}, dtype={dtype}")
            else:
                self.outputs[name] = binding_info
                print(f"[TRT]   Output {i}: {name}, shape={shape}, dtype={dtype}")

    def infer(self, template: np.ndarray, search: np.ndarray, template_anno: np.ndarray):
        """
        根据名称匹配推理
        """
        # 1. 拷贝输入
        input_map = {
            "template": template,
            "search": search,
            "template_anno": template_anno
        }
        for name, data in input_map.items():
            if name in self.inputs:
                inp = self.inputs[name]
                # 确保内存连续并强制转换为正确类型
                host_data = np.ascontiguousarray(data.astype(inp['dtype']))
                cuda.memcpy_htod_async(inp['device_mem'], host_data, self.stream)

        # 2. 执行
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 3. 拷贝输出
        results = {}
        for name, out in self.outputs.items():
            host_mem = np.empty(out['size'], dtype=out['dtype'])
            cuda.memcpy_dtoh_async(host_mem, out['device_mem'], self.stream)
            results[name] = host_mem.reshape(out['shape'])

        self.stream.synchronize()

        # 按名称返回，确保顺序无关
        return results.get("pred_boxes"), results.get("score_map")


# ========= 3. SUTrack TensorRT 跟踪器封装 =========

class SUTrackTRTTracker:
    """
    使用 TensorRT 引擎的 SUTrack 跟踪器
    """

    def __init__(self, engine_path: str, template_size=112, search_size=224):
        self.engine = TRTInferenceEngine(engine_path)
        self.template_size = template_size
        self.search_size = search_size
        self.initialized = False
        self.state = None  # [x, y, w, h]
        self.template_patch = None
        self.template_anno = None

        # 归一化参数
        self.mean = np.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 6, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 6, 1, 1)

        # Hann 窗：针对 14x14 响应图
        self.output_sz = self.search_size // 16
        self.hann_window = self._create_hann2d(self.output_sz)

        # 平滑处理
        self.alpha = 0.6  # 框平滑系数
        self.last_bbox_raw = None

        self.fps_history = []

    def initialize(self, rgbd_image: np.ndarray, init_bbox: list):
        """
        初始化跟踪器
        Args:
            rgbd_image: (H, W, 6) uint8
            init_bbox: [x, y, w, h] 绝对坐标
        """
        self.state = init_bbox
        
        # 1. 采样模板 Patch (使用 factor=2.0，与训练一致)
        template_patch, resize_factor = self._sample_target(
            rgbd_image, self.state, factor=2.0, output_sz=self.template_size
        )
        
        # 2. 归一化并存储
        self.template_patch = self._normalize_patch(template_patch)
        
        # 3. 准备 template_anno (B, 4) -> (x, y, w, h) 相对坐标 [0,1]
        # 根据 lib/test/tracker/utils.py 中的 transform_image_to_crop 逻辑：
        # 目标中心在 Patch 中心 (output_sz-1)/2，宽高比例由 resize_factor 决定
        w, h = self.state[2], self.state[3]
        w_patch = w * resize_factor
        h_patch = h * resize_factor
        
        # 使用模板尺寸进行归一化 (减1是为了对齐像素索引，参考 transform_image_to_crop)
        norm_val = self.template_size - 1
        w_rel = w_patch / norm_val
        h_rel = h_patch / norm_val
        x_rel = 0.5 - 0.5 * w_rel
        y_rel = 0.5 - 0.5 * h_rel
        
        self.template_anno = np.array([[x_rel, y_rel, w_rel, h_rel]], dtype=np.float32)
        
        self.initialized = True
        self.last_bbox_raw = None
        print(f"[Tracker] Initialized with bbox: {init_bbox}")
        print(f"[Tracker] Template anno (rel): {self.template_anno}")

    def _create_hann2d(self, sz):
        # 产生更强的中心约束
        hann_1d = np.hanning(sz + 2)[1:-1]
        window = np.outer(hann_1d, hann_1d)
        return window.astype(np.float32)

    def track(self, rgbd_image: np.ndarray):
        if not self.initialized:
            return [0, 0, 50, 50], 0.0

        H, W, _ = rgbd_image.shape

        # 1. 裁剪并正确 Resize (RGBD 6通道)
        search_patch, resize_factor = self._sample_target(
            rgbd_image, self.state, factor=4.0, output_sz=self.search_size
        )

        # 2. 归一化
        search_norm = self._normalize_patch(search_patch)

        # 3. 推理
        t0 = time.perf_counter()
        pred_boxes, score_map = self.engine.infer(self.template_patch, search_norm, self.template_anno)
        t1 = time.perf_counter()
        self.fps_history.append(1.0 / max(t1 - t0, 1e-6))

        # 4. 对 score_map 应用 Sigmoid（如果模型未内置）
        # score_map 原始输出是 logit，需要转换为概率
        def sigmoid(x):
            return np.clip(1.0 / (1.0 + np.exp(-x)), 1e-4, 1 - 1e-4)
        
        score_map_prob = sigmoid(score_map)
        
        # 5. Hann 窗惩罚
        response = score_map_prob[0, 0]
        if response.shape == self.hann_window.shape:
            # 这里的置信度反映了目标在中心出现的概率
            response = response * self.hann_window
        
        confidence = float(np.max(response))
        
        # 6. 坐标还原
        pred_box = pred_boxes[0, 0, :]  # [cx, cy, w, h]
        bbox_raw = self._map_box_back(pred_box, resize_factor)
        
        # 7. 平滑与更新逻辑
        # Tiny 模型置信度通常较低，降低阈值到 0.05
        if confidence > 0.05:
            if self.last_bbox_raw is None:
                self.last_bbox_raw = bbox_raw
            else:
                # 指数平滑减少晃动
                bbox_raw = [
                    self.alpha * bbox_raw[i] + (1 - self.alpha) * self.last_bbox_raw[i]
                    for i in range(4)
                ]
                self.last_bbox_raw = bbox_raw
            
            # 边界裁剪
            x, y, w, h = bbox_raw
            x = max(0, min(W - 10, x))
            y = max(0, min(H - 10, y))
            w = max(10, min(W, w))
            h = max(10, min(H, h))
            self.state = [x, y, w, h]
        else:
            # 置信度太低，保持上一帧位置不更新搜索区域
            bbox_raw = self.state

        return self.state, confidence

    def _sample_target(self, image: np.ndarray, target_bb: list, factor: float, output_sz: int):
        """支持 6 通道的正确裁剪与 Resize"""
        x, y, w, h = target_bb
        crop_sz = math.ceil(math.sqrt(w * h) * factor)
        if crop_sz < 1: crop_sz = 1

        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        
        # 使用 OpenCV 的常规裁剪（会自动处理边界吗？不，我们手动 handle）
        # 这里为了简单直接用切片，外加 padding
        def get_patch(img, x1, y1, sz):
            h, w = img.shape[:2]
            x2, y2 = x1 + sz, y1 + sz
            
            # 这里的 padding 逻辑需要处理 6 通道
            pad_x1 = max(0, -x1)
            pad_y1 = max(0, -y1)
            pad_x2 = max(0, x2 - w)
            pad_y2 = max(0, y2 - h)
            
            roi_x1, roi_y1 = max(0, x1), max(0, y1)
            roi_x2, roi_y2 = min(w, x2), min(h, y2)
            
            patch = img[roi_y1:roi_y2, roi_x1:roi_x2]
            if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
                patch = cv2.copyMakeBorder(patch, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_CONSTANT, value=0)
            return patch

        im_crop = get_patch(image, x1, y1, crop_sz)
        
        # 关键修复：分开 Resize
        rgb_part = cv2.resize(im_crop[:, :, :3], (output_sz, output_sz))
        depth_part = cv2.resize(im_crop[:, :, 3:], (output_sz, output_sz))
        im_resized = np.concatenate([rgb_part, depth_part], axis=2)

        return im_resized, output_sz / crop_sz

    def _normalize_patch(self, patch: np.ndarray):
        """
        归一化 patch：uint8 [0,255] -> float32 [-mean/std]
        Args:
            patch: (H, W, 6) uint8
        Returns:
            normalized: (1, 6, H, W) float32
        """
        # HWC -> CHW
        patch_chw = patch.transpose(2, 0, 1).astype(np.float32)  # (6, H, W)
        patch_chw = patch_chw[np.newaxis, ...]  # (1, 6, H, W)

        # /255 -> normalize
        patch_norm = (patch_chw / 255.0 - self.mean) / self.std

        return patch_norm.astype(np.float32)

    def _map_box_back(self, pred_box: np.ndarray, resize_factor: float):
        """
        将预测 bbox 从 search patch 坐标映射回原图坐标
        Args:
            pred_box: (4,) [cx, cy, w, h] 归一化坐标 [0,1]
            resize_factor: float
        Returns:
            bbox: [x, y, w, h] 绝对坐标
        """
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx_rel, cy_rel, w_rel, h_rel = pred_box

        # 归一化坐标 -> search patch 像素坐标
        cx_patch = cx_rel * self.search_size
        cy_patch = cy_rel * self.search_size
        w_patch = w_rel * self.search_size
        h_patch = h_rel * self.search_size

        # patch 像素坐标 -> 原图像素坐标
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx_patch / resize_factor + (cx_prev - half_side)
        cy_real = cy_patch / resize_factor + (cy_prev - half_side)
        w_real = w_patch / resize_factor
        h_real = h_patch / resize_factor

        x_real = cx_real - 0.5 * w_real
        y_real = cy_real - 0.5 * h_real

        return [x_real, y_real, w_real, h_real]

    def get_avg_fps(self):
        """返回平均 FPS"""
        if len(self.fps_history) == 0:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)


# ========= 4. 主循环 =========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True,
                        help='Path to TensorRT engine file (sutrack_tiny_rgbd_fp16.engine)')
    parser.add_argument('--template_size', type=int, default=112)
    parser.add_argument('--search_size', type=int, default=224)
    args = parser.parse_args()

    print("[INFO] Initializing RealSense camera...")
    pipeline, align, depth_scale = create_realsense_pipeline()

    print("[INFO] Waiting for camera to stabilize...")
    for _ in range(30):
        try:
            pipeline.wait_for_frames(timeout_ms=1000)
        except:
            pass

    print(f"[INFO] Loading TensorRT tracker from: {args.engine}")
    tracker = SUTrackTRTTracker(args.engine, args.template_size, args.search_size)

    frame_id = 0
    init_bbox = None

    win_name = "SUTrack TRT RGB-D Tracking (D435i)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print("[INFO] Starting tracking loop...")
    print("[INFO] Press 's' to select ROI, ESC to quit")

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

            # 未初始化：等待用户按 's'
            if init_bbox is None:
                cv2.putText(vis, "Press 's' to select ROI, ESC to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                cv2.imshow(win_name, vis)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break

                if key == ord('s'):
                    print("\n[INFO] Please select ROI...")
                    roi = cv2.selectROI(win_name, vis, fromCenter=False, showCrosshair=True)
                    x, y, w, h = roi
                    print(f"[INFO] ROI selected: x={x}, y={y}, w={w}, h={h}")

                    if w > 0 and h > 0:
                        init_bbox = [float(x), float(y), float(w), float(h)]
                        # 预处理
                        color_rgb_uint8, depth_3ch_uint8 = preprocess_rgb_depth(color_bgr, depth_raw, depth_scale)
                        rgbd_image = np.concatenate([color_rgb_uint8, depth_3ch_uint8], axis=2)  # (H,W,6) uint8
                        # 初始化
                        tracker.initialize(rgbd_image, init_bbox)
                        print(f"\n[SUCCESS] Tracking started! Init bbox={init_bbox}\n")
                    else:
                        print("[WARNING] Invalid ROI, please try again")
                continue

            # 正常跟踪
            t_frame_start = time.perf_counter()

            # 预处理
            color_rgb_uint8, depth_3ch_uint8 = preprocess_rgb_depth(color_bgr, depth_raw, depth_scale)
            rgbd_image = np.concatenate([color_rgb_uint8, depth_3ch_uint8], axis=2)

            # 跟踪
            bbox, confidence = tracker.track(rgbd_image)
            x, y, w, h = bbox

            t_frame_end = time.perf_counter()
            frame_time = t_frame_end - t_frame_start
            fps = 1.0 / max(frame_time, 1e-6)

            # 每50帧打印一次
            if frame_id % 50 == 0:
                avg_fps = tracker.get_avg_fps()
                print(f"[INFO] Frame {frame_id}: bbox=[{x:.1f},{y:.1f},{w:.1f},{h:.1f}], "
                      f"conf={confidence:.3f}, FPS={fps:.1f}, Avg_Infer_FPS={avg_fps:.1f}")

            # 画框
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            box_color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)
            cv2.rectangle(vis, p1, p2, box_color, 2)

            # 显示信息
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"Infer FPS: {tracker.get_avg_fps():.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(vis, f"Frame: {frame_id}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"Conf: {confidence:.3f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if confidence > 0.7 else (0, 0, 255), 2)

            # 深度图
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.rectangle(depth_colormap, p1, p2, (0, 255, 0), 2)
            cv2.putText(depth_colormap, "Depth", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(win_name, vis)
            cv2.imshow("Depth View", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # 重新初始化
                print("\n[INFO] Re-initialization requested...")
                roi = cv2.selectROI(win_name, vis, fromCenter=False, showCrosshair=True)
                x_new, y_new, w_new, h_new = roi
                if w_new > 0 and h_new > 0:
                    init_bbox = [float(x_new), float(y_new), float(w_new), float(h_new)]
                    color_rgb_uint8, depth_3ch_uint8 = preprocess_rgb_depth(color_bgr, depth_raw, depth_scale)
                    rgbd_image = np.concatenate([color_rgb_uint8, depth_3ch_uint8], axis=2)
                    tracker.initialize(rgbd_image, init_bbox)
                    print(f"[SUCCESS] Re-initialized with bbox={init_bbox}")
                    frame_id = 0
                continue

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
