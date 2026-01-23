#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUTrack TensorRT 推理脚本 (Jetson 平台)

基于 demo_realsense.py 的完整预处理流程，使用 TensorRT 进行推理。
确保与 PyTorch 版本的预处理完全一致。

运行方式（Jetson Orin NX）：
  1) 确保已用 trtexec 生成 sutrack_tiny_rgbd_fp16.engine
  2) python demo_trt_jetson.py --engine /path/to/sutrack_tiny_rgbd_fp16.engine

操作说明：
  - 按 's' 键：选取初始目标（用鼠标框选）
  - 按 'r' 键：重新初始化
  - 按 ESC：退出
"""

import os
import sys
import time
import math
import argparse
import cv2
import numpy as np

# NumPy 兼容性补丁：解决 NumPy 1.24+ 移除 np.bool 导致 TensorRT 报错的问题
if not hasattr(np, "bool_"):
    np.bool_ = bool
if not hasattr(np, "bool"):
    np.bool = np.bool_

import pyrealsense2 as rs

# TensorRT Python API
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # 自动初始化 CUDA context
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError as e:
    print(f"[ERROR] 无法导入 TensorRT/PyCUDA: {e}")
    print("请先安装：pip install tensorrt pycuda")
    sys.exit(1)


# ============================================================================
# 1. RealSense 相机模块
# ============================================================================

class RealSenseCamera:
    """RealSense D435i 相机封装"""
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        
        self._init_camera()
    
    def _init_camera(self):
        """初始化相机"""
        print("[Camera] Initializing RealSense D435i...")
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        
        profile = self.pipeline.start(config)
        
        # 深度对齐到彩色
        self.align = rs.align(rs.stream.color)
        
        # 获取深度比例
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        print(f"[Camera] Started. depth_scale = {self.depth_scale:.6f} m/unit")
        
        # 等待相机稳定
        print("[Camera] Waiting for stabilization...")
        for _ in range(30):
            try:
                self.pipeline.wait_for_frames(timeout_ms=1000)
            except:
                pass
        print("[Camera] Ready.")
    
    def get_frame(self, timeout_ms=5000):
        """
        获取一帧数据
        
        Returns:
            color_bgr: (H, W, 3) BGR uint8
            depth_raw: (H, W) uint16
        """
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            
            return color_bgr, depth_raw
            
        except Exception as e:
            print(f"[Camera] Error getting frame: {e}")
            return None, None
    
    def preprocess_rgbd(self, color_bgr, depth_raw, max_dist_m=5.0):
        """
        预处理为 6 通道 RGBD 图像（与 demo_realsense.py 一致）
        
        Args:
            color_bgr: (H, W, 3) BGR uint8
            depth_raw: (H, W) uint16
            max_dist_m: 最大深度距离（米）
        
        Returns:
            rgbd_image: (H, W, 6) uint8 [0-255]
                前3通道: RGB
                后3通道: Depth（重复3次）
        """
        # BGR -> RGB
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        
        # 深度处理: raw -> meters -> [0,1] -> [0,255]
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        depth_m = np.clip(depth_m, 0.0, max_dist_m)
        depth_norm = depth_m / max_dist_m
        depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
        
        # 深度扩展为 3 通道
        depth_3ch = np.stack([depth_uint8, depth_uint8, depth_uint8], axis=2)
        
        # 合并为 6 通道
        rgbd_image = np.concatenate([color_rgb, depth_3ch], axis=2)
        
        return rgbd_image
    
    def cleanup(self):
        """清理资源"""
        if self.pipeline:
            self.pipeline.stop()


# ============================================================================
# 2. TensorRT 推理引擎
# ============================================================================

class TRTEngine:
    """TensorRT 推理引擎封装"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = cuda.Stream()
        
        self.inputs = {}
        self.outputs = {}
        self.bindings = []
        
        self._load_engine()
        self._allocate_buffers()
    
    def _load_engine(self):
        """加载 TensorRT 引擎"""
        print(f"[TRT] Loading engine: {self.engine_path}")
        
        with open(self.engine_path, "rb") as f:
            self.runtime = trt.Runtime(TRT_LOGGER)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        print(f"[TRT] Engine loaded. Bindings: {self.engine.num_bindings}")
    
    def _allocate_buffers(self):
        """分配 GPU 内存"""
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
        执行推理
        
        Args:
            template: (1, 6, 112, 112) float32
            search: (1, 6, 224, 224) float32
            template_anno: (1, 4) float32
        
        Returns:
            pred_boxes: (1, 1, 4) float32 - (cx, cy, w, h) 归一化坐标
            score_map: (1, 1, 14, 14) float32 - 原始 logit（未经 Sigmoid）
        """
        # 拷贝输入到 GPU
        input_map = {
            "template": template,
            "search": search,
            "template_anno": template_anno
        }
        
        for name, data in input_map.items():
            if name in self.inputs:
                inp = self.inputs[name]
                host_data = np.ascontiguousarray(data.astype(inp['dtype']))
                cuda.memcpy_htod_async(inp['device_mem'], host_data, self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 拷贝输出到 CPU
        results = {}
        for name, out in self.outputs.items():
            host_mem = np.empty(out['size'], dtype=out['dtype'])
            cuda.memcpy_dtoh_async(host_mem, out['device_mem'], self.stream)
            results[name] = host_mem.reshape(out['shape'])
        
        self.stream.synchronize()
        
        return results.get("pred_boxes"), results.get("score_map")


# ============================================================================
# 3. SUTrack TensorRT 跟踪器（与 PyTorch 版本预处理完全一致）
# ============================================================================

class SUTrackTRTTracker:
    """
    使用 TensorRT 的 SUTrack 跟踪器
    预处理流程与 lib/test/tracker/sutrack.py 完全一致
    """
    
    def __init__(self, engine_path: str, template_size=112, search_size=224):
        self.engine = TRTEngine(engine_path)
        self.template_size = template_size
        self.search_size = search_size
        
        # 配置参数（与 sutrack_t224.yaml 一致）
        self.template_factor = 2.0
        self.search_factor = 4.0
        
        # 归一化参数（与 Preprocessor 一致）
        # 6 通道：RGB + Depth（Depth 复用 RGB 的归一化参数）
        self.mean = np.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 6, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 6, 1, 1)
        
        # Hann 窗（与 PyTorch 版本一致）
        self.fx_sz = self.search_size // 16  # stride=16
        self.hann_window = self._create_hann2d(self.fx_sz)
        
        # 状态
        self.initialized = False
        self.state = None  # [x, y, w, h] 绝对坐标
        self.template_patch = None
        self.template_anno = None
        
        # 性能统计
        self.fps_history = []
    
    def _create_hann2d(self, size):
        """创建 2D Hann 窗"""
        hann_1d = np.hanning(size + 2)[1:-1]
        hann_2d = np.outer(hann_1d, hann_1d)
        return hann_2d.astype(np.float32)
    
    def _sample_target(self, image: np.ndarray, target_bb: list, search_area_factor: float, output_sz: int):
        """
        裁剪目标区域（与 lib/test/tracker/utils.py 中的 sample_target 完全一致）
        
        Args:
            image: (H, W, 6) uint8 RGBD 图像
            target_bb: [x, y, w, h] 绝对坐标
            search_area_factor: 裁剪区域相对目标的倍数
            output_sz: 输出尺寸
        
        Returns:
            im_crop_padded: (output_sz, output_sz, 6) uint8
            resize_factor: float
        """
        x, y, w, h = target_bb
        
        # 计算裁剪尺寸
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
        if crop_sz < 1:
            crop_sz = 1
        
        # 计算裁剪区域
        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz
        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz
        
        # 计算 padding
        x1_pad = max(0, -x1)
        x2_pad = max(x2 - image.shape[1] + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - image.shape[0] + 1, 0)
        
        # 裁剪
        im_crop = image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        
        # Padding（使用 0 填充）
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=0)
        
        # Resize（分别处理 RGB 和 Depth 通道，避免插值问题）
        resize_factor = output_sz / crop_sz
        rgb_resized = cv2.resize(im_crop_padded[:, :, :3], (output_sz, output_sz))
        depth_resized = cv2.resize(im_crop_padded[:, :, 3:], (output_sz, output_sz))
        im_resized = np.concatenate([rgb_resized, depth_resized], axis=2)
        
        return im_resized, resize_factor
    
    def _preprocess(self, patch: np.ndarray):
        """
        归一化预处理（与 Preprocessor.process 完全一致）
        
        Args:
            patch: (H, W, 6) uint8 [0-255]
        
        Returns:
            normalized: (1, 6, H, W) float32
        """
        # HWC -> CHW
        patch_chw = patch.transpose(2, 0, 1).astype(np.float32)
        patch_chw = patch_chw[np.newaxis, ...]  # (1, 6, H, W)
        
        # 归一化: /255 -> (x - mean) / std
        patch_norm = (patch_chw / 255.0 - self.mean) / self.std
        
        return patch_norm.astype(np.float32)
    
    def _transform_image_to_crop(self, box_in: list, box_extract: list, resize_factor: float, crop_sz: int):
        """
        计算 bbox 在裁剪图像中的归一化坐标（与 transform_image_to_crop 一致）
        
        Args:
            box_in: [x, y, w, h] 输入框
            box_extract: [x, y, w, h] 裁剪中心框
            resize_factor: 缩放因子
            crop_sz: 裁剪后尺寸
        
        Returns:
            box_out: [x, y, w, h] 归一化坐标 [0, 1]
        """
        # 中心点
        box_extract_center = np.array([box_extract[0] + 0.5 * box_extract[2],
                                       box_extract[1] + 0.5 * box_extract[3]])
        box_in_center = np.array([box_in[0] + 0.5 * box_in[2],
                                  box_in[1] + 0.5 * box_in[3]])
        
        # 转换到裁剪坐标系
        box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
        box_out_wh = np.array([box_in[2], box_in[3]]) * resize_factor
        
        # 转为 [x, y, w, h] 格式
        box_out = np.array([
            box_out_center[0] - 0.5 * box_out_wh[0],
            box_out_center[1] - 0.5 * box_out_wh[1],
            box_out_wh[0],
            box_out_wh[1]
        ])
        
        # 归一化到 [0, 1]
        box_out_norm = box_out / (crop_sz - 1)
        
        return box_out_norm
    
    def initialize(self, rgbd_image: np.ndarray, init_bbox: list):
        """
        初始化跟踪器
        
        Args:
            rgbd_image: (H, W, 6) uint8 RGBD 图像
            init_bbox: [x, y, w, h] 初始边界框（绝对坐标）
        """
        self.state = init_bbox
        
        # 1. 裁剪模板 patch
        z_patch, resize_factor = self._sample_target(
            rgbd_image, init_bbox, self.template_factor, self.template_size
        )
        
        # 2. 归一化
        self.template_patch = self._preprocess(z_patch)
        
        # 3. 计算 template_anno（与 PyTorch 版本一致）
        template_anno = self._transform_image_to_crop(
            init_bbox, init_bbox, resize_factor, self.template_size
        )
        self.template_anno = template_anno.reshape(1, 4).astype(np.float32)
        
        self.initialized = True
        self.fps_history = []
        
        print(f"[Tracker] Initialized with bbox: {init_bbox}")
        print(f"[Tracker] Template anno (normalized): {self.template_anno}")
    
    def track(self, rgbd_image: np.ndarray):
        """
        执行跟踪
        
        Args:
            rgbd_image: (H, W, 6) uint8 RGBD 图像
        
        Returns:
            bbox: [x, y, w, h] 跟踪结果（绝对坐标）
            confidence: float 置信度
        """
        if not self.initialized:
            return [0, 0, 50, 50], 0.0
        
        H, W = rgbd_image.shape[:2]
        
        # 1. 裁剪搜索区域
        x_patch, resize_factor = self._sample_target(
            rgbd_image, self.state, self.search_factor, self.search_size
        )
        
        # 2. 归一化
        search_norm = self._preprocess(x_patch)
        
        # 3. TensorRT 推理
        t0 = time.perf_counter()
        pred_boxes, score_map = self.engine.infer(self.template_patch, search_norm, self.template_anno)
        t1 = time.perf_counter()
        self.fps_history.append(1.0 / max(t1 - t0, 1e-6))
        
        # 4. 后处理
        # 4.1 对 score_map 应用 Sigmoid（ONNX 导出时未内置）
        score_map_prob = 1.0 / (1.0 + np.exp(-score_map))
        
        # 4.2 应用 Hann 窗惩罚
        response = score_map_prob[0, 0]  # (14, 14)
        if response.shape == self.hann_window.shape:
            response = response * self.hann_window
        
        confidence = float(np.max(response))
        
        # 4.3 解析预测框
        # pred_boxes 输出格式: (1, 1, 4) -> (cx, cy, w, h) 归一化坐标
        pred_box = pred_boxes[0, 0, :]  # (4,)
        
        # 4.4 映射回原图坐标
        bbox = self._map_box_back(pred_box, resize_factor, H, W)
        
        # 4.5 更新状态（仅当置信度足够高时）
        # Tiny 模型置信度普遍较低，阈值设为 0.05
        if confidence > 0.05:
            self.state = bbox
        
        return self.state, confidence
    
    def _map_box_back(self, pred_box: np.ndarray, resize_factor: float, img_h: int, img_w: int):
        """
        将预测框从搜索区域坐标映射回原图坐标（与 PyTorch 版本一致）
        
        Args:
            pred_box: (4,) [cx, cy, w, h] 归一化坐标 [0, 1]
            resize_factor: 缩放因子
            img_h, img_w: 原图尺寸
        
        Returns:
            bbox: [x, y, w, h] 绝对坐标
        """
        # 上一帧状态的中心
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        
        # 从归一化坐标还原到搜索区域像素坐标
        cx_rel, cy_rel, w_rel, h_rel = pred_box
        
        # 方法与 PyTorch 版本一致：
        # pred_box 是归一化坐标，乘以 search_size 得到像素坐标
        cx_patch = cx_rel * self.search_size
        cy_patch = cy_rel * self.search_size
        w_patch = w_rel * self.search_size
        h_patch = h_rel * self.search_size
        
        # 从搜索区域坐标映射到原图坐标
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx_patch / resize_factor + (cx_prev - half_side)
        cy_real = cy_patch / resize_factor + (cy_prev - half_side)
        w_real = w_patch / resize_factor
        h_real = h_patch / resize_factor
        
        # 转为 [x, y, w, h] 格式
        x = cx_real - 0.5 * w_real
        y = cy_real - 0.5 * h_real
        
        # 边界裁剪（与 clip_box 一致）
        margin = 10
        x = max(0, min(img_w - margin, x))
        y = max(0, min(img_h - margin, y))
        w = max(margin, min(img_w, w_real))
        h = max(margin, min(img_h, h_real))
        
        return [x, y, w, h]
    
    def get_avg_fps(self):
        """获取平均推理 FPS"""
        if len(self.fps_history) == 0:
            return 0.0
        return sum(self.fps_history[-100:]) / min(len(self.fps_history), 100)


# ============================================================================
# 4. 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SUTrack TensorRT Demo for Jetson')
    parser.add_argument('--engine', type=str, required=True,
                        help='Path to TensorRT engine file (e.g., sutrack_tiny_rgbd_fp16.engine)')
    parser.add_argument('--template_size', type=int, default=112,
                        help='Template size (default: 112)')
    parser.add_argument('--search_size', type=int, default=224,
                        help='Search size (default: 224)')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("SUTrack TensorRT Demo (Jetson)")
    print("=" * 70)
    print(f"Engine: {args.engine}")
    print(f"Template size: {args.template_size}")
    print(f"Search size: {args.search_size}")
    print("=" * 70 + "\n")
    
    # 检查引擎文件
    if not os.path.exists(args.engine):
        print(f"[ERROR] Engine file not found: {args.engine}")
        print("[INFO] Please run trtexec to convert ONNX to TensorRT engine:")
        print("  /usr/src/tensorrt/bin/trtexec \\")
        print("      --onnx=sutrack_tiny_rgbd.onnx \\")
        print("      --saveEngine=sutrack_tiny_rgbd_fp16.engine \\")
        print("      --fp16 --useCudaGraph --verbose")
        return 1
    
    # 初始化相机
    try:
        camera = RealSenseCamera()
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera: {e}")
        return 1
    
    # 初始化跟踪器
    try:
        tracker = SUTrackTRTTracker(args.engine, args.template_size, args.search_size)
    except Exception as e:
        print(f"[ERROR] Failed to initialize tracker: {e}")
        camera.cleanup()
        return 1
    
    # 创建窗口
    win_name = "SUTrack TRT Tracking (Jetson)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    frame_id = 0
    init_bbox = None
    
    print("\n[INFO] Starting tracking loop...")
    print("[INFO] Press 's' to select target, 'r' to re-init, ESC to quit\n")
    
    try:
        while True:
            # 获取帧
            color_bgr, depth_raw = camera.get_frame()
            if color_bgr is None:
                print("[WARNING] Failed to get frame, skipping...")
                continue
            
            vis = color_bgr.copy()
            
            # 未初始化：等待用户选择目标
            if init_bbox is None:
                cv2.putText(vis, "Press 's' to select target, ESC to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(win_name, vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('s'):
                    print("\n[INFO] Please select ROI...")
                    roi = cv2.selectROI(win_name, vis, fromCenter=False, showCrosshair=True)
                    x, y, w, h = roi
                    
                    if w > 0 and h > 0:
                        init_bbox = [float(x), float(y), float(w), float(h)]
                        
                        # 预处理 RGBD
                        rgbd_image = camera.preprocess_rgbd(color_bgr, depth_raw)
                        
                        # 初始化跟踪器
                        tracker.initialize(rgbd_image, init_bbox)
                        print(f"\n[SUCCESS] Tracking started! bbox={init_bbox}\n")
                    else:
                        print("[WARNING] Invalid ROI selection")
                continue
            
            # 正常跟踪
            t_start = time.perf_counter()
            
            # 预处理 RGBD
            rgbd_image = camera.preprocess_rgbd(color_bgr, depth_raw)
            
            # 跟踪
            bbox, confidence = tracker.track(rgbd_image)
            
            t_end = time.perf_counter()
            fps = 1.0 / max(t_end - t_start, 1e-6)
            
            # 可视化
            x, y, w, h = bbox
            
            # 根据置信度选择颜色
            if confidence > 0.8:
                color = (0, 255, 0)    # 绿色 - 高置信度
            elif confidence > 0.5:
                color = (0, 255, 255)  # 黄色 - 中等置信度
            else:
                color = (0, 0, 255)    # 红色 - 低置信度
            
            # 绘制边界框
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(vis, p1, p2, color, 2)
            
            # 显示信息
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"Infer FPS: {tracker.get_avg_fps():.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, f"Frame: {frame_id}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"Conf: {confidence:.3f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 低置信度警告
            if confidence < 0.3:
                cv2.putText(vis, "LOW CONFIDENCE - Press 'r' to re-init",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 显示深度图（可选）
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03),
                cv2.COLORMAP_JET
            )
            cv2.rectangle(depth_vis, p1, p2, (255, 255, 255), 2)
            
            cv2.imshow(win_name, vis)
            cv2.imshow("Depth View", depth_vis)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('s'):
                print("\n[INFO] Re-initialization requested...")
                roi = cv2.selectROI(win_name, vis, fromCenter=False, showCrosshair=True)
                x_new, y_new, w_new, h_new = roi
                if w_new > 0 and h_new > 0:
                    init_bbox = [float(x_new), float(y_new), float(w_new), float(h_new)]
                    rgbd_image = camera.preprocess_rgbd(color_bgr, depth_raw)
                    tracker.initialize(rgbd_image, init_bbox)
                    print(f"[SUCCESS] Re-initialized with bbox={init_bbox}")
                    frame_id = 0
                continue
            
            # 每 50 帧打印状态
            if frame_id % 50 == 0:
                print(f"[INFO] Frame {frame_id}: bbox=[{x:.1f},{y:.1f},{w:.1f},{h:.1f}], "
                      f"conf={confidence:.3f}, FPS={fps:.1f}")
            
            frame_id += 1
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Cleaning up...")
        camera.cleanup()
        cv2.destroyAllWindows()
        print("[INFO] Done.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
