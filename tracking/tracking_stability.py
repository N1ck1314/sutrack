"""
跟踪稳定性增强模块
用于提高目标丢失时的跟踪稳定性
"""
import numpy as np
import cv2
from collections import deque


class TrackingStabilityEnhancer:
    """
    跟踪稳定性增强器
    提供多种机制来提高跟踪稳定性
    """
    
    def __init__(self, 
                 confidence_threshold=0.3,
                 low_confidence_threshold=0.5,
                 max_bbox_velocity=0.3,
                 smoothing_window=5,
                 search_expansion_factor=1.5,
                 lost_frames_threshold=10):
        """
        Args:
            confidence_threshold: 置信度阈值，低于此值认为目标丢失
            low_confidence_threshold: 低置信度阈值，低于此值启用稳定性机制
            max_bbox_velocity: 最大bbox变化速度（相对于图像尺寸的比例）
            smoothing_window: 平滑窗口大小（帧数）
            search_expansion_factor: 低置信度时搜索区域扩大倍数
            lost_frames_threshold: 连续丢失帧数阈值
        """
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.max_bbox_velocity = max_bbox_velocity
        self.smoothing_window = smoothing_window
        self.search_expansion_factor = search_expansion_factor
        self.lost_frames_threshold = lost_frames_threshold
        
        # 历史记录
        self.bbox_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        self.velocity_history = deque(maxlen=3)
        
        # 状态
        self.lost_frames = 0
        self.is_lost = False
        self.last_valid_bbox = None
        self.last_valid_confidence = 0.0
        
    def update(self, bbox, confidence, image_shape):
        """
        更新跟踪状态
        
        Args:
            bbox: [x, y, w, h] 当前预测的bbox
            confidence: 当前置信度分数
            image_shape: (H, W) 图像尺寸
        
        Returns:
            stabilized_bbox: 稳定化后的bbox
            status: 状态字典 {'is_lost', 'confidence', 'lost_frames'}
        """
        H, W = image_shape[:2]
        
        # 归一化bbox到[0,1]范围以便计算速度
        norm_bbox = np.array([
            bbox[0] / W, bbox[1] / H,
            bbox[2] / W, bbox[3] / H
        ])
        
        # 计算速度（如果历史记录存在）
        velocity = 0.0
        if len(self.bbox_history) > 0:
            last_bbox = self.bbox_history[-1]
            velocity = np.sqrt(np.sum((norm_bbox - last_bbox) ** 2))
        
        # 检测目标丢失
        if confidence < self.confidence_threshold:
            self.lost_frames += 1
            self.is_lost = True
        else:
            if self.lost_frames > 0:
                self.lost_frames = max(0, self.lost_frames - 1)
            if self.lost_frames == 0:
                self.is_lost = False
                self.last_valid_bbox = bbox.copy()
                self.last_valid_confidence = confidence
        
        # 应用稳定性机制
        stabilized_bbox = self._apply_stability_mechanisms(
            bbox, confidence, velocity, image_shape
        )
        
        # 更新历史记录
        self.bbox_history.append(norm_bbox)
        self.confidence_history.append(confidence)
        self.velocity_history.append(velocity)
        
        status = {
            'is_lost': self.is_lost,
            'confidence': confidence,
            'lost_frames': self.lost_frames,
            'velocity': velocity,
            'needs_recovery': self.lost_frames > self.lost_frames_threshold
        }
        
        return stabilized_bbox, status
    
    def _apply_stability_mechanisms(self, bbox, confidence, velocity, image_shape):
        """
        应用稳定性机制
        
        Returns:
            stabilized_bbox: 稳定化后的bbox
        """
        H, W = image_shape[:2]
        
        # 如果目标完全丢失，使用历史预测
        if self.is_lost and self.last_valid_bbox is not None:
            # 使用简单的运动模型预测
            if len(self.bbox_history) >= 2:
                # 计算平均速度
                velocities = list(self.velocity_history)
                if len(velocities) >= 2:
                    avg_velocity = np.mean(velocities[-2:])
                    # 基于速度预测位置
                    predicted_bbox = self._predict_bbox_from_history(avg_velocity, image_shape)
                    if predicted_bbox is not None:
                        return predicted_bbox
            
            # 如果无法预测，返回最后有效bbox
            return self.last_valid_bbox.copy()
        
        # 低置信度时的处理
        if confidence < self.low_confidence_threshold:
            # 应用平滑
            if len(self.bbox_history) >= 2:
                smoothed_bbox = self._smooth_bbox(bbox, image_shape)
            else:
                smoothed_bbox = bbox
            
            # 限制速度
            if velocity > self.max_bbox_velocity:
                # 限制bbox变化
                if len(self.bbox_history) > 0:
                    last_bbox = self.bbox_history[-1]
                    last_bbox_abs = np.array([
                        last_bbox[0] * W, last_bbox[1] * H,
                        last_bbox[2] * W, last_bbox[3] * H
                    ])
                    # 限制变化幅度
                    max_change = self.max_bbox_velocity * np.sqrt(W * H)
                    change = np.array(smoothed_bbox) - last_bbox_abs
                    change_norm = np.linalg.norm(change)
                    if change_norm > max_change:
                        change = change / change_norm * max_change
                        smoothed_bbox = (last_bbox_abs + change).tolist()
            
            return smoothed_bbox
        
        # 正常情况：应用轻微平滑
        if len(self.bbox_history) >= 2:
            return self._smooth_bbox(bbox, image_shape)
        
        return bbox
    
    def _smooth_bbox(self, current_bbox, image_shape):
        """
        使用移动平均平滑bbox
        """
        H, W = image_shape[:2]
        
        if len(self.bbox_history) < 2:
            return current_bbox
        
        # 将历史bbox转换回绝对坐标
        history_abs = []
        for norm_bbox in self.bbox_history:
            history_abs.append([
                norm_bbox[0] * W, norm_bbox[1] * H,
                norm_bbox[2] * W, norm_bbox[3] * H
            ])
        
        # 计算加权平均（最近帧权重更高）
        weights = np.linspace(0.5, 1.0, len(history_abs))
        weights = weights / weights.sum()
        
        smoothed = np.zeros(4)
        for i, bbox in enumerate(history_abs):
            smoothed += np.array(bbox) * weights[i]
        
        # 当前帧权重
        current_weight = 0.7
        smoothed = smoothed * (1 - current_weight) + np.array(current_bbox) * current_weight
        
        return smoothed.tolist()
    
    def _predict_bbox_from_history(self, velocity, image_shape):
        """
        基于历史轨迹预测bbox位置
        """
        if len(self.bbox_history) < 2:
            return None
        
        H, W = image_shape[:2]
        
        # 获取最后两个bbox
        last_bbox = self.bbox_history[-1]
        prev_bbox = self.bbox_history[-2]
        
        # 计算位移
        displacement = (np.array(last_bbox) - np.array(prev_bbox)) * np.array([W, H, W, H])
        
        # 预测下一帧位置
        predicted_norm = last_bbox + displacement / np.array([W, H, W, H]) * 0.5  # 衰减因子
        predicted_abs = [
            predicted_norm[0] * W,
            predicted_norm[1] * H,
            predicted_norm[2] * W,
            predicted_norm[3] * H
        ]
        
        # 边界检查
        predicted_abs[0] = max(0, min(W - predicted_abs[2], predicted_abs[0]))
        predicted_abs[1] = max(0, min(H - predicted_abs[3], predicted_abs[1]))
        predicted_abs[2] = max(10, min(W - predicted_abs[0], predicted_abs[2]))
        predicted_abs[3] = max(10, min(H - predicted_abs[1], predicted_abs[3]))
        
        return predicted_abs
    
    def get_search_expansion_factor(self, confidence):
        """
        根据置信度获取搜索区域扩大倍数
        """
        if confidence < self.low_confidence_threshold:
            return self.search_expansion_factor
        return 1.0
    
    def reset(self):
        """重置状态"""
        self.bbox_history.clear()
        self.confidence_history.clear()
        self.velocity_history.clear()
        self.lost_frames = 0
        self.is_lost = False
        self.last_valid_bbox = None
        self.last_valid_confidence = 0.0


class KalmanFilter:
    """
    简单的卡尔曼滤波器用于bbox平滑
    """
    def __init__(self, state_dim=4):
        self.state_dim = state_dim
        self.state = np.zeros(state_dim)  # [x, y, w, h]
        self.covariance = np.eye(state_dim) * 1000
        
        # 过程噪声和测量噪声
        self.process_noise = np.eye(state_dim) * 0.1
        self.measurement_noise = np.eye(state_dim) * 10
        
        # 状态转移矩阵（简单恒速模型）
        self.transition_matrix = np.eye(state_dim)
        
        # 测量矩阵
        self.measurement_matrix = np.eye(state_dim)
        
        self.initialized = False
    
    def update(self, measurement):
        """
        更新卡尔曼滤波器
        
        Args:
            measurement: [x, y, w, h] 测量值
        
        Returns:
            filtered_state: 滤波后的状态
        """
        if not self.initialized:
            self.state = np.array(measurement)
            self.initialized = True
            return self.state.copy()
        
        # 预测步骤
        predicted_state = self.transition_matrix @ self.state
        predicted_cov = self.transition_matrix @ self.covariance @ self.transition_matrix.T + self.process_noise
        
        # 更新步骤
        innovation = np.array(measurement) - self.measurement_matrix @ predicted_state
        innovation_cov = self.measurement_matrix @ predicted_cov @ self.measurement_matrix.T + self.measurement_noise
        
        kalman_gain = predicted_cov @ self.measurement_matrix.T @ np.linalg.inv(innovation_cov)
        
        self.state = predicted_state + kalman_gain @ innovation
        self.covariance = (np.eye(self.state_dim) - kalman_gain @ self.measurement_matrix) @ predicted_cov
        
        return self.state.copy()
    
    def predict(self):
        """预测下一帧状态"""
        if not self.initialized:
            return None
        return self.transition_matrix @ self.state
    
    def reset(self):
        """重置滤波器"""
        self.state = np.zeros(self.state_dim)
        self.covariance = np.eye(self.state_dim) * 1000
        self.initialized = False

