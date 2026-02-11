#!/usr/bin/env python3
import numpy as np
import glob
import os

trackers = [
    'sutrack_t224',
    'sutrack_mlka_t224',
    'sutrack_sgla_t224',
    'sutrack_or_t224',
    'sutrack_smfa_t224',
    'sutrack_rmt_t224',
    'sutrack_mfe_t224'
]

print('='*80)
print('所有模型的极慢帧统计')
print('='*80)
print(f"{'模型':<25} {'最慢帧(s)':<12} {'最慢FPS':<12} {'>0.1s帧数':<12} {'占比':<10}")
print('-'*80)

for tracker in trackers:
    tracker_path = f'vot-workspace-rgbd2019/results/{tracker}'
    time_files = glob.glob(os.path.join(tracker_path, '**/*_time.value'), recursive=True)
    
    all_times = []
    for f in time_files:
        with open(f) as file:
            times = [float(line.strip()) for line in file if line.strip()]
            all_times.extend(times)
    
    times_array = np.array(all_times)
    max_time = np.max(times_array)
    min_fps = 1.0 / max_time
    slow_frames = np.sum(times_array > 0.1)
    slow_ratio = slow_frames / len(times_array) * 100
    
    print(f'{tracker:<25} {max_time:<12.4f} {min_fps:<12.2f} {slow_frames:<12} {slow_ratio:<10.3f}%')

print('='*80)
print('\n极慢帧的可能原因分析:')
print('1. 目标丢失后的全局搜索（重新初始化）')
print('2. 严重遮挡导致的特征不匹配，需要额外计算')
print('3. 复杂背景下的大范围搜索')
print('4. 尺度剧烈变化时的多尺度计算')
print('5. 跟踪失败后的恢复机制触发')
