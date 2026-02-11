#!/usr/bin/env python3
"""
从VOT现有结果中提取FPS数据
解析*_time.value文件并生成fps_summary.txt
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path


def extract_fps_from_time_files(tracker_path):
    """从time.value文件中提取FPS数据"""
    
    # 查找所有time.value文件
    time_files = glob.glob(os.path.join(tracker_path, '**/*_time.value'), recursive=True)
    
    if not time_files:
        return None
    
    all_times = []
    
    # 读取所有时间数据
    for time_file in time_files:
        try:
            with open(time_file, 'r') as f:
                times = [float(line.strip()) for line in f if line.strip()]
                all_times.extend(times)
        except Exception as e:
            print(f"Warning: Failed to read {time_file}: {e}")
            continue
    
    if not all_times:
        return None
    
    # 计算FPS统计
    times_array = np.array(all_times)
    
    # 过滤异常值：
    # 1. 排除过小的时间（< 0.001s，即 > 1000 FPS）- 这些是初始化帧或缓存帧
    # 2. 排除过大的时间（> 0.5s，即 < 2 FPS）- 这些是异常慢帧
    min_time_threshold = 0.001  # 1ms
    max_time_threshold = 0.5    # 500ms
    
    valid_mask = (times_array >= min_time_threshold) & (times_array <= max_time_threshold)
    valid_times = times_array[valid_mask]
    
    # 如果过滤后数据太少，使用全部数据
    if len(valid_times) < len(all_times) * 0.5:
        print(f"  Warning: Too many outliers filtered ({len(all_times) - len(valid_times)}/{len(all_times)}), using all data")
        valid_times = times_array
    
    fps_array = 1.0 / valid_times
    
    fps_data = {
        'avg_fps': np.mean(fps_array),
        'max_fps': np.max(fps_array),
        'min_fps': np.min(fps_array),
        'total_frames': len(all_times),
        'valid_frames': len(valid_times),
        'total_time': np.sum(times_array),
        'filtered_outliers': len(all_times) - len(valid_times)
    }
    
    return fps_data


def generate_fps_summary(tracker_path, tracker_name):
    """生成fps_summary.txt文件"""
    
    fps_data = extract_fps_from_time_files(tracker_path)
    
    if not fps_data:
        print(f"Warning: No timing data found for {tracker_name}")
        return False
    
    # 写入fps_summary.txt
    summary_file = os.path.join(tracker_path, 'fps_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write(f"Average FPS: {fps_data['avg_fps']:.2f}\n")
        f.write(f"Max FPS: {fps_data['max_fps']:.2f}\n")
        f.write(f"Min FPS: {fps_data['min_fps']:.2f}\n")
        f.write(f"Total Frames: {fps_data['total_frames']}\n")
        f.write(f"Valid Frames: {fps_data['valid_frames']}\n")
        if fps_data['filtered_outliers'] > 0:
            f.write(f"Filtered Outliers: {fps_data['filtered_outliers']}\n")
        f.write(f"Total Time: {fps_data['total_time']:.2f}s\n")
    
    print(f"✓ Generated FPS summary for {tracker_name}")
    print(f"  Average FPS: {fps_data['avg_fps']:.2f}")
    print(f"  Valid Frames: {fps_data['valid_frames']}/{fps_data['total_frames']}")
    if fps_data['filtered_outliers'] > 0:
        print(f"  Filtered {fps_data['filtered_outliers']} outlier frames")
    
    return True


def process_workspace(workspace_path):
    """处理整个VOT工作区"""
    
    results_dir = os.path.join(workspace_path, 'results')
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return 0
    
    processed_count = 0
    
    # 遍历所有tracker目录
    for tracker_name in os.listdir(results_dir):
        tracker_path = os.path.join(results_dir, tracker_name)
        
        if not os.path.isdir(tracker_path):
            continue
        
        # 检查是否已有fps_summary.txt
        summary_file = os.path.join(tracker_path, 'fps_summary.txt')
        if os.path.exists(summary_file):
            print(f"○ Skipping {tracker_name} (fps_summary.txt already exists)")
            processed_count += 1
            continue
        
        # 生成FPS摘要
        if generate_fps_summary(tracker_path, tracker_name):
            processed_count += 1
    
    return processed_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='从VOT结果中提取FPS数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 处理VOT2019工作区
  python extract_vot_fps.py --workspace vot-workspace-rgbd2019
  
  # 处理VOT2022工作区
  python extract_vot_fps.py --workspace vot-workspace-rgbd2022
  
  # 处理所有工作区
  python extract_vot_fps.py --all
  
  # 强制重新生成（覆盖现有文件）
  python extract_vot_fps.py --workspace vot-workspace-rgbd2019 --force
        '''
    )
    
    parser.add_argument('--workspace', type=str,
                        help='VOT工作区路径')
    parser.add_argument('--all', action='store_true',
                        help='处理所有VOT工作区')
    parser.add_argument('--force', action='store_true',
                        help='强制重新生成（覆盖现有fps_summary.txt）')
    
    args = parser.parse_args()
    
    if args.all:
        workspaces = [
            'vot-workspace-rgbd2019',
            'vot-workspace-rgbd2022'
        ]
    elif args.workspace:
        workspaces = [args.workspace]
    else:
        print("Error: Please specify --workspace or --all")
        parser.print_help()
        sys.exit(1)
    
    total_processed = 0
    
    for workspace in workspaces:
        if not os.path.exists(workspace):
            print(f"\nWarning: Workspace not found: {workspace}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {workspace}")
        print('='*80)
        
        # 如果force模式，删除现有fps_summary.txt
        if args.force:
            results_dir = os.path.join(workspace, 'results')
            if os.path.exists(results_dir):
                for tracker_name in os.listdir(results_dir):
                    summary_file = os.path.join(results_dir, tracker_name, 'fps_summary.txt')
                    if os.path.exists(summary_file) and os.path.isfile(summary_file):
                        os.remove(summary_file)
                        print(f"Removed existing: {summary_file}")
        
        count = process_workspace(workspace)
        total_processed += count
        
        print(f"\nProcessed {count} trackers in {workspace}")
    
    print(f"\n{'='*80}")
    print(f"Total: {total_processed} trackers processed")
    print('='*80)
    
    if total_processed > 0:
        print("\n现在可以运行 ./show_vot_fps.sh 查看FPS统计了！")


if __name__ == '__main__':
    main()
