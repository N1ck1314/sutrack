#!/usr/bin/env python3
"""
VOT FPS统计脚本
从VOT评测结果中提取和汇总FPS指标
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import argparse


def extract_fps_from_summary(summary_file):
    """从fps_summary.txt提取FPS数据"""
    if not os.path.exists(summary_file):
        return None
    
    fps_data = {}
    try:
        with open(summary_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Average FPS:'):
                    fps_data['avg_fps'] = float(line.split(':')[1].strip())
                elif line.startswith('Max FPS:'):
                    fps_data['max_fps'] = float(line.split(':')[1].strip())
                elif line.startswith('Min FPS:'):
                    fps_data['min_fps'] = float(line.split(':')[1].strip())
                elif line.startswith('Total Frames:'):
                    fps_data['total_frames'] = int(line.split(':')[1].strip())
                elif line.startswith('Total Time:'):
                    fps_data['total_time'] = float(line.split(':')[1].strip().rstrip('s'))
    except Exception as e:
        print(f"Warning: Failed to parse {summary_file}: {e}")
        return None
    
    return fps_data if fps_data else None


def scan_vot_workspace(workspace_path):
    """扫描VOT工作区的results目录，提取所有模型的FPS数据"""
    results_dir = os.path.join(workspace_path, 'results')
    
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return {}
    
    fps_results = {}
    
    # 遍历所有tracker目录
    for tracker_name in os.listdir(results_dir):
        tracker_path = os.path.join(results_dir, tracker_name)
        if not os.path.isdir(tracker_path):
            continue
        
        # 查找fps_summary.txt
        fps_summary = os.path.join(tracker_path, 'fps_summary.txt')
        fps_data = extract_fps_from_summary(fps_summary)
        
        if fps_data:
            fps_results[tracker_name] = fps_data
    
    return fps_results


def print_fps_table(fps_results, workspace_name):
    """打印格式化的FPS表格"""
    if not fps_results:
        print(f"\n{workspace_name}: 未找到FPS数据")
        return
    
    print(f"\n{'='*80}")
    print(f"{workspace_name} - FPS 统计")
    print('='*80)
    print(f"{'模型名称':<30} {'平均FPS':>12} {'最大FPS':>12} {'最小FPS':>12} {'总帧数':>10}")
    print('-'*80)
    
    # 按平均FPS降序排列
    sorted_results = sorted(fps_results.items(), key=lambda x: x[1]['avg_fps'], reverse=True)
    
    for tracker_name, fps_data in sorted_results:
        print(f"{tracker_name:<30} {fps_data['avg_fps']:>12.2f} {fps_data['max_fps']:>12.2f} "
              f"{fps_data['min_fps']:>12.2f} {fps_data['total_frames']:>10}")
    
    print('='*80)
    
    # 计算平均值
    avg_fps_all = sum(data['avg_fps'] for data in fps_results.values()) / len(fps_results)
    print(f"所有模型平均FPS: {avg_fps_all:.2f}")
    print('='*80)


def save_fps_json(fps_results, output_file, workspace_name):
    """保存FPS数据为JSON格式"""
    data = {
        'workspace': workspace_name,
        'results': fps_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"\nFPS数据已保存到: {output_file}")


def compare_workspaces(workspace_paths, workspace_names):
    """对比多个工作区的FPS数据"""
    all_results = {}
    
    for workspace_path, workspace_name in zip(workspace_paths, workspace_names):
        fps_results = scan_vot_workspace(workspace_path)
        all_results[workspace_name] = fps_results
        print_fps_table(fps_results, workspace_name)
    
    # 生成对比表格
    if len(all_results) > 1:
        print(f"\n{'='*100}")
        print("跨数据集FPS对比")
        print('='*100)
        
        # 获取所有模型名称
        all_models = set()
        for fps_results in all_results.values():
            all_models.update(fps_results.keys())
        
        # 打印表头
        header = f"{'模型名称':<30}"
        for ws_name in workspace_names:
            header += f" {ws_name:>15}"
        print(header)
        print('-'*100)
        
        # 打印每个模型的FPS
        for model in sorted(all_models):
            row = f"{model:<30}"
            for ws_name in workspace_names:
                if model in all_results[ws_name]:
                    fps = all_results[ws_name][model]['avg_fps']
                    row += f" {fps:>15.2f}"
                else:
                    row += f" {'-':>15}"
            print(row)
        
        print('='*100)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='VOT FPS统计工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 分析VOT2019的FPS数据
  python vot_fps_analysis.py --workspace vot-workspace-rgbd2019
  
  # 分析VOT2022的FPS数据
  python vot_fps_analysis.py --workspace vot-workspace-rgbd2022
  
  # 对比两个数据集的FPS
  python vot_fps_analysis.py --compare vot-workspace-rgbd2019 vot-workspace-rgbd2022
  
  # 保存为JSON
  python vot_fps_analysis.py --workspace vot-workspace-rgbd2019 --output fps_vot2019.json
        '''
    )
    
    parser.add_argument('--workspace', type=str,
                        help='VOT工作区路径')
    parser.add_argument('--compare', nargs='+',
                        help='对比多个工作区的FPS（提供多个工作区路径）')
    parser.add_argument('--output', type=str,
                        help='保存结果为JSON文件')
    parser.add_argument('--workspace-root', type=str, default='.',
                        help='工作区根目录（默认为当前目录）')
    
    args = parser.parse_args()
    
    workspace_root = Path(args.workspace_root)
    
    if args.compare:
        # 对比模式
        workspace_paths = [workspace_root / ws for ws in args.compare]
        workspace_names = [ws.replace('vot-workspace-', 'VOT-').replace('rgbd', 'RGBD') 
                          for ws in args.compare]
        
        all_results = compare_workspaces(workspace_paths, workspace_names)
        
        # 保存对比结果
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            print(f"\n对比结果已保存到: {args.output}")
    
    elif args.workspace:
        # 单个工作区模式
        workspace_path = workspace_root / args.workspace
        workspace_name = args.workspace.replace('vot-workspace-', 'VOT-').replace('rgbd', 'RGBD')
        
        fps_results = scan_vot_workspace(workspace_path)
        print_fps_table(fps_results, workspace_name)
        
        if args.output:
            save_fps_json(fps_results, args.output, workspace_name)
    
    else:
        # 默认：扫描所有VOT工作区
        vot_workspaces = [
            ('vot-workspace-rgbd2019', 'VOT-RGBD2019'),
            ('vot-workspace-rgbd2022', 'VOT-RGBD2022'),
        ]
        
        found_workspaces = []
        for ws_dir, ws_name in vot_workspaces:
            ws_path = workspace_root / ws_dir
            if ws_path.exists():
                found_workspaces.append((ws_path, ws_name))
        
        if not found_workspaces:
            print("未找到任何VOT工作区")
            print("请使用 --workspace 或 --compare 参数指定工作区路径")
            sys.exit(1)
        
        compare_workspaces(
            [ws[0] for ws in found_workspaces],
            [ws[1] for ws in found_workspaces]
        )


if __name__ == '__main__':
    main()
