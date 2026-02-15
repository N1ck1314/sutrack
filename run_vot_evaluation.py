#!/usr/bin/env python3
"""
VOT评测自动化脚本
支持在VOT-RGBD2019和VOT-RGBD2022数据集上批量测试多个模型
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path
import json

# 模型配置列表（按照test_config_example.json的顺序）
AVAILABLE_MODELS = [
    'sutrack_t224',           # 基础模型
    'sutrack_mlka_t224',      # MLKA
    'sutrack_or_t224',        # OR
    'sutrack_sgla_t224',      # SGLA
    'sutrack_smfa_t224',      # SMFA
    'sutrack_rmt_t224',       # RMT
    'sutrack_mfe_t224',       # MFE
    'sutrack_sparsevit_t224', # SparseViT
    'sutrack_S4F_t224',       # S4F
    'sutrack_active_t224',    # Active (动态激活)
    'sutrack_activev1_t224',  # Active V1 (特征增强+RGBD动态深度融合)
]

# VOT工作区配置
VOT_WORKSPACES = {
    'vot2019': 'vot-workspace-rgbd2019',
    'vot2022': 'vot-workspace-rgbd2022',
}

class VOTEvaluator:
    """VOT评测器"""
    
    def __init__(self, workspace_root, log_dir='./vot_test_logs'):
        self.workspace_root = Path(workspace_root)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建主日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.main_log = self.log_dir / f'vot_test_summary_{timestamp}.log'
        
        # 测试结果记录
        self.results = {
            'start_time': timestamp,
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
    
    def log(self, message, level='INFO'):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f'[{timestamp}] [{level}] {message}'
        print(log_message)
        
        with open(self.main_log, 'a') as f:
            f.write(log_message + '\n')
    
    def check_workspace(self, workspace_name):
        """检查VOT工作区是否存在"""
        workspace_path = self.workspace_root / VOT_WORKSPACES[workspace_name]
        
        if not workspace_path.exists():
            self.log(f'工作区不存在: {workspace_path}', 'ERROR')
            return False
        
        config_file = workspace_path / 'config.yaml'
        if not config_file.exists():
            self.log(f'配置文件不存在: {config_file}', 'ERROR')
            return False
        
        return True
    
    def check_model_ini(self, workspace_name, model_name):
        """检查模型的ini文件是否存在"""
        workspace_path = self.workspace_root / VOT_WORKSPACES[workspace_name]
        ini_file = workspace_path / f'{model_name}.ini'
        
        if not ini_file.exists():
            self.log(f'模型配置文件不存在: {ini_file}', 'WARNING')
            return False
        
        return True
    
    def run_evaluation(self, workspace_name, model_name):
        """运行单个模型的VOT评测"""
        workspace_path = self.workspace_root / VOT_WORKSPACES[workspace_name]
        
        test_name = f'{model_name}_{workspace_name}'
        self.log(f'开始测试: {test_name}')
        
        # 检查ini文件
        if not self.check_model_ini(workspace_name, model_name):
            self.log(f'跳过测试 {test_name} (配置文件不存在)', 'SKIP')
            self.results['skipped'] += 1
            self.results['details'].append({
                'model': model_name,
                'dataset': workspace_name,
                'status': 'skipped',
                'message': '配置文件不存在'
            })
            return False
        
        # 创建日志文件
        log_file = self.log_dir / f'{test_name}.log'
        
        # 构建VOT命令
        cmd = ['vot', 'evaluate', model_name]
        
        self.log(f'命令: {" ".join(cmd)}')
        self.log(f'工作目录: {workspace_path}')
        
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as f:
                f.write(f'VOT Evaluation: {test_name}\n')
                f.write(f'Command: {" ".join(cmd)}\n')
                f.write(f'Working Directory: {workspace_path}\n')
                f.write('='*80 + '\n\n')
                
                process = subprocess.Popen(
                    cmd,
                    cwd=str(workspace_path),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                return_code = process.wait()
            
            elapsed_time = time.time() - start_time
            
            if return_code == 0:
                self.log(f'测试成功: {test_name} (耗时: {elapsed_time:.2f}s)', 'SUCCESS')
                self.results['success'] += 1
                self.results['details'].append({
                    'model': model_name,
                    'dataset': workspace_name,
                    'status': 'success',
                    'time': elapsed_time,
                    'log': str(log_file)
                })
                return True
            else:
                self.log(f'测试失败: {test_name} (返回码: {return_code})', 'ERROR')
                self.results['failed'] += 1
                self.results['details'].append({
                    'model': model_name,
                    'dataset': workspace_name,
                    'status': 'failed',
                    'return_code': return_code,
                    'log': str(log_file)
                })
                return False
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.log(f'测试异常: {test_name} - {str(e)}', 'ERROR')
            self.results['failed'] += 1
            self.results['details'].append({
                'model': model_name,
                'dataset': workspace_name,
                'status': 'error',
                'error': str(e),
                'log': str(log_file)
            })
            return False
    
    def run_all_tests(self, models, datasets, continue_on_error=True):
        """批量运行所有测试"""
        self.log('='*80)
        self.log('开始VOT自动评测')
        self.log('='*80)
        
        # 检查工作区
        valid_datasets = []
        for dataset in datasets:
            if self.check_workspace(dataset):
                valid_datasets.append(dataset)
                self.log(f'✓ 工作区就绪: {dataset}')
            else:
                self.log(f'✗ 工作区不可用: {dataset}', 'ERROR')
        
        if not valid_datasets:
            self.log('没有可用的工作区，退出', 'ERROR')
            return False
        
        # 统计总任务数
        total_tasks = len(models) * len(valid_datasets)
        self.results['total'] = total_tasks
        
        self.log(f'总任务数: {total_tasks}')
        self.log(f'模型列表: {", ".join(models)}')
        self.log(f'数据集列表: {", ".join(valid_datasets)}')
        self.log('='*80)
        
        current_task = 0
        
        # 按数据集循环（每个数据集测试完所有模型再换下一个数据集）
        for dataset in valid_datasets:
            self.log(f'\n{"="*80}')
            self.log(f'开始在 {dataset.upper()} 数据集上测试')
            self.log('='*80)
            
            for model in models:
                current_task += 1
                self.log(f'\n进度: [{current_task}/{total_tasks}]')
                
                success = self.run_evaluation(dataset, model)
                
                if not success and not continue_on_error:
                    self.log('遇到错误，停止测试', 'ERROR')
                    self.save_results()
                    return False
        
        self.log('\n' + '='*80)
        self.log('所有测试完成！')
        self.log('='*80)
        self.save_results()
        self.print_summary()
        return True
    
    def save_results(self):
        """保存测试结果"""
        result_file = self.log_dir / 'vot_test_results.json'
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        self.log(f'测试结果已保存到: {result_file}')
    
    def print_summary(self):
        """打印测试摘要"""
        self.log('\n' + '='*80)
        self.log('测试摘要')
        self.log('='*80)
        self.log(f'总任务数: {self.results["total"]}')
        self.log(f'成功: {self.results["success"]}')
        self.log(f'失败: {self.results["failed"]}')
        self.log(f'跳过: {self.results["skipped"]}')
        self.log('='*80)
        
        if self.results['failed'] > 0:
            self.log('\n失败的测试:')
            for detail in self.results['details']:
                if detail['status'] in ['failed', 'error']:
                    self.log(f"  - {detail['model']} on {detail['dataset']}")
                    self.log(f"    日志: {detail['log']}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='VOT-RGBD 自动评测脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 测试所有模型在两个数据集上
  python run_vot_evaluation.py --all
  
  # 只测试MLKA和SGLA在VOT2019上
  python run_vot_evaluation.py --models mlka sgla --datasets vot2019
  
  # 测试所有改进模型
  python run_vot_evaluation.py --improved
  
  # 只测试VOT2022数据集
  python run_vot_evaluation.py --all --datasets vot2022
        '''
    )
    
    parser.add_argument('--models', nargs='+', default=None,
                        help='要测试的模型列表（如: mlka sgla or），留空则使用配置')
    parser.add_argument('--datasets', nargs='+', default=['vot2019', 'vot2022'],
                        choices=['vot2019', 'vot2022'],
                        help='要测试的数据集')
    parser.add_argument('--all', action='store_true',
                        help='测试所有模型')
    parser.add_argument('--improved', action='store_true',
                        help='测试所有改进模型（排除基础模型）')
    parser.add_argument('--log_dir', type=str, default='./vot_test_logs',
                        help='日志目录')
    parser.add_argument('--continue_on_error', action='store_true', default=True,
                        help='遇到错误时继续测试')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 确定要测试的模型
    if args.all:
        models = AVAILABLE_MODELS
    elif args.improved:
        # 排除基础模型
        models = [m for m in AVAILABLE_MODELS if m != 'sutrack_t224']
    elif args.models:
        # 处理用户指定的模型名（支持简写）
        models = []
        for model_short in args.models:
            # 尝试匹配完整名称
            matched = False
            for full_name in AVAILABLE_MODELS:
                if model_short.lower() in full_name.lower():
                    models.append(full_name)
                    matched = True
                    break
            if not matched:
                print(f'警告: 未找到模型 {model_short}')
    else:
        # 默认测试所有改进模型
        models = [m for m in AVAILABLE_MODELS if m != 'sutrack_t224']
    
    if not models:
        print('错误: 没有选择任何模型')
        sys.exit(1)
    
    # 获取SUTrack根目录
    workspace_root = Path(__file__).parent.absolute()
    
    print('='*80)
    print('VOT-RGBD 自动评测脚本')
    print('='*80)
    print(f'工作目录: {workspace_root}')
    print(f'日志目录: {args.log_dir}')
    print(f'数据集: {", ".join(args.datasets)}')
    print(f'模型数量: {len(models)}')
    print('='*80)
    
    # 创建评测器并运行
    evaluator = VOTEvaluator(workspace_root, args.log_dir)
    success = evaluator.run_all_tests(models, args.datasets, args.continue_on_error)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
