#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SUTrack 自动测试脚本
用于自动测试不同改进模型在UAV、UAVDT和VisDrone2018数据集上的性能

使用方法:
    python auto_test_models.py --models all --datasets all --threads 4
    python auto_test_models.py --models MLKA SCSA --datasets uav uavdt --threads 2
    python auto_test_models.py --config config.json
"""

import os
import sys
import argparse
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path


# 定义所有可用的模型和配置
AVAILABLE_MODELS = {
    # 基础模型
    'sutrack': ['sutrack_t224'],
    
    # 改进模型
    'sutrack_ASSA': ['sutrack_assa_t224'],
    'sutrack_CMA': ['sutrack_cma_t224'],
    'sutrack_CPAM': ['sutrack_cpam_t224'],
    'sutrack_DynRes': ['sutrack_dynres_t224'],
    'sutrack_MFE': ['sutrack_mfe_t224'],
    'sutrack_MLKA': ['sutrack_mlka_t224', 'sutrack_mlka_b256'],
    'sutrack_Mamba': ['sutrack_mamba_t224'],
    'sutrack_OR': ['sutrack_or_t224', 'sutrack_or_b224'],
    'sutrack_RMT': ['sutrack_t224'],
    'sutrack_S4F': ['sutrack_t224'],
    'sutrack_SCSA': ['sutrack_scsa_t224', 'sutrack_scsa_b224'],
    'sutrack_SGLA': ['sutrack_sgla_t224', 'sutrack_sgla_b224'],
    'sutrack_SMFA': ['sutrack_smfa_t224', 'sutrack_smfa_b224'],
    'sutrack_STAtten': ['sutrack_statten_t224'],
    'sutrack_SparseViT': ['sutrack_sparsevit_t224'],
    'sutrack_active': ['sutrack_t224'],
    'sutrack_patch': ['sutrack_t224', 'sutrack_b224'],
    'sutrack_rewight': ['sutrack_t224', 'sutrack_b224'],
    'sutrack_scale': ['sutrack_t224', 'sutrack_b224'],
}

# UAV相关数据集
UAV_DATASETS = ['uav', 'uavdt', 'visdrone2018']


class TestConfig:
    """测试配置类"""
    def __init__(self):
        self.models = []
        self.datasets = UAV_DATASETS
        self.threads = 2
        self.num_gpus = 8
        self.log_dir = './test_logs'
        self.skip_existing = True
        self.continue_on_error = True
        
    def from_json(self, json_file):
        """从JSON文件加载配置"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.models = data.get('models', [])
        self.datasets = data.get('datasets', UAV_DATASETS)
        self.threads = data.get('threads', 2)
        self.num_gpus = data.get('num_gpus', 8)
        self.log_dir = data.get('log_dir', './test_logs')
        self.skip_existing = data.get('skip_existing', True)
        self.continue_on_error = data.get('continue_on_error', True)
        
    def to_json(self, json_file):
        """保存配置到JSON文件"""
        data = {
            'models': self.models,
            'datasets': self.datasets,
            'threads': self.threads,
            'num_gpus': self.num_gpus,
            'log_dir': self.log_dir,
            'skip_existing': self.skip_existing,
            'continue_on_error': self.continue_on_error
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)


class AutoTester:
    """自动测试器"""
    def __init__(self, config: TestConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建主日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.main_log_file = self.log_dir / f'test_summary_{timestamp}.log'
        
        self.results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
    def log(self, message, level='INFO'):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f'[{timestamp}] [{level}] {message}'
        print(log_msg)
        
        with open(self.main_log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def check_result_exists(self, tracker_name, config_name, dataset):
        """检查结果是否已存在"""
        # 检查结果目录
        result_dir = Path('test/tracking_results') / tracker_name / config_name / dataset
        if result_dir.exists() and any(result_dir.glob('*.txt')):
            return True
        return False
    
    def run_test(self, tracker_name, config_name, dataset):
        """运行单个测试"""
        test_name = f'{tracker_name}_{config_name}_{dataset}'
        
        # 检查是否跳过已有结果
        if self.config.skip_existing and self.check_result_exists(tracker_name, config_name, dataset):
            self.log(f'跳过测试 {test_name} (结果已存在)', 'SKIP')
            self.results['skipped'] += 1
            self.results['details'].append({
                'tracker': tracker_name,
                'config': config_name,
                'dataset': dataset,
                'status': 'skipped',
                'message': '结果已存在'
            })
            return True
        
        # 构建测试命令
        cmd = [
            'python', 'tracking/test.py',
            tracker_name,
            config_name,
            '--dataset_name', dataset,
            '--threads', str(self.config.threads),
            '--num_gpus', str(self.config.num_gpus)
        ]
        
        self.log(f'开始测试: {test_name}')
        self.log(f'命令: {" ".join(cmd)}')
        
        # 创建日志文件
        log_file = self.log_dir / f'{test_name}.log'
        
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
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
                    'tracker': tracker_name,
                    'config': config_name,
                    'dataset': dataset,
                    'status': 'success',
                    'time': elapsed_time,
                    'log': str(log_file)
                })
                return True
            else:
                self.log(f'测试失败: {test_name} (返回码: {return_code})', 'ERROR')
                self.results['failed'] += 1
                self.results['details'].append({
                    'tracker': tracker_name,
                    'config': config_name,
                    'dataset': dataset,
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
                'tracker': tracker_name,
                'config': config_name,
                'dataset': dataset,
                'status': 'error',
                'error': str(e),
                'log': str(log_file)
            })
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        self.log('=' * 80)
        self.log('开始自动测试')
        self.log('=' * 80)
        
        # 统计总任务数
        total_tasks = 0
        for model_configs in self.config.models:
            total_tasks += len(model_configs['configs']) * len(self.config.datasets)
        
        self.log(f'总任务数: {total_tasks}')
        self.log(f'测试数据集: {", ".join(self.config.datasets)}')
        self.log(f'线程数: {self.config.threads}')
        self.log(f'GPU数: {self.config.num_gpus}')
        self.log('=' * 80)
        
        self.results['total'] = total_tasks
        current_task = 0
        
        # 遍历所有模型和配置
        for model_info in self.config.models:
            tracker_name = model_info['tracker']
            configs = model_info['configs']
            
            self.log(f'\n开始测试模型: {tracker_name}')
            
            for config_name in configs:
                for dataset in self.config.datasets:
                    current_task += 1
                    self.log(f'\n进度: [{current_task}/{total_tasks}]')
                    
                    success = self.run_test(tracker_name, config_name, dataset)
                    
                    if not success and not self.config.continue_on_error:
                        self.log('遇到错误，停止测试', 'ERROR')
                        self.save_results()
                        return False
        
        self.log('\n' + '=' * 80)
        self.log('所有测试完成！')
        self.log('=' * 80)
        self.save_results()
        self.print_summary()
        return True
    
    def save_results(self):
        """保存测试结果"""
        result_file = self.log_dir / 'test_results.json'
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        self.log(f'测试结果已保存到: {result_file}')
    
    def print_summary(self):
        """打印测试摘要"""
        self.log('\n' + '=' * 80)
        self.log('测试摘要')
        self.log('=' * 80)
        self.log(f'总任务数: {self.results["total"]}')
        self.log(f'成功: {self.results["success"]}')
        self.log(f'失败: {self.results["failed"]}')
        self.log(f'跳过: {self.results["skipped"]}')
        self.log('=' * 80)
        
        if self.results['failed'] > 0:
            self.log('\n失败的测试:')
            for detail in self.results['details']:
                if detail['status'] in ['failed', 'error']:
                    self.log(f"  - {detail['tracker']} {detail['config']} on {detail['dataset']}")
                    self.log(f"    日志: {detail['log']}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SUTrack 自动测试脚本')
    
    parser.add_argument('--models', nargs='+', default=['all'],
                        help='要测试的模型列表，可以是: all, MLKA, SCSA, OR等')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='指定配置文件名称，如果不指定则使用模型的所有配置')
    parser.add_argument('--datasets', nargs='+', default=UAV_DATASETS,
                        choices=UAV_DATASETS + ['all'],
                        help='要测试的数据集')
    parser.add_argument('--threads', type=int, default=2,
                        help='测试线程数')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='GPU数量')
    parser.add_argument('--log_dir', type=str, default='./test_logs',
                        help='日志目录')
    parser.add_argument('--config', type=str, default=None,
                        help='从JSON配置文件加载测试配置')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='跳过已有结果的测试')
    parser.add_argument('--no_skip_existing', action='store_false', dest='skip_existing',
                        help='不跳过已有结果，重新测试')
    parser.add_argument('--continue_on_error', action='store_true', default=True,
                        help='遇到错误时继续测试')
    parser.add_argument('--save_config', type=str, default=None,
                        help='保存当前配置到JSON文件')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建配置
    config = TestConfig()
    
    # 如果指定了配置文件，从文件加载
    if args.config:
        print(f'从配置文件加载: {args.config}')
        config.from_json(args.config)
    else:
        # 从命令行参数构建配置
        config.threads = args.threads
        config.num_gpus = args.num_gpus
        config.log_dir = args.log_dir
        config.skip_existing = args.skip_existing
        config.continue_on_error = args.continue_on_error
        
        # 处理数据集
        if 'all' in args.datasets:
            config.datasets = UAV_DATASETS
        else:
            config.datasets = args.datasets
        
        # 处理模型
        config.models = []
        
        if 'all' in args.models:
            # 测试所有模型
            for tracker_name, configs in AVAILABLE_MODELS.items():
                config.models.append({
                    'tracker': tracker_name,
                    'configs': configs if args.configs is None else args.configs
                })
        else:
            # 测试指定模型
            for model_name in args.models:
                # 查找匹配的tracker
                tracker_name = None
                if model_name in AVAILABLE_MODELS:
                    tracker_name = model_name
                else:
                    # 尝试匹配 sutrack_XXX 格式
                    for key in AVAILABLE_MODELS.keys():
                        if model_name.lower() in key.lower() or key.lower().endswith(model_name.lower()):
                            tracker_name = key
                            break
                
                if tracker_name:
                    configs = AVAILABLE_MODELS[tracker_name]
                    if args.configs:
                        # 过滤指定的配置
                        configs = [c for c in configs if c in args.configs]
                    
                    config.models.append({
                        'tracker': tracker_name,
                        'configs': configs
                    })
                else:
                    print(f'警告: 未找到模型 {model_name}')
    
    # 保存配置（如果指定）
    if args.save_config:
        config.to_json(args.save_config)
        print(f'配置已保存到: {args.save_config}')
    
    # 创建测试器并运行
    tester = AutoTester(config)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
