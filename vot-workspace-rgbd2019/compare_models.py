#!/usr/bin/env python3
"""
VOT模型对比工具
自动收集多个tracker的性能指标并生成对比报告
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelComparator:
    def __init__(self, workspace_dir="."):
        self.workspace_dir = Path(workspace_dir)
        self.results_dir = self.workspace_dir / "results"
        self.analysis_dir = self.workspace_dir / "analysis"
        self.output_dir = self.workspace_dir / "model_comparison"
        self.output_dir.mkdir(exist_ok=True)
        
    def collect_fps_data(self):
        """收集FPS数据"""
        fps_data = {}
        
        for tracker_dir in self.results_dir.iterdir():
            if not tracker_dir.is_dir():
                continue
                
            tracker_name = tracker_dir.name
            fps_summary = tracker_dir / "fps_summary.txt"
            
            if fps_summary.exists():
                with open(fps_summary, 'r') as f:
                    fps_info = {}
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            fps_info[key.strip()] = value.strip()
                    
                    fps_data[tracker_name] = {
                        'avg_fps': float(fps_info.get('Average FPS', '0').split()[0]) if 'Average FPS' in fps_info else 0,
                        'max_fps': float(fps_info.get('Max FPS', '0').split()[0]) if 'Max FPS' in fps_info else 0,
                        'min_fps': float(fps_info.get('Min FPS', '0').split()[0]) if 'Min FPS' in fps_info else 0,
                        'total_frames': int(fps_info.get('Total Frames', '0')) if 'Total Frames' in fps_info else 0,
                    }
        
        return fps_data
    
    def collect_vot_metrics(self):
        """收集VOT指标（从最新的analysis结果）"""
        metrics = {}
        
        # 查找最新的analysis结果
        if not self.analysis_dir.exists():
            return metrics
            
        analysis_dirs = sorted(self.analysis_dir.iterdir(), key=lambda x: x.name, reverse=True)
        
        for analysis_dir in analysis_dirs:
            if not analysis_dir.is_dir():
                continue
                
            # 尝试读取JSON结果
            for json_file in analysis_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        # 解析VOT metrics
                        # 这里需要根据实际的JSON结构调整
                        if data:
                            return data
                except:
                    continue
        
        return metrics
    
    def create_comparison_table(self, fps_data):
        """创建对比表格"""
        if not fps_data:
            print("⚠️  没有找到FPS数据")
            return None
            
        # 创建DataFrame
        df = pd.DataFrame.from_dict(fps_data, orient='index')
        df.index.name = 'Tracker'
        df = df.sort_values('avg_fps', ascending=False)
        
        # 保存为CSV
        csv_path = self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path)
        print(f"✅ 对比表格已保存: {csv_path}")
        
        # 保存为Excel（如果可用）
        try:
            excel_path = self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(excel_path, engine='openpyxl')
            print(f"✅ Excel表格已保存: {excel_path}")
        except ImportError:
            print("ℹ️  安装openpyxl以支持Excel导出: pip install openpyxl")
        
        return df
    
    def plot_fps_comparison(self, df):
        """绘制FPS对比图"""
        if df is None or df.empty:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('模型FPS性能对比', fontsize=16, fontweight='bold')
        
        # 1. 平均FPS柱状图
        ax1 = axes[0, 0]
        df['avg_fps'].plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('平均FPS对比')
        ax1.set_ylabel('FPS')
        ax1.set_xlabel('Tracker')
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱状图上显示数值
        for i, v in enumerate(df['avg_fps']):
            ax1.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
        
        # 2. FPS范围图（min, avg, max）
        ax2 = axes[0, 1]
        x = range(len(df))
        ax2.plot(x, df['avg_fps'], 'o-', label='Average', linewidth=2, markersize=8)
        ax2.fill_between(x, df['min_fps'], df['max_fps'], alpha=0.3, label='Min-Max Range')
        ax2.set_title('FPS范围对比')
        ax2.set_ylabel('FPS')
        ax2.set_xlabel('Tracker')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df.index, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. 总帧数对比
        ax3 = axes[1, 0]
        df['total_frames'].plot(kind='bar', ax=ax3, color='lightgreen', edgecolor='black')
        ax3.set_title('总处理帧数')
        ax3.set_ylabel('帧数')
        ax3.set_xlabel('Tracker')
        ax3.grid(axis='y', alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. FPS热力图
        ax4 = axes[1, 1]
        heatmap_data = df[['min_fps', 'avg_fps', 'max_fps']].T
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4, 
                    cbar_kws={'label': 'FPS'})
        ax4.set_title('FPS热力图')
        ax4.set_xlabel('Tracker')
        ax4.set_ylabel('指标')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = self.output_dir / f"fps_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ 对比图表已保存: {plot_path}")
        
        # 显示图表
        plt.show()
    
    def generate_report(self):
        """生成完整的对比报告"""
        print("="*60)
        print("🔍 开始收集模型性能指标...")
        print("="*60)
        
        # 收集FPS数据
        fps_data = self.collect_fps_data()
        
        if not fps_data:
            print("❌ 未找到任何FPS数据！")
            print("请先运行 VOT 评估: vot evaluate <tracker_name> --force")
            return
        
        print(f"\n✅ 找到 {len(fps_data)} 个tracker的数据:")
        for tracker in fps_data.keys():
            print(f"  - {tracker}")
        
        # 创建对比表格
        print("\n📊 生成对比表格...")
        df = self.create_comparison_table(fps_data)
        
        if df is not None:
            print("\n" + "="*60)
            print("📋 性能对比表格:")
            print("="*60)
            print(df.to_string())
            print("="*60)
        
        # 生成可视化图表
        print("\n📈 生成对比图表...")
        self.plot_fps_comparison(df)
        
        print("\n" + "="*60)
        print("✅ 对比报告生成完成！")
        print(f"📁 输出目录: {self.output_dir.absolute()}")
        print("="*60)


if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.generate_report()
