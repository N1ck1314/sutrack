#!/bin/bash
# VOT FPS统计快速启动脚本

cd "$(dirname "$0")"

# 显示使用说明
show_help() {
    cat << EOF
VOT FPS统计工具

用法: ./show_vot_fps.sh [选项]

选项:
    --vot2019          只显示VOT2019的FPS数据
    --vot2022          只显示VOT2022的FPS数据
    --compare          对比两个数据集的FPS（默认）
    --output FILE      保存结果为JSON文件
    -h, --help         显示此帮助信息

示例:
    # 显示所有数据集的FPS对比（默认）
    ./show_vot_fps.sh
    
    # 只显示VOT2019的FPS
    ./show_vot_fps.sh --vot2019
    
    # 对比两个数据集并保存为JSON
    ./show_vot_fps.sh --compare --output fps_comparison.json

EOF
}

# 默认参数
MODE="compare"
OUTPUT=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --vot2019)
            MODE="vot2019"
            shift
            ;;
        --vot2022)
            MODE="vot2022"
            shift
            ;;
        --compare)
            MODE="compare"
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python vot_fps_analysis.py"

case $MODE in
    vot2019)
        CMD="$CMD --workspace vot-workspace-rgbd2019"
        ;;
    vot2022)
        CMD="$CMD --workspace vot-workspace-rgbd2022"
        ;;
    compare)
        CMD="$CMD --compare vot-workspace-rgbd2019 vot-workspace-rgbd2022"
        ;;
esac

if [ -n "$OUTPUT" ]; then
    CMD="$CMD --output $OUTPUT"
fi

echo "=========================================="
echo "VOT FPS 统计分析"
echo "=========================================="
echo ""

# 执行命令
$CMD
