#!/bin/bash
# SUTrack 快速测试启动脚本

echo "=========================================="
echo "  SUTrack 自动测试脚本"
echo "=========================================="
echo ""

# 设置Python路径
export PYTHONPATH=/home/nick/code/code.sutrack/SUTrack:$PYTHONPATH

# 默认参数
THREADS=1
NUM_GPUS=1
DATASETS="uav uavdt visdrone2018"

# 显示使用说明
show_usage() {
    cat << EOF
使用方法:
  $0 [选项]

选项:
  --all              测试所有模型（默认）
  --mlka             仅测试MLKA模型
  --scsa             仅测试SCSA模型
  --or               仅测试OR模型
  --improved         测试所有改进模型（MLKA, SCSA, SGLA, SMFA, OR等）
  --config <file>    使用配置文件
  --threads <n>      设置线程数（默认4）
  --gpus <n>         设置GPU数量（默认8）
  --help             显示此帮助信息

示例:
  $0 --mlka                    # 测试MLKA模型
  $0 --improved --threads 2    # 测试所有改进模型，使用2线程
  $0 --config test_config.json # 使用配置文件
EOF
}

# 解析命令行参数
MODE="all"
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            MODE="all"
            shift
            ;;
        --mlka)
            MODE="mlka"
            shift
            ;;
        --scsa)
            MODE="scsa"
            shift
            ;;
        --or)
            MODE="or"
            shift
            ;;
        --improved)
            MODE="improved"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

cd /home/nick/code/code.sutrack/SUTrack

# 根据模式执行测试
if [ -n "$CONFIG_FILE" ]; then
    echo "使用配置文件: $CONFIG_FILE"
    python auto_test_models.py --config "$CONFIG_FILE"
elif [ "$MODE" = "all" ]; then
    echo "测试所有模型..."
    python auto_test_models.py --models all --datasets $DATASETS --threads $THREADS --num_gpus $NUM_GPUS
elif [ "$MODE" = "mlka" ]; then
    echo "测试 MLKA 模型..."
    python auto_test_models.py --models MLKA --datasets $DATASETS --threads $THREADS --num_gpus $NUM_GPUS
elif [ "$MODE" = "scsa" ]; then
    echo "测试 SCSA 模型..."
    python auto_test_models.py --models SCSA --datasets $DATASETS --threads $THREADS --num_gpus $NUM_GPUS
elif [ "$MODE" = "or" ]; then
    echo "测试 OR 模型..."
    python auto_test_models.py --models OR --datasets $DATASETS --threads $THREADS --num_gpus $NUM_GPUS
elif [ "$MODE" = "improved" ]; then
    echo "测试所有改进模型..."
    python auto_test_models.py --models MLKA SCSA SGLA SMFA OR MFE --datasets $DATASETS --threads $THREADS --num_gpus $NUM_GPUS
fi

echo ""
echo "=========================================="
echo "  测试完成！"
echo "=========================================="
