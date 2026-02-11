#!/bin/bash
# VOT评测快速启动脚本

# 设置工作目录
cd "$(dirname "$0")"

# 设置Python路径
export PYTHONPATH=$(pwd):$PYTHONPATH

# 激活conda环境（如果需要）
# source /home/nick/anaconda3/bin/activate sutrack

# 默认参数
LOG_DIR="./vot_test_logs"

# 显示使用说明
show_help() {
    cat << EOF
VOT-RGBD 自动评测脚本

用法: ./run_vot_evaluation.sh [选项]

选项:
    --all              测试所有模型（包括基础模型）
    --improved         测试所有改进模型（排除基础模型）
    --mlka             只测试MLKA模型
    --sgla             只测试SGLA模型
    --or               只测试OR模型
    --smfa             只测试SMFA模型
    --mfe              只测试MFE模型
    --rmt              只测试RMT模型
    --sparsevit        只测试SparseViT模型
    --s4f              只测试S4F模型
    
    --vot2019          只在VOT2019数据集上测试
    --vot2022          只在VOT2022数据集上测试
    --both             在两个数据集上测试（默认）
    
    --log-dir DIR      指定日志目录（默认: ./vot_test_logs）
    -h, --help         显示此帮助信息

示例:
    # 测试所有改进模型在两个数据集上
    ./run_vot_evaluation.sh --improved
    
    # 只测试MLKA和SGLA在VOT2019上
    ./run_vot_evaluation.sh --mlka --sgla --vot2019
    
    # 测试所有模型（包括基础模型）
    ./run_vot_evaluation.sh --all
    
    # 只测试VOT2022数据集
    ./run_vot_evaluation.sh --improved --vot2022
    
    # 测试S4F和SparseViT模型
    ./run_vot_evaluation.sh --s4f --sparsevit

EOF
}

# 解析参数
MODELS=""
DATASETS="vot2019 vot2022"
MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            MODE="--all"
            shift
            ;;
        --improved)
            MODE="--improved"
            shift
            ;;
        --mlka)
            MODELS="$MODELS mlka"
            shift
            ;;
        --sgla)
            MODELS="$MODELS sgla"
            shift
            ;;
        --or)
            MODELS="$MODELS or"
            shift
            ;;
        --smfa)
            MODELS="$MODELS smfa"
            shift
            ;;
        --mfe)
            MODELS="$MODELS mfe"
            shift
            ;;
        --rmt)
            MODELS="$MODELS rmt"
            shift
            ;;
        --sparsevit)
            MODELS="$MODELS sparsevit"
            shift
            ;;
        --s4f)
            MODELS="$MODELS s4f"
            shift
            ;;
        --vot2019)
            DATASETS="vot2019"
            shift
            ;;
        --vot2022)
            DATASETS="vot2022"
            shift
            ;;
        --both)
            DATASETS="vot2019 vot2022"
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
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

# 构建Python命令
CMD="python run_vot_evaluation.py"

if [ -n "$MODE" ]; then
    CMD="$CMD $MODE"
elif [ -n "$MODELS" ]; then
    CMD="$CMD --models $MODELS"
else
    # 默认测试改进模型
    CMD="$CMD --improved"
fi

CMD="$CMD --datasets $DATASETS"
CMD="$CMD --log_dir $LOG_DIR"

# 显示即将执行的命令
echo "=========================================="
echo "VOT-RGBD 自动评测"
echo "=========================================="
echo "命令: $CMD"
echo "数据集: $DATASETS"
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""

# 执行命令
$CMD
