#!/bin/bash
# 手动下载 DINOv3 ConvNeXt-Tiny 模型文件

echo "================================"
echo "DINOv3 ConvNeXt-Tiny 模型下载"
echo "================================"

MODEL_DIR="./pretrained/dinov3_convnext_tiny"
mkdir -p "$MODEL_DIR"

echo ""
echo "模型将保存到: $MODEL_DIR"
echo ""

# 使用 huggingface-cli 下载（需要先安装）
if command -v huggingface-cli &> /dev/null; then
    echo "使用 huggingface-cli 下载..."
    huggingface-cli download facebook/dinov3-convnext-tiny-pretrain-lvd1689m \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
else
    echo "huggingface-cli 未安装，尝试使用 wget..."
    
    # 基础 URL
    BASE_URL="https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m/resolve/main"
    
    # 必需文件列表
    FILES=(
        "config.json"
        "preprocessor_config.json"
        "model.safetensors"
    )
    
    for file in "${FILES[@]}"; do
        echo "下载 $file ..."
        wget -c "$BASE_URL/$file" -O "$MODEL_DIR/$file" || {
            echo "❌ 下载 $file 失败"
            echo "请尝试手动从浏览器下载："
            echo "  $BASE_URL/$file"
            exit 1
        }
    done
fi

echo ""
echo "✅ 下载完成！"
echo ""
echo "现在修改配置使用本地模型："
echo "  在 lib/config/sutrack_dinov3/config.py 中："
echo "  cfg.MODEL.ENCODER.PRETRAIN_TYPE = \"$MODEL_DIR\""
echo ""
