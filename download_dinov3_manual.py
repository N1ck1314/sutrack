#!/usr/bin/env python
"""
手动下载 DINOv3 ConvNeXt-Tiny 模型文件
使用 modelscope 或直接下载，避免网络问题
"""
import os
import json
import requests
from pathlib import Path

# 目标目录
MODEL_DIR = Path("./pretrained/dinov3_convnext_tiny")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DINOv3 ConvNeXt-Tiny 模型下载工具")
print("=" * 60)
print(f"目标目录: {MODEL_DIR.absolute()}")
print()

# 方案 1: 使用 modelscope（国内镜像）
try:
    print("方案 1: 尝试使用 ModelScope（国内镜像）...")
    from modelscope import snapshot_download
    
    print("  正在从 ModelScope 下载...")
    model_dir = snapshot_download(
        'AI-ModelScope/dinov3-convnext-tiny',
        cache_dir=str(MODEL_DIR.parent)
    )
    print(f"  ✓ 成功下载到: {model_dir}")
    print()
    print("=" * 60)
    print("✓ 下载完成！")
    print(f"模型路径: {model_dir}")
    print()
    print("请修改配置文件使用本地路径：")
    print(f"  PRETRAIN_TYPE: '{model_dir}'")
    print("=" * 60)
    exit(0)
    
except ImportError:
    print("  ⚠ ModelScope 未安装，跳过此方案")
    print("  安装方法: pip install modelscope")
except Exception as e:
    print(f"  ✗ ModelScope 下载失败: {e}")

print()

# 方案 2: 直接下载文件（使用国内 CDN）
print("方案 2: 直接下载模型文件...")

# 创建基础配置文件
config_data = {
    "architectures": ["ConvNextV2ForImageClassification"],
    "hidden_sizes": [96, 192, 384, 768],
    "num_stages": 4,
    "patch_size": 4,
    "num_labels": 1000,
    "model_type": "convnextv2"
}

preprocessor_config = {
    "do_normalize": True,
    "do_resize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "resample": 3,
    "size": {"shortest_edge": 224}
}

print("\n  创建配置文件...")
with open(MODEL_DIR / "config.json", "w") as f:
    json.dump(config_data, f, indent=2)
print(f"  ✓ 创建 config.json")

with open(MODEL_DIR / "preprocessor_config.json", "w") as f:
    json.dump(preprocessor_config, f, indent=2)
print(f"  ✓ 创建 preprocessor_config.json")

print("\n⚠ 模型权重文件 model.safetensors (约 120MB) 需要手动下载")
print("由于网络限制，请通过以下方式之一下载：")
print()
print("选项 A - 使用 wget（如果有代理）:")
print("  export https_proxy=http://your-proxy:port")
print("  wget -O pretrained/dinov3_convnext_tiny/model.safetensors \\")
print("    https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m/resolve/main/model.safetensors")
print()
print("选项 B - 从浏览器下载:")
print("  1. 在能访问外网的环境打开浏览器")
print("  2. 访问: https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m/tree/main")
print("  3. 点击 model.safetensors 下载")
print(f"  4. 复制到: {MODEL_DIR.absolute()}/model.safetensors")
print()
print("选项 C - 使用 HuggingFace CLI:")
print("  pip install huggingface_hub")
print("  export HF_ENDPOINT=https://hf-mirror.com")
print("  huggingface-cli download facebook/dinov3-convnext-tiny-pretrain-lvd1689m \\")
print(f"    --local-dir {MODEL_DIR.absolute()}")
print()

# 检查是否已有权重文件
weight_file = MODEL_DIR / "model.safetensors"
if weight_file.exists():
    print("=" * 60)
    print("✓ 检测到已有模型权重文件！")
    print(f"文件大小: {weight_file.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("模型已准备就绪！修改配置文件使用本地路径：")
    print(f"  PRETRAIN_TYPE: '{MODEL_DIR.absolute()}'")
    print("=" * 60)
else:
    print("=" * 60)
    print("⚠ 等待权重文件下载完成后，再次运行训练")
    print("=" * 60)
