#!/bin/bash

# 植物识别API启动脚本

echo "🚀 启动植物识别API v2..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装"
    exit 1
fi

# 检查必要的模型文件
MODEL_DIR="../real_data"
if [ ! -f "$MODEL_DIR/organ_classification_mybest.pth" ]; then
    echo "❌ 缺少器官分类模型: $MODEL_DIR/organ_classification_mybest.pth"
    exit 1
fi

# 检查种类分类模型
for organ in "bark" "flower" "fruit" "leaf"; do
    if [ ! -f "$MODEL_DIR/${organ}_species_model.pth" ]; then
        echo "❌ 缺少${organ}种类分类模型: $MODEL_DIR/${organ}_species_model.pth"
        exit 1
    fi
done

echo "✅ 模型文件检查通过"

# 创建上传目录
mkdir -p uploads

# 启动API服务
echo "🌐 启动FastAPI服务器 (端口: 8000)..."
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

