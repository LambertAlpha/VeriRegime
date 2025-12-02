#!/bin/bash
# 构建EZKL Docker镜像（从源码）

set -e

echo "=========================================="
echo "构建EZKL Docker镜像"
echo "=========================================="

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# 创建临时目录
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "1. 克隆EZKL仓库..."
git clone --depth 1 https://github.com/zkonduit/ezkl.git
cd ezkl

echo "2. 检查是否有Dockerfile..."
if [ -f "Dockerfile" ]; then
    echo "✅ 找到Dockerfile，开始构建..."
    docker build -t zkonduit/ezkl:local .
else
    echo "⚠️ 未找到Dockerfile，创建自定义Dockerfile..."
    
    # 创建Dockerfile
    cat > Dockerfile << 'DOCKERFILE'
FROM rustlang/rust:nightly

WORKDIR /ezkl

# 安装依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 克隆并构建EZKL
RUN git clone https://github.com/zkonduit/ezkl.git /ezkl && \
    cargo install --path . --locked

ENTRYPOINT ["ezkl"]
DOCKERFILE

    echo "开始构建（这可能需要30-60分钟）..."
    docker build -t zkonduit/ezkl:local .
fi

echo ""
echo "✅ Docker镜像构建完成！"
echo "   镜像名称: zkonduit/ezkl:local"

# 清理
cd "$PROJECT_ROOT"
rm -rf "$TEMP_DIR"

echo ""
echo "现在可以运行: ./scripts/zkml_docker.sh"

