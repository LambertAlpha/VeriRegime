#!/bin/bash
# VeriRegime - EZKL环境配置脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "VeriRegime - EZKL环境配置"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检测操作系统
OS="$(uname -s)"
case "${OS}" in
    Darwin*)    MACHINE=Mac;;
    Linux*)     MACHINE=Linux;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo -e "${GREEN}检测到系统: ${MACHINE}${NC}"

# 1. 安装Rust (如果未安装)
echo ""
echo "=========================================="
echo "1. 检查Rust环境"
echo "=========================================="

if command -v rustc &> /dev/null
then
    RUST_VERSION=$(rustc --version)
    echo -e "${GREEN}✅ Rust已安装: ${RUST_VERSION}${NC}"
else
    echo -e "${YELLOW}⚠️ Rust未安装，正在安装...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    echo -e "${GREEN}✅ Rust安装完成${NC}"
fi

# 2. 安装EZKL
echo ""
echo "=========================================="
echo "2. 安装EZKL"
echo "=========================================="

if command -v ezkl &> /dev/null
then
    EZKL_VERSION=$(ezkl --version 2>&1 || echo "unknown")
    echo -e "${GREEN}✅ EZKL已安装: ${EZKL_VERSION}${NC}"
    echo -e "${YELLOW}如需更新，请运行: cargo install --force --git https://github.com/zkonduit/ezkl${NC}"
else
    echo -e "${YELLOW}正在安装EZKL（这可能需要10-20分钟）...${NC}"
    echo -e "${YELLOW}从GitHub源码编译安装...${NC}"
    
    # 从GitHub安装EZKL
    cargo install --git https://github.com/zkonduit/ezkl
    
    echo -e "${GREEN}✅ EZKL安装完成${NC}"
fi

# 3. 验证安装
echo ""
echo "=========================================="
echo "3. 验证安装"
echo "=========================================="

if command -v ezkl &> /dev/null
then
    echo -e "${GREEN}✅ EZKL可用${NC}"
    ezkl --version
else
    echo -e "${RED}❌ EZKL安装失败${NC}"
    exit 1
fi

# 4. 安装Python依赖
echo ""
echo "=========================================="
echo "4. 安装Python依赖"
echo "=========================================="

if conda info --envs | grep -q "ml"; then
    echo -e "${GREEN}检测到ml环境${NC}"
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate ml
    
    echo "安装ONNX和onnxruntime..."
    pip install onnx onnxruntime
    
    echo -e "${GREEN}✅ Python依赖安装完成${NC}"
else
    echo -e "${YELLOW}⚠️ 未找到ml环境，请手动安装: pip install onnx onnxruntime${NC}"
fi

# 5. 创建必要目录
echo ""
echo "=========================================="
echo "5. 创建输出目录"
echo "=========================================="

cd "$(dirname "$0")/.."  # 回到项目根目录

mkdir -p results/onnx
mkdir -p results/zkml/compiled
mkdir -p results/zkml/proof
mkdir -p results/zkml/settings

echo -e "${GREEN}✅ 目录创建完成${NC}"

# 完成
echo ""
echo "=========================================="
echo "🎉 EZKL环境配置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 运行 notebooks/export_onnx.ipynb 导出ONNX模型"
echo "  2. 运行 notebooks/zkml_pipeline.ipynb 生成ZK证明"
echo ""
echo "验证安装："
echo "  ezkl --version"
echo ""

