#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
CLEAN_BEFORE=false
WATCH_MODE=false

# 解析参数
TEX_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN_BEFORE=true
            shift
            ;;
        -w|--watch)
            WATCH_MODE=true
            shift
            ;;
        *.tex)
            TEX_FILE="$1"
            shift
            ;;
        *)
            TEX_FILE="$1"
            shift
            ;;
    esac
done

# 如果没有指定文件，查找当前目录中的 .tex 文件
if [ -z "$TEX_FILE" ]; then
    TEX_FILES=($(find "$SCRIPT_DIR" -maxdepth 1 -name "*.tex" -type f))
    
    if [ ${#TEX_FILES[@]} -eq 0 ]; then
        echo "错误: 未找到 .tex 文件"
        exit 1
    elif [ ${#TEX_FILES[@]} -eq 1 ]; then
        TEX_FILE="${TEX_FILES[0]}"
        TEX_FILE=$(basename "$TEX_FILE")
    else
        echo "找到多个 .tex 文件："
        for i in "${!TEX_FILES[@]}"; do
            echo "  $((i+1)). $(basename "${TEX_FILES[$i]}")"
        done
        echo ""
        read -p "请输入文件编号或文件名: " choice
        
        if [[ "$choice" =~ ^[0-9]+$ ]]; then
            TEX_FILE=$(basename "${TEX_FILES[$((choice-1))]}")
        else
            TEX_FILE="$choice"
        fi
    fi
fi

# 检查文件是否存在
if [ ! -f "$SCRIPT_DIR/$TEX_FILE" ]; then
    echo "错误: 文件不存在: $TEX_FILE"
    exit 1
fi

# 创建 build 目录
mkdir -p "$BUILD_DIR"

# 清理 build 目录（如果需要）
if [ "$CLEAN_BEFORE" = true ]; then
    echo "清理 build 目录..."
    rm -rf "$BUILD_DIR"/*
fi

echo "========================================="
echo "编译 LaTeX 文档: $TEX_FILE"
echo "输出目录: $BUILD_DIR"
echo "========================================="
echo ""

# 切换到文档目录
cd "$SCRIPT_DIR"

# 获取不带扩展名的文件名
BASE_NAME="${TEX_FILE%.tex}"

if [ "$WATCH_MODE" = true ]; then
    # 监视模式：使用 latexmk -pvc
    echo "启动监视模式（按 Ctrl+C 退出）..."
    echo ""
    
    if command -v latexmk &> /dev/null; then
        latexmk -pvc -pdf -interaction=nonstopmode \
            -output-directory="$BUILD_DIR" \
            "$TEX_FILE"
    else
        echo "错误: 未找到 latexmk，请先安装"
        echo "macOS: brew install basictex"
        exit 1
    fi
else
    # 普通编译模式
    if command -v latexmk &> /dev/null; then
        # 使用 latexmk（推荐）
        echo "使用 latexmk 编译..."
        latexmk -pdf -interaction=nonstopmode \
            -output-directory="$BUILD_DIR" \
            "$TEX_FILE"
        
        # 将 PDF 复制回源目录
        if [ -f "$BUILD_DIR/$BASE_NAME.pdf" ]; then
            cp "$BUILD_DIR/$BASE_NAME.pdf" "$SCRIPT_DIR/"
            echo ""
            echo "✓ PDF 已复制到源目录: $BASE_NAME.pdf"
        fi
    elif command -v pdflatex &> /dev/null; then
        # 使用 pdflatex
        echo "使用 pdflatex 编译..."
        pdflatex -interaction=nonstopmode \
            -output-directory="$BUILD_DIR" \
            "$TEX_FILE"
        
        # 将 PDF 复制回源目录
        if [ -f "$BUILD_DIR/$BASE_NAME.pdf" ]; then
            cp "$BUILD_DIR/$BASE_NAME.pdf" "$SCRIPT_DIR/"
            echo ""
            echo "✓ PDF 已复制到源目录: $BASE_NAME.pdf"
        fi
    else
        echo "错误: 未找到 pdflatex 或 latexmk"
        exit 1
    fi
    
    echo ""
    echo "========================================="
    echo "编译完成！"
    echo "输出文件位置: $BUILD_DIR/"
    echo "PDF 位置: $SCRIPT_DIR/$BASE_NAME.pdf"
    echo "========================================="
fi
