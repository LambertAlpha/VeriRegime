#!/bin/bash

# 清理 docs 目录中 LaTeX 编译产生的输出文件
# 使用方法: ./scripts/clean_docs.sh [--dry-run] [--backup]

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"
BACKUP_DIR="$PROJECT_ROOT/docs/.latex_output_backup"

# 解析参数
DRY_RUN=false
BACKUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --backup)
            BACKUP=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法: $0 [--dry-run] [--backup]"
            exit 1
            ;;
    esac
done

# LaTeX 输出文件扩展名列表
LATEX_OUTPUT_EXTENSIONS=(
    "*.aux"
    "*.fdb_latexmk"
    "*.fls"
    "*.out"
    "*.synctex.gz"
    "*.bak"
    "*.bbl"
    "*.blg"
    "*.lof"
    "*.lot"
    "*.toc"
    "*.log"
    "*.pdf"
)

echo "========================================="
echo "清理 docs 目录中的 LaTeX 输出文件"
echo "========================================="
echo "文档目录: $DOCS_DIR"
if [ "$DRY_RUN" = true ]; then
    echo "模式: 预览模式（不会实际删除文件）"
fi
if [ "$BACKUP" = true ]; then
    echo "备份目录: $BACKUP_DIR"
fi
echo ""

# 切换到 docs 目录
cd "$DOCS_DIR"

BUILD_DIR="$DOCS_DIR/build"

# 收集所有需要清理的文件（优先收集 build 目录中的文件）
FILES_TO_CLEAN=()

# 首先收集 build 目录中的文件（新配置的输出目录）
if [ -d "$BUILD_DIR" ]; then
    for ext in "${LATEX_OUTPUT_EXTENSIONS[@]}"; do
        while IFS= read -r -d '' file; do
            FILES_TO_CLEAN+=("$file")
        done < <(find "$BUILD_DIR" -type f -name "$ext" -print0 2>/dev/null)
    done
fi

# 然后收集 docs 目录中遗留的文件（排除 build 目录和 PDF 文件）
for ext in "${LATEX_OUTPUT_EXTENSIONS[@]}"; do
    # 排除 PDF，因为我们想保留它们
    if [ "$ext" = "*.pdf" ]; then
        continue
    fi
    while IFS= read -r -d '' file; do
        # 排除 build 目录和 PDF 文件
        if [[ "$file" != *"/build/"* ]] && [[ "$file" != *.pdf ]]; then
            FILES_TO_CLEAN+=("$file")
        fi
    done < <(find . -maxdepth 1 -type f -name "$ext" -print0 2>/dev/null)
done

if [ ${#FILES_TO_CLEAN[@]} -eq 0 ]; then
    echo "✓ 没有找到需要清理的文件"
    exit 0
fi

echo "找到 ${#FILES_TO_CLEAN[@]} 个文件需要清理:"
echo ""

# 显示文件列表
for file in "${FILES_TO_CLEAN[@]}"; do
    size=$(du -h "$file" 2>/dev/null | cut -f1)
    echo "  - $file ($size)"
done

echo ""

if [ "$DRY_RUN" = true ]; then
    echo "预览模式：不会实际删除文件"
    echo "运行 './scripts/clean_docs.sh' 来实际清理文件"
    exit 0
fi

# 确认删除
read -p "确定要删除这些文件吗？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 如果需要备份，创建备份目录
if [ "$BACKUP" = true ]; then
    mkdir -p "$BACKUP_DIR"
    BACKUP_SUBDIR="$BACKUP_DIR/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_SUBDIR"
    echo "创建备份目录: $BACKUP_SUBDIR"
fi

# 清理文件
DELETED_COUNT=0
FAILED_COUNT=0

for file in "${FILES_TO_CLEAN[@]}"; do
    if [ "$BACKUP" = true ]; then
        # 备份文件（保持目录结构）
        file_dir=$(dirname "$file")
        if [ "$file_dir" != "." ]; then
            mkdir -p "$BACKUP_SUBDIR/$file_dir"
        fi
        cp "$file" "$BACKUP_SUBDIR/$file" 2>/dev/null || true
    fi
    
    # 删除文件
    if rm "$file" 2>/dev/null; then
        DELETED_COUNT=$((DELETED_COUNT + 1))
    else
        echo "警告: 无法删除 $file"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "========================================="
echo "清理完成！"
echo "  删除: $DELETED_COUNT 个文件"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "  失败: $FAILED_COUNT 个文件"
fi
if [ "$BACKUP" = true ]; then
    echo "  备份位置: $BACKUP_SUBDIR"
fi
echo "========================================="
