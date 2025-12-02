#!/bin/bash
# LaTeX编译脚本

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "编译LaTeX文档"
echo "=========================================="

# 检查LaTeX是否安装
if ! command -v pdflatex &> /dev/null; then
    echo "❌ LaTeX未安装"
    echo ""
    echo "请选择安装方式："
    echo ""
    echo "方法1: 使用Homebrew安装BasicTeX（推荐，快速）"
    echo "  brew install --cask basictex"
    echo "  然后运行: eval \"\$(/usr/libexec/path_helper)\""
    echo ""
    echo "方法2: 下载完整MacTeX（约4GB）"
    echo "  open https://www.tug.org/mactex/"
    echo ""
    echo "方法3: 使用在线编译"
    echo "  - Overleaf: https://www.overleaf.com/"
    echo "  - 上传pre.tex文件即可编译"
    echo ""
    exit 1
fi

# 编译LaTeX
echo "正在编译pre.tex..."
pdflatex -interaction=nonstopmode pre.tex

# 再次编译以生成交叉引用
echo "第二次编译（生成交叉引用）..."
pdflatex -interaction=nonstopmode pre.tex

# 清理辅助文件
echo "清理辅助文件..."
rm -f pre.aux pre.log pre.out pre.toc

echo ""
echo "✅ 编译完成！"
echo "   PDF文件: pre.pdf"
echo ""

# 尝试打开PDF
if command -v open &> /dev/null; then
    echo "正在打开PDF..."
    open pre.pdf
fi

