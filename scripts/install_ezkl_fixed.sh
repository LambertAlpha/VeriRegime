#!/bin/bash
# EZKL安装脚本（修复SSL问题）

set -e

echo "=========================================="
echo "VeriRegime - EZKL安装（修复版）"
echo "=========================================="

cd /Users/lambertlin/Projects/VeriRegime

# 1. 确保使用nightly
echo ""
echo "1. 检查Rust版本..."
rustup override set nightly
rustc --version

# 2. 配置Cargo使用git CLI（解决SSL问题）
echo ""
echo "2. 配置Cargo使用git CLI..."
mkdir -p ~/.cargo
cat >> ~/.cargo/config.toml << 'EOF'
[net]
git-fetch-with-cli = true
EOF

# 3. 清理之前的失败缓存
echo ""
echo "3. 清理缓存..."
rm -rf ~/.cargo/git/db/halo2-*
rm -rf ~/.cargo/git/checkouts/halo2-*
rm -rf ~/.cargo/git/db/ezkl-*
rm -rf ~/.cargo/git/checkouts/ezkl-*

# 4. 安装EZKL
echo ""
echo "4. 开始安装EZKL（预计20-30分钟）..."
echo "   这可能需要一段时间，请耐心等待..."
cargo install --git https://github.com/zkonduit/ezkl

# 5. 验证安装
echo ""
echo "5. 验证安装..."
source $HOME/.cargo/env
if ezkl --version; then
    echo ""
    echo "=========================================="
    echo "✅ EZKL安装成功！"
    echo "=========================================="
    ezkl --version
else
    echo ""
    echo "❌ 安装验证失败"
    exit 1
fi

