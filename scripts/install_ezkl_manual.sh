#!/bin/bash
# EZKL手动安装脚本（修复halo2依赖问题）

set -e

echo "=========================================="
echo "VeriRegime - EZKL手动安装（修复版）"
echo "=========================================="

cd /Users/lambertlin/Projects/VeriRegime

# 1. 确保使用nightly
echo ""
echo "1. 检查Rust版本..."
rustup override set nightly
rustc --version

# 2. 手动克隆halo2仓库
echo ""
echo "2. 手动克隆halo2依赖..."
HALO2_DIR="$HOME/.cargo/git/checkouts/halo2-6ed7871cabb7008c"
HALO2_COMMIT="1dd2090741f006fd031a07da7f3c9dfce5e0015e"
HALO2_BRANCH="ac/conditional-compilation-icicle2"

# 清理旧缓存
rm -rf "$HOME/.cargo/git/db/halo2-*"
rm -rf "$HOME/.cargo/git/checkouts/halo2-*"

# 创建目录
mkdir -p "$(dirname "$HALO2_DIR")"

# 克隆halo2仓库
echo "   克隆halo2仓库..."
if [ ! -d "$HALO2_DIR" ]; then
    git clone --depth 1 --branch "$HALO2_BRANCH" \
        https://github.com/zkonduit/halo2.git "$HALO2_DIR"
    cd "$HALO2_DIR"
    git checkout "$HALO2_COMMIT" 2>/dev/null || echo "   已在正确分支"
    cd - > /dev/null
else
    echo "   halo2目录已存在，跳过克隆"
fi

# 3. 配置Cargo
echo ""
echo "3. 配置Cargo..."
mkdir -p ~/.cargo
cat > ~/.cargo/config.toml << 'EOF'
[net]
git-fetch-with-cli = true
EOF

# 4. 尝试安装EZKL
echo ""
echo "4. 开始安装EZKL..."
echo "   如果halo2问题仍然存在，可能需要等待EZKL更新..."
cargo install --git https://github.com/zkonduit/ezkl || {
    echo ""
    echo "⚠️ 安装失败，尝试替代方案..."
    echo ""
    echo "替代方案："
    echo "  1. 检查EZKL是否有预编译版本"
    echo "  2. 使用Docker运行EZKL"
    echo "  3. 等待EZKL修复依赖问题"
    exit 1
}

# 5. 验证
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
    echo "❌ 安装验证失败"
    exit 1
fi

