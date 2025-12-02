# EZKL 安装指南

EZKL需要从GitHub源码编译，这里提供两种安装方法。

## ⚠️ 重要提示

- **预计时间**: 15-30分钟（首次编译）
- **需要**: 稳定的网络连接
- **磁盘空间**: ~2-3GB（包括依赖）

---

## 方法1: 使用自动脚本（推荐）

```bash
cd /Users/lambertlin/Projects/VeriRegime

# 重新运行安装脚本（已修正）
./scripts/setup_ezkl.sh
```

**该脚本现在会**：
1. ✅ 检查并安装Rust
2. ✅ 从GitHub克隆并编译EZKL
3. ✅ 安装Python依赖

---

## 方法2: 手动安装

### Step 1: 确保Rust已安装

```bash
# 检查Rust
rustc --version

# 如未安装，运行
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Step 2: 安装EZKL

#### 选项A: 从GitHub安装（最新版本）

```bash
# 这会下载源码并编译（15-30分钟）
cargo install --git https://github.com/zkonduit/ezkl
```

#### 选项B: 克隆仓库手动编译

```bash
# 克隆EZKL仓库
cd ~/Projects  # 或任何临时目录
git clone https://github.com/zkonduit/ezkl.git
cd ezkl

# 编译并安装
cargo build --release
cargo install --path .
```

#### 选项C: 使用预编译二进制（如果可用）

访问 https://github.com/zkonduit/ezkl/releases 查看是否有适合您系统的预编译版本。

对于Mac (Apple Silicon):
```bash
# 下载最新release（示例，请查看实际版本）
# wget https://github.com/zkonduit/ezkl/releases/download/vX.X.X/ezkl-macos-arm64
# chmod +x ezkl-macos-arm64
# mv ezkl-macos-arm64 ~/.cargo/bin/ezkl
```

### Step 3: 验证安装

```bash
# 重新加载shell配置
source $HOME/.cargo/env

# 验证EZKL
ezkl --version

# 应该看到类似输出：
# ezkl 0.x.x
```

### Step 4: 安装Python依赖

```bash
conda activate ml
pip install onnx onnxruntime
```

---

## 🐛 常见问题

### Q1: 编译时间过长

**A**: 这是正常的。EZKL是用Rust编写的复杂项目，包含密码学库和零知识证明系统。首次编译需要：
- 下载所有依赖
- 编译几百个crate
- 预计15-30分钟（取决于CPU性能）

**建议**: 让它在后台运行，可以在编译时做其他事情。

### Q2: 编译失败 - 内存不足

**A**: EZKL编译需要较多内存（建议8GB+）。

**解决方案**：
```bash
# 限制并行编译数量
cargo install --git https://github.com/zkonduit/ezkl --jobs 2
```

### Q3: 网络问题 - 下载依赖失败

**A**: 使用中国镜像加速：

```bash
# 配置Cargo使用国内镜像
mkdir -p ~/.cargo
cat > ~/.cargo/config.toml << 'EOF'
[source.crates-io]
replace-with = 'ustc'

[source.ustc]
registry = "https://mirrors.ustc.edu.cn/crates.io-index"
EOF

# 然后重试安装
cargo install --git https://github.com/zkonduit/ezkl
```

### Q4: `linker 'cc' not found`

**A**: 需要安装C编译器：

**Mac**:
```bash
xcode-select --install
```

**Linux**:
```bash
sudo apt-get install build-essential
```

### Q5: OpenSSL错误

**A**: 安装OpenSSL开发库：

**Mac**:
```bash
brew install openssl
export OPENSSL_DIR=$(brew --prefix openssl)
```

**Linux**:
```bash
sudo apt-get install libssl-dev pkg-config
```

---

## ✅ 验证安装成功

运行以下命令确认一切正常：

```bash
# 1. 检查EZKL版本
ezkl --version

# 2. 查看帮助信息
ezkl --help

# 3. 测试一个简单命令
ezkl gen-settings --help
```

如果以上命令都正常运行，说明安装成功！✅

---

## 📊 编译进度参考

编译过程中会看到类似输出：

```
   Compiling proc-macro2 v1.0.66
   Compiling unicode-ident v1.0.11
   Compiling quote v1.0.33
   ...
   Compiling halo2_proofs v0.x.x
   Compiling ezkl v0.x.x
   Finished release [optimized] target(s) in 18m 32s
   Installing ~/.cargo/bin/ezkl
   Installed package `ezkl v0.x.x`
```

**关键里程碑**：
- 前5分钟: 下载和编译基础依赖
- 5-15分钟: 编译密码学库（最耗时）
- 15-20分钟: 编译EZKL主体
- 最后几分钟: 链接和安装

---

## 🚀 安装完成后

继续执行zkML转换流程：

1. ✅ EZKL已安装
2. 📝 运行 `notebooks/export_onnx.ipynb` 导出ONNX
3. 🔐 运行 `scripts/zkml_generate_proof.sh` 生成证明

---

## 💡 提示

如果编译时间实在太长（>45分钟），可以考虑：
1. 检查是否有预编译版本
2. 使用更快的机器编译
3. 或者先进行ONNX导出，EZKL可以后台慢慢编译

编译EZKL与导出ONNX是独立的，可以并行进行！

