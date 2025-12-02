# Docker + EZKL 安装指南

## 🐳 安装Docker Desktop（Mac）

### Step 1: 下载Docker Desktop

**方法A: 直接下载（推荐）**
```bash
# 打开下载页面
open https://www.docker.com/products/docker-desktop/
```

**方法B: 使用Homebrew**
```bash
# 如果已安装Homebrew
brew install --cask docker
```

### Step 2: 安装并启动

1. 下载 `Docker.dmg` 文件
2. 双击安装
3. 打开Docker Desktop应用
4. 等待Docker启动（菜单栏会显示Docker图标）

### Step 3: 验证安装

```bash
docker --version
# 应该看到: Docker version 24.x.x

docker ps
# 应该看到容器列表（可能为空）
```

---

## 🚀 使用Docker运行EZKL

安装Docker后，运行：

```bash
cd /Users/lambertlin/Projects/VeriRegime
./scripts/zkml_docker.sh
```

或者手动运行：

```bash
# 拉取EZKL镜像
docker pull zkonduit/ezkl:latest

# 测试EZKL
docker run --rm zkonduit/ezkl ezkl --version

# 运行zkML转换
docker run --rm -v $(pwd):/workspace zkonduit/ezkl \
    ezkl gen-settings -M /workspace/results/onnx/student_model.onnx \
    -O /workspace/results/zkml/settings/settings.json
```

---

## ⚠️ 注意事项

1. **Docker Desktop需要启动**
   - 确保菜单栏有Docker图标
   - 图标必须是运行状态（不是停止）

2. **磁盘空间**
   - Docker Desktop需要 ~2-3GB
   - EZKL镜像需要 ~500MB-1GB

3. **性能**
   - Docker版本可能稍慢（虚拟化开销）
   - 但比编译安装简单得多

---

## 🎯 快速开始

```bash
# 1. 安装Docker Desktop（见上方）

# 2. 启动Docker Desktop应用

# 3. 验证
docker --version

# 4. 运行我们的Docker脚本
./scripts/zkml_docker.sh
```

---

## 📚 参考

- Docker Desktop下载: https://www.docker.com/products/docker-desktop/
- EZKL Docker Hub: https://hub.docker.com/r/zkonduit/ezkl

