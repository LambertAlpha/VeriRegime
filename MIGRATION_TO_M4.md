# VeriRegime 迁移指南: MacBook Air M3 → Mac Mini M4

本文档记录了如何将VeriRegime项目从MacBook Air M3迁移到Mac Mini M4。

## 迁移概述

- **源设备**: MacBook Air M3
- **目标设备**: Mac Mini M4 (顶配)
- **迁移方法**: Git + GitHub
- **优化策略**: 利用M4更大内存和更强GPU

---

## 步骤1: 在MacBook Air M3上准备Git仓库

### 1.1 初始化Git仓库

```bash
cd /Users/lambertlin/Courses/VeriRegime

# 初始化Git
git init

# 添加所有文件
git add .

# 创建初始提交
git commit -m "Initial commit: VeriRegime project

- CNN Teacher baseline implementation
- Data collection and preprocessing scripts
- Training pipeline with early stopping
- Midterm report (LaTeX)
- Trained model checkpoint (cnn_teacher_best.pth)

Dataset: 974,907 BTC/USDT 1-min samples (2022-2024)
Current status: Week 10 - CNN baseline training"
```

### 1.2 推送到GitHub

```bash
# 创建GitHub仓库 (在GitHub网站上创建)
# 仓库名: VeriRegime
# 可见性: Private (推荐) 或 Public

# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/VeriRegime.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

**注意**:
- 数据文件(data/*.csv)总共约200MB,Git可以处理
- 如果超过100MB,考虑使用Git LFS: `git lfs track "data/*.csv"`

---

## 步骤2: 在Mac Mini M4上克隆仓库

### 2.1 克隆项目

```bash
# 在Mac Mini M4上打开终端
cd ~/Courses  # 或你喜欢的位置

# 克隆仓库
git clone https://github.com/YOUR_USERNAME/VeriRegime.git
cd VeriRegime
```

### 2.2 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 2.3 验证数据和模型

```bash
# 检查数据文件
ls -lh data/
# 应该看到: train.csv, val.csv, test.csv

# 检查模型文件
ls -lh models/
# 应该看到: cnn_teacher_best.pth

# 验证数据完整性
python3 << EOF
import pandas as pd
train = pd.read_csv('data/train.csv', index_col=0)
val = pd.read_csv('data/val.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)
print(f"✅ Train: {len(train):,} samples")
print(f"✅ Val: {len(val):,} samples")
print(f"✅ Test: {len(test):,} samples")
EOF
```

---

## 步骤3: 在Mac Mini M4上运行优化训练

### 3.1 使用M4优化版训练脚本

VeriRegime包含针对Mac Mini M4优化的训练脚本:

**文件**: `src/train_cnn_m4.py`

**主要优化**:
- Batch size: 256 → 512 (利用M4更大内存)
- Num workers: 4 → 8 (利用M4更多CPU核心)
- Epochs: 50 → 100 (更充分训练)
- Early stopping patience: 10 → 15
- 支持Apple Silicon MPS加速
- 可选混合精度训练

### 3.2 开始训练

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行M4优化版训练
python src/train_cnn_m4.py

# 如果想后台运行
nohup python src/train_cnn_m4.py > training_m4.log 2>&1 &

# 查看实时日志
tail -f training_m4.log
```

### 3.3 预期性能提升

**MacBook Air M3** (当前配置):
- Batch size: 256
- Epoch时间: ~10-15分钟
- 总训练时间: ~8-12小时 (50 epochs)

**Mac Mini M4** (优化配置):
- Batch size: 512
- 预计Epoch时间: ~6-8分钟 (提升40-50%)
- 预计总训练时间: ~10-13小时 (100 epochs,但单epoch更快)
- 更稳定的性能(主动散热,无降频)

---

## 步骤4: 训练结果对比

### 4.1 M4训练产出

训练完成后,Mac Mini M4版本会生成:

- `models/cnn_teacher_best_m4.pth` - M4训练的最佳模型
- `results/figures/cnn_confusion_matrix_m4.png` - 混淆矩阵
- `results/figures/cnn_training_curves_m4.png` - 训练曲线
- `results/logs/cnn_training_history_m4.json` - 训练历史

### 4.2 性能对比

```bash
# 在Mac Mini M4上创建对比脚本
python3 << 'EOF'
import json
import pandas as pd

# 读取M3训练历史
try:
    with open('results/logs/cnn_training_history.json') as f:
        m3_history = json.load(f)
    print("M3 (Air) 最佳结果:")
    print(f"  Val Acc: {max(m3_history['val_acc']):.2f}%")
    print(f"  Val F1: {max(m3_history['val_f1']):.4f}")
except:
    print("M3训练历史未找到")

# 读取M4训练历史
try:
    with open('results/logs/cnn_training_history_m4.json') as f:
        m4_history = json.load(f)
    print("\nM4 (Mini) 最佳结果:")
    print(f"  Val Acc: {max(m4_history['val_acc']):.2f}%")
    print(f"  Val F1: {max(m4_history['val_f1']):.4f}")
except:
    print("M4训练历史未找到")
EOF
```

---

## 步骤5: 提交M4训练结果

### 5.1 Git提交

```bash
# 添加M4训练产出
git add models/cnn_teacher_best_m4.pth
git add results/figures/*_m4.png
git add results/logs/*_m4.json

# 提交
git commit -m "Add Mac Mini M4 training results

- Trained with optimized config (batch_size=512, num_workers=8)
- Achieved XX% validation accuracy (XX% F1 score)
- Training time: XX hours on M4
- Performance improvement over M3: +XX%"

# 推送到GitHub
git push origin main
```

### 5.2 在MacBook Air M3上同步

如果你之后想在M3上查看M4的结果:

```bash
# 在MacBook Air M3上
cd /Users/lambertlin/Courses/VeriRegime
git pull origin main
```

---

## 故障排除

### 问题1: MPS设备错误

如果遇到 `RuntimeError: MPS backend not available`:

```bash
# 降级到CPU训练
# 在 train_cnn_m4.py 中修改:
device = 'cpu'  # 强制使用CPU
```

### 问题2: 内存不足

如果512 batch size太大:

```bash
# 减小batch size
# 在 train_cnn_m4.py 的 create_dataloaders 中:
batch_size=384,  # 尝试384或256
```

### 问题3: 数据加载慢

如果num_workers=8导致问题:

```bash
# 减少workers
num_workers=4,  # 或尝试6
```

---

## 性能基准参考

### MacBook Air M3
- **CPU**: 8核 (4性能+4能效)
- **GPU**: 8-10核
- **内存**: 8-16GB统一内存
- **散热**: 被动散热(无风扇)
- **限制**: 长时间负载会降频

### Mac Mini M4 (顶配)
- **CPU**: 10核 (4性能+6能效)
- **GPU**: 10核
- **内存**: 最高32GB统一内存
- **散热**: 主动散热(风扇)
- **优势**: 持续高性能,无降频

**预期加速比**: 1.5-2× (考虑batch size增加和无降频)

---

## 下一步工作

在Mac Mini M4上完成CNN baseline后:

1. **Week 11**: 实现MLP Student架构
2. **Week 11**: 实现知识蒸馏(Experiment 1)
3. **Week 12**: 多项式激活优化(Experiment 2)
4. **Week 12**: 自适应量化(Experiment 3)
5. **Week 13**: EZKL编译与部署(Experiment 4)

所有这些实验都可以在M4上更快完成!

---

**最后更新**: 2024-11-08
**作者**: Lin Boyi
**课程**: DDA4220 Deep Learning
