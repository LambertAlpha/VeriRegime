# VeriRegime

**通过知识蒸馏将高性能CNN优化为zkML友好的MLP用于加密货币波动率预测**

[![Python 3.12](https://img.shields.io/badge/python-3.12.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)](https://pytorch.org/)
[![MPS](https://img.shields.io/badge/GPU-MPS-green.svg)](https://developer.apple.com/metal/)

---

## 📋 项目简介

VeriRegime研究如何将高性能CNN模型蒸馏为zkML友好的MLP模型，用于**4小时波动率预测**和**DeFi风险管理**。

### 任务定义

**预测目标**: 未来4小时的BTC波动率水平（低/高）  
**波动率阈值**: 0.05% (平衡分布，训练稳定)  
**标签分布**: 低波动 ~50%, 高波动 ~50%

### 研究动机

DeFi协议（借贷、期权、AMM）需要动态调整风险参数，4小时波动率预测可以：
- **借贷协议**: 高波动时提高抵押率，降低清算风险
- **期权定价**: 波动率是Black-Scholes模型的核心输入
- **AMM做市**: 高波动时提高交易手续费保护LP

**zkML优势**：
✅ 链上可验证的AI推理  
✅ 去中心化风险评估  
✅ 4小时更新周期适合DeFi

---

## 🏗️ 项目结构

```
VeriRegime/
├── train/                  ⭐ 模块化训练组件
│   ├── models.py          # CNN Teacher + MLP Student
│   ├── dataset.py         # 数据加载器
│   ├── trainer.py         # CNN训练器
│   └── distillation.py    # 知识蒸馏训练器
│
├── notebooks/              ⭐ 训练notebooks
│   ├── train_volatility.ipynb     # 4h波动率CNN训练
│   ├── train_distillation.ipynb  # 知识蒸馏MLP训练
│   └── export_onnx.ipynb          # ONNX模型导出
│
├── scripts/                ⭐ 自动化脚本
│   ├── setup_ezkl.sh              # EZKL环境安装
│   ├── zkml_generate_proof.sh    # ZK证明生成
│   └── relabel_volatility.py     # 数据标注
│
├── results/                ⭐ 训练结果
│   ├── checkpoints/       # 模型checkpoint
│   ├── onnx/              # ONNX导出模型
│   └── zkml/              # zkML证明文件
│   ├── figures/           # 可视化图表
│   └── logs/              # 训练日志
│
├── scripts/                # 数据处理脚本
│   ├── data_collection.py      # 数据收集
│   ├── relabel_volatility.py   # 4h波动率标注
│   ├── data_split.py           # 数据分割
│   └── feature_engineering_v2.py
│
└── data/                   # 数据文件
    ├── btc_usdt_1m_volatility_4h.csv  # 完整标注数据
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 使用已有的conda ml环境
conda activate ml

# 如需安装额外依赖
pip install -r requirements.txt
```

### 2. 训练CNN Teacher（波动率预测）

```bash
# 确保在ml环境中
conda activate ml

# 启动Jupyter
jupyter lab

# 打开 notebooks/train_volatility.ipynb
# 按顺序执行所有cell
```

**预期结果**：
- 准确率: ~74% ✅ (目标65%)
- F1分数: ~0.74 ✅ (目标0.63)
- 模型保存: `results/checkpoints/best_model.pth`

### 3. 知识蒸馏MLP Student（zkML优化）

```bash
# 继续在ml环境中
# 打开 notebooks/train_distillation.ipynb
# 按顺序执行所有cell
```

**预期结果**：
- Student准确率: ≥63% (Teacher的85%)
- 理想准确率: ≥67% (Teacher的90%)
- 参数压缩: ~85% (36k → 5k)
- 模型保存: `results/checkpoints/best_student.pth`

### 4. zkML转换 🔐

#### 4.1 安装EZKL环境

```bash
# 安装EZKL和依赖（首次运行）
./scripts/setup_ezkl.sh
```

预计时间：5-10分钟

#### 4.2 导出ONNX模型

```bash
# 打开 notebooks/export_onnx.ipynb
# 执行所有cell，导出ONNX模型
```

输出：`results/onnx/student_model.onnx`

#### 4.3 生成ZK证明

```bash
# 方法1: 使用脚本（推荐）
./scripts/zkml_generate_proof.sh

# 方法2: 手动逐步执行
# 详见 ZKML_GUIDE.md
```

**预期性能**：
- 证明生成时间：5-10秒
- 验证时间：50-200ms
- 证明大小：~128-256KB
- 预估Gas成本：~300-600K

**详细指南**：参见 `ZKML_GUIDE.md`

### 5. 关键配置

在notebook中可调整：

```python
# 训练配置
SEQ_LENGTH = 240  # 4小时窗口
BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-3
DROPOUT = 0.3

# zkML配置
ONNX_OPSET = 14  # EZKL推荐
INPUT_SCALE = 7  # 量化精度
PARAM_SCALE = 7
```

---

## 📊 数据统计

- **总样本**: 973,351
- **训练集**: 681,345 (70%)
  - 低波动: 366,095 (53.7%)
  - 高波动: 315,250 (46.3%)
- **验证集**: 146,003 (15%)
  - 低波动: 62,540 (42.8%)
  - 高波动: 83,463 (57.2%)
- **测试集**: 146,003 (15%)
  - 低波动: 59,604 (40.8%)
  - 高波动: 86,399 (59.2%)
- **波动率阈值**: 0.05%
- **特征**: 7个技术指标（EMA, RSI, MACD, VolumeMA）
- **时间窗口**: 240分钟（4小时）

---

## 🎯 性能目标

| 指标 | 最低目标 | 预期目标 | 理想目标 |
|------|---------|---------|---------|
| CNN准确率 | 60% | 65% | 70% |
| CNN F1-score | 0.58 | 0.63 | 0.68 |
| MLP准确率保留 | 80% | 85% | 90% |
| 证明加速 | 5× | 10× | 20-50× |
| 准确率下降 | <8% | <5% | <3% |

**注**: 平衡数据集（50:50），随机猜测baseline是50%。

---

## 📈 研究路线图

```
Week 10: CNN Baseline训练                    ← 当前阶段
  └─ 目标: 准确率 ≥ 90%

Week 11: MLP蒸馏
  └─ 目标: 保留85%+ CNN性能

Week 12: zkML优化
  ├─ 多项式激活（减少50%约束）
  └─ 自适应量化（再减少40%约束）

Week 13: EZKL编译
  └─ 目标: 10-50×证明加速

Week 14-15: 分析与报告
```

---

## 🔧 自定义训练

### 调整超参数

在notebook中修改配置cell：

```python
EPOCHS = 100        # 增加训练轮数
LR = 5e-4          # 调整学习率
BATCH_SIZE = 256   # 修改批次大小
```

### 修改模型架构

编辑 `train/models.py`：

```python
class CNNVolatility(nn.Module):
    def __init__(...):
        # 修改卷积层配置
        self.conv1 = nn.Conv1d(..., kernel_size=7)  # 改kernel size
        # 添加更多层
```

### 添加新特征

1. 在 `scripts/feature_engineering_v2.py` 中计算新特征
2. 在notebook中更新 `FEATURE_COLS`
3. 调整模型 `input_channels` 参数

---

## 🔬 处理类别不平衡

当前数据是90:10的不平衡分布，训练器已内置：

1. **类别权重**: 自动计算并应用到损失函数
2. **Focal Loss**: 可在 `train/trainer.py` 中启用
3. **评估指标**: 使用F1-score而非准确率作为主指标

查看混淆矩阵重点关注：
- **高波动召回率**: 模型识别到多少真实高波动
- **高波动精确率**: 预测为高波动时有多少是对的

---

## 📝 使用说明

### 查看结果

- **训练日志**: `results/logs/`
- **模型checkpoint**: `results/checkpoints/best_model.pth`
- **可视化图表**: `results/figures/`

### 重新标注数据

```bash
# 如需修改阈值，编辑 scripts/relabel_volatility.py
# VOLATILITY_THRESHOLD = 0.001  # 当前0.1%

python scripts/relabel_volatility.py
python scripts/data_split.py
```

---

## 🤝 贡献

本项目为香港中文大学（深圳）DDA4220深度学习课程项目。

**学生**: Lin Boyi (Lambert)  
**学期**: Fall 2024

---

## 📄 License

MIT License

---

**最后更新**: 2024-12-01  
**版本**: 2.0 (4小时波动率版本)
