# VeriRegime

**Distilling High-Performance CNNs to zkML-Optimized MLPs for Trading Signal Generation**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 项目简介

VeriRegime是一个深度学习课程项目（DDA4220），研究**如何将高性能CNN模型蒸馏为zkML友好的MLP模型**，用于金融时间序列的交易信号生成。

### 研究动机

零知识机器学习（zkML）允许在区块链上验证ML推理，但面临两大挑战：
1. **高证明成本**: CNN/LSTM等复杂模型生成证明需要数百秒甚至几小时
2. **准确率损失**: 简单MLP模型在zkML中高效，但性能通常较差

本项目通过**知识蒸馏**将CNN的性能迁移到MLP，同时通过**多项式激活**和**自适应量化**进一步优化zkML效率。

### 核心贡献

1. **CNN→MLP知识蒸馏框架**：证明CNN的时序特征提取能力可以被MLP学习（目标：85%+准确率保留）
2. **多项式激活优化**：用zkML友好的多项式函数替换ReLU（减少约50%约束）
3. **自适应量化**：基于Hessian的混合精度量化（进一步减少约40%约束）
4. **端到端基准测试**：使用EZKL完成链上部署和性能验证

**预期结果**: 10-50×证明加速，准确率下降<5%

---

## 技术栈

- **深度学习**: PyTorch 2.9.0
- **数据源**: Binance API (BTC/USDT 1分钟K线)
- **技术指标**: pandas-ta (EMA, RSI, MACD等)
- **zkML工具**: EZKL (Halo2证明系统)
- **可视化**: matplotlib, seaborn
- **开发环境**: Python 3.13

---

## 项目架构

```
CNN Teacher (Baseline)
    ↓ Knowledge Distillation (Exp 1)
MLP Student
    ↓ Polynomial Activation (Exp 2)
MLP + Poly Activation
    ↓ Adaptive Quantization (Exp 3)
Optimized MLP
    ↓ EZKL Compilation (Exp 4)
On-chain Verifiable Model
```

### CNN Teacher (28K参数)
```
Input: (batch, 60, 8) - 60分钟窗口，8维特征
→ Conv1D(8→64, k=5) + BN + ReLU + MaxPool(2)
→ Conv1D(64→128, k=3) + BN + ReLU + MaxPool(2)
→ GlobalAvgPool
→ FC(128→3)
Output: (batch, 3) - [SELL, HOLD, BUY]
```

### MLP Student (80K参数)
```
Input: x_flat ∈ R^480
→ FC(128) → σ_poly → Dropout(0.3)
→ FC(64) → σ_poly → Dropout(0.3)
→ FC(32) → σ_poly
→ FC(3) → Softmax
```

---

## 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/yourusername/VeriRegime.git
cd VeriRegime

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据收集

```bash
# 从Binance获取BTC/USDT 1分钟数据 (2023-2024)
python src/data_collection.py

# 数据将保存至 data/btc_usdt_1m_processed.csv
# 预计耗时: 2-4小时（约100万条数据）
```

### 3. 数据分割

```bash
# 按时间顺序分割为 train/val/test (70/15/15)
python src/data_split.py

# 输出:
# - data/train.csv (~500K样本)
# - data/val.csv (~100K样本)
# - data/test.csv (~100K样本)
```

### 4. 训练CNN Teacher

```bash
# 训练CNN baseline模型
python src/train_cnn.py

# 训练配置:
# - Batch size: 256
# - Learning rate: 1e-3 (AdamW)
# - Epochs: 50 (early stop patience=10)
# - Label smoothing: 0.1

# 输出:
# - models/cnn_teacher_best.pth
# - results/figures/cnn_training_curves.png
# - results/figures/cnn_confusion_matrix.png
```

**目标性能**: Test Accuracy ≥ 60%, F1 Score ≥ 60%

### 5. 探索性数据分析

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## 实验计划

### **Exp 0: CNN Teacher Baseline**
- 建立性能基准（Accuracy, F1 Score）
- 估算zkML编译成本

### **Exp 1: 知识蒸馏 (Week 8)**
```bash
python src/distill.py
```
- 对比MLP独立训练 vs 知识蒸馏
- 超参数搜索（Temperature, α/β权重）
- **目标**: 85%+ CNN准确率保留

### **Exp 2: 多项式激活 (Week 9)**
```bash
python src/optimize.py --activation poly
```
- 测试Quadratic, Cubic, GELU-Poly激活
- **目标**: 准确率下降 ≤ 3%, 约束减少~50%

### **Exp 3: 自适应量化 (Week 9)**
```bash
python src/optimize.py --quantize adaptive
```
- Hessian-based bit-width分配 (4-8 bit)
- **目标**: 准确率下降 ≤ 2%, 平均bit-width ≤ 6.5

### **Exp 4: EZKL编译与部署 (Week 10-11)**
```bash
# 导出ONNX
python src/zkml/onnx_export.py

# EZKL编译
ezkl gen-settings -M models/mlp_student.onnx
ezkl compile-circuit
ezkl setup
ezkl prove --data test_input.json
ezkl verify
```
- 测量Proving/Verification时间
- 计算Constraint数量和Proof大小
- **目标**: 10-50× 证明加速 vs CNN

---

## 项目时间线

| Week | 任务 | 状态 |
|------|------|------|
| Week 7 (Nov 8-14) | 数据收集 + CNN Baseline | 🔄 进行中 (85%) |
| Week 8 (Nov 15-21) | MLP Student + 知识蒸馏 | ⏳ 待开始 |
| Week 9 (Nov 22-28) | 多项式激活 + 自适应量化 | ⏳ 待开始 |
| Week 10 (Nov 29-Dec 5) | EZKL编译与基准测试 | ⏳ 待开始 |
| Week 11 (Dec 6-12) | 链上部署（可选） | ⏳ 待开始 |
| Week 12 (Dec 13-19) | 结果分析与Pareto优化 | ⏳ 待开始 |
| Week 13 (Dec 20-26) | 最终报告撰写 | ⏳ 待开始 |
| Week 14 (Dec 27-31) | 展示准备（可选） | ⏳ 待开始 |

详细计划见 [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)

---

## 目录结构

```
VeriRegime/
├── README.md                    # 本文档
├── PROJECT_ROADMAP.md          # 8周执行计划
├── requirements.txt             # Python依赖
│
├── docs/
│   └── proposal_final_cnn.tex  # 研究提案
│
├── data/                        # 数据集
│   ├── btc_usdt_1m_processed.csv
│   ├── train.csv, val.csv, test.csv
│
├── src/
│   ├── data_collection.py       # 数据获取
│   ├── data_split.py            # 数据分割
│   ├── train_cnn.py             # CNN训练
│   ├── distill.py               # 知识蒸馏（待实现）
│   ├── optimize.py              # 优化实验（待实现）
│   │
│   ├── models/
│   │   ├── cnn_teacher.py       # CNN架构
│   │   └── mlp_student.py       # MLP架构（待实现）
│   │
│   ├── data/
│   │   └── dataset.py           # PyTorch Dataset
│   │
│   ├── distillation/            # 知识蒸馏（待实现）
│   ├── activations/             # 多项式激活（待实现）
│   ├── quantization/            # 自适应量化（待实现）
│   └── zkml/                    # EZKL接口（待实现）
│
├── models/                      # 保存的模型
│   └── cnn_teacher_best.pth
│
├── notebooks/
│   └── 01_eda.ipynb             # 探索性分析
│
├── results/
│   ├── figures/                 # 可视化图表
│   └── logs/                    # 训练日志
│
└── reports/                     # 报告文档
    └── week7_progress.md
```

---

## 性能目标

### 最小可接受目标
- ✅ CNN Teacher准确率 ≥ 55%
- ✅ MLP蒸馏准确率保留 ≥ 80%
- ✅ 完成多项式激活**或**量化之一
- ✅ 完成EZKL编译

### 预期目标
- ✅ CNN Teacher准确率 ≥ 60%
- ✅ MLP蒸馏准确率保留 ≥ 85%
- ✅ 完成多项式激活**和**量化
- ✅ Proving时间 < 60s
- ✅ 准确率下降 < 5%
- ✅ 10× 证明加速

### 理想目标
- ✅ CNN Teacher准确率 ≥ 65%
- ✅ MLP蒸馏准确率保留 ≥ 90%
- ✅ 完成链上部署
- ✅ Proving时间 < 30s
- ✅ 准确率下降 < 3%
- ✅ 20-50× 证明加速

---

## 参考文献

### 知识蒸馏
- Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2014
- Urban et al., "Do Deep Convolutional Nets Really Need to be Deep?", ICLR 2017

### zkML优化
- EZKL Documentation: https://docs.ezkl.xyz/
- Kang et al., "ZKML: Efficient and Privacy-Preserving Neural Network Inference", 2023

### 量化神经网络
- Gholami et al., "A Survey of Quantization Methods for Efficient NN Inference", 2021
- Dong et al., "HAWQ: Hessian Aware Quantization", ICCV 2019

---

## 致谢

本项目为香港中文大学（深圳）DDA4220 Deep Learning课程项目。

**导师**: [Instructor Name]
**学生**: Lambert Lin
**学期**: Fall 2024

---

## License

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 联系方式

- **项目负责人**: Lambert Lin
- **Email**: [your-email@example.com]
- **课程**: DDA4220 Deep Learning
- **截止日期**: 2024-12-31

---

**最后更新**: 2024-11-08
