# VeriRegime 项目执行路线图

## 项目概述

**项目名称**: VeriRegime - CNN→MLP知识蒸馏优化zkML系统
**时间跨度**: Week 7 - Week 14 (Nov 8 - Dec 31, 2024)
**核心目标**: 将高性能CNN蒸馏为zkML友好的MLP，实现85%+准确率保留，10-50×证明加速

## 总体技术路线

```
CNN Teacher (Baseline)
    ↓ Distillation
MLP Student (Exp 1)
    ↓ Polynomial Activation (Exp 2)
MLP + Poly Activation
    ↓ Adaptive Quantization (Exp 3)
Optimized MLP
    ↓ EZKL Compilation (Exp 4)
On-chain Verifiable Model
```

## 8周详细计划

### **Week 7: 数据收集与CNN Baseline (Nov 8-14)**

**目标**: 获取高质量训练数据，建立CNN Teacher性能基准

#### Day 1-2: 数据收集
- **任务**:
  - 安装依赖: `pip install ccxt pandas ta-lib torch torchvision numpy matplotlib scikit-learn`
  - 运行 `src/data_collection.py` 获取BTC/USDT 1分钟K线
  - 时间范围: 2023-01-01 至 2024-11-08 (~500K样本)
  - 计算8维特征: EMA(5,10,20), RSI(14), MACD, VolumeMA(5,10)
  - 生成三分类标签: BUY(+2%), HOLD, SELL(-2%)

- **成功标准**:
  - 数据文件 `data/btc_usdt_1m_processed.csv` 生成
  - 样本数 ≥ 400K (考虑前期指标NaN)
  - 标签分布相对均衡 (每类 ≥ 20%)

#### Day 3: 数据分割与探索性分析
- **任务**:
  - 创建 `src/data_split.py`: 按时间顺序分割 70/15/15
  - 创建 `notebooks/01_eda.ipynb`:
    - 价格趋势可视化
    - 特征相关性矩阵
    - 标签分布统计
    - 技术指标有效性分析

- **成功标准**:
  - Train/Val/Test 文件生成
  - 数据质量报告完成
  - 识别潜在数据质量问题

#### Day 4-6: CNN Teacher训练
- **任务**:
  - 实现 `src/models/cnn_teacher.py`:
    ```python
    Architecture:
    X ∈ R^(60×8) → Conv1D(64, k=5, stride=1) → ReLU → MaxPool(2)
                 → Conv1D(128, k=3, stride=1) → ReLU → MaxPool(2)
                 → GlobalAvgPool → FC(128→3) → Softmax
    ```
  - 实现 `src/data/dataset.py`: 滑动窗口数据加载
  - 实现 `src/train_cnn.py`: 训练循环 + 早停
  - 训练配置:
    - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
    - Batch size: 256
    - Epochs: 50 (early stop patience=10)
    - Loss: CrossEntropy + Label Smoothing (0.1)

- **成功标准**:
  - **Test Accuracy ≥ 60%**
  - **Test F1 Score ≥ 60%**
  - 模型文件保存至 `models/cnn_teacher_best.pth`
  - 混淆矩阵和训练曲线保存

#### Day 7: Week总结与代码审查
- **任务**:
  - 代码清理和文档补充
  - 创建 `reports/week7_summary.md`
  - 验证所有实验可复现性

---

### **Week 8: MLP Student与知识蒸馏 (Nov 15-21)**

**目标**: 实现MLP Student架构，完成知识蒸馏实验(Exp 1)

#### Day 1-2: MLP Student实现
- **任务**:
  - 实现 `src/models/mlp_student.py`:
    ```python
    Input: x_flat ∈ R^480 → FC(128) → ReLU
                          → Dropout(0.3) → FC(64) → ReLU
                          → Dropout(0.3) → FC(32) → ReLU
                          → FC(3) → Softmax
    ```
  - 独立训练验证MLP容量 (无蒸馏baseline)

- **成功标准**:
  - MLP独立训练准确率记录 (预期45-55%)

#### Day 3-5: 知识蒸馏实现
- **任务**:
  - 实现 `src/distillation/kd_trainer.py`:
    ```python
    L = α·L_CE(y_true, ŷ_student)
      + β·L_KD(z_teacher/T, z_student/T)
      + γ·(λ₁‖W‖₁ + λ₂·Σ ReLU(|w|-τ))
    ```
  - 超参数搜索:
    - Temperature T ∈ {3, 5, 7}
    - α:β ∈ {0.5:0.5, 0.3:0.7}
    - γ ∈ {1e-4, 1e-5}

- **成功标准**:
  - **Student准确率 ≥ 85% * CNN Teacher准确率**
  - 保存最佳蒸馏模型至 `models/mlp_student_kd.pth`

#### Day 6-7: 实验结果分析
- **任务**:
  - 创建对比表格:
    | Model | Acc | F1 | Params | Retention |
    |-------|-----|----|---------|----|
    | CNN Teacher | X% | X% | ~50K | - |
    | MLP Baseline | X% | X% | ~80K | X% |
    | MLP + KD | X% | X% | ~80K | X% |

  - 可视化logits分布差异
  - 撰写 `reports/week8_distillation.md`

---

### **Week 9: 多项式激活与自适应量化 (Nov 22-28)**

**目标**: 完成Exp 2(多项式激活)和Exp 3(自适应量化)

#### Day 1-3: 多项式激活函数替换 (Exp 2)
- **任务**:
  - 实现 `src/activations/polynomial.py`:
    ```python
    - σ_quad(x) = x²
    - σ_cubic(x) = x - x³/6
    - σ_gelu_poly(x) = x·(a₀ + a₁x² + a₂x⁴)
    ```
  - 微调策略:
    1. 冻结所有权重
    2. 仅优化多项式系数 (10 epochs)
    3. 全参数微调 (20 epochs, lr=1e-4)

- **成功标准**:
  - **准确率下降 ≤ 3%**
  - 保存模型至 `models/mlp_student_poly.pth`

#### Day 4-6: 自适应量化 (Exp 3)
- **任务**:
  - 实现 `src/quantization/adaptive_quant.py`:
    ```python
    # Hessian-based sensitivity分析
    H = ∇²_w L(w, D_val)
    sensitivity[i] = tr(H_i) / ‖w_i‖₂

    # 分配bit-width
    b[i] = b_min + (b_max - b_min) * sigmoid(sensitivity[i])
    ```
  - 量化配置:
    - b_min = 4, b_max = 8
    - 使用PACT量化感知训练

- **成功标准**:
  - **准确率下降 ≤ 2%**
  - 平均bit-width ≤ 6.5
  - 保存至 `models/mlp_student_quantized.pth`

#### Day 7: Week总结
- **任务**:
  - 创建性能对比表:
    | Optimization | Acc | ΔAcc | Constraints (est.) |
    |--------------|-----|------|--------------------|
    | MLP + KD | X% | - | ~100K |
    | + Poly Act | X% | -X% | ~50K |
    | + Quant | X% | -X% | ~30K |

  - 撰写 `reports/week9_optimization.md`

---

### **Week 10: EZKL编译与离线基准测试 (Nov 29 - Dec 5)**

**目标**: 将优化后MLP编译为Halo2电路，完成Exp 4

#### Day 1-2: EZKL环境配置
- **任务**:
  - 安装EZKL: `cargo install ezkl` (或从源码编译)
  - 导出ONNX模型:
    ```python
    torch.onnx.export(mlp_student, dummy_input,
                      "models/mlp_student.onnx")
    ```
  - 验证ONNX正确性

#### Day 3-5: 电路编译与证明生成
- **任务**:
  - 生成电路: `ezkl gen-settings -M models/mlp_student.onnx`
  - 编译电路: `ezkl compile-circuit`
  - 本地证明生成:
    ```bash
    ezkl setup
    ezkl prove --data test_input.json
    ezkl verify
    ```

- **基准测试指标**:
  - Constraint数量
  - Proving时间 (100个样本平均)
  - Verification时间
  - Proof大小

#### Day 6-7: 性能对比与分析
- **任务**:
  - 对比CNN Teacher (如果可编译):
    | Model | Constraints | Prove Time | Verify Time | Proof Size |
    |-------|-------------|------------|-------------|------------|
    | CNN Teacher | X | X s | X ms | X KB |
    | MLP Optimized | X | X s | X ms | X KB |
    | Speedup | - | X× | X× | X× |

  - 创建Pareto前沿图: Accuracy vs Proof Time
  - 撰写 `reports/week10_zkml_benchmark.md`

---

### **Week 11: 链上部署(可选) (Dec 6-12)**

**注意**: 此周任务为可选，根据Week 10结果决定是否执行

#### Day 1-3: EVM合约部署
- **任务**:
  - 部署验证合约至Sepolia测试网
  - 实现链上推理调用接口
  - 测试Gas消耗

#### Day 4-7: 端到端验证
- **任务**:
  - 创建完整交易流程:
    1. 链下模型推理
    2. 生成ZK证明
    3. 链上验证
    4. 返回交易信号

  - 性能测试:
    - 端到端延迟
    - Gas成本分析
    - 可扩展性评估

**如Week 10发现重大问题，改用此周进行优化和调试**

---

### **Week 12: 结果分析与Pareto优化 (Dec 13-19)**

**目标**: 完成所有实验对比，生成最终性能报告

#### Day 1-3: 综合性能分析
- **任务**:
  - 汇总所有实验结果:
    | Experiment | Accuracy | Retention | Constraints | Prove Time | Speedup |
    |------------|----------|-----------|-------------|------------|---------|
    | Exp 0: CNN Baseline | X% | - | ~200K | Xs | 1× |
    | Exp 1: KD | X% | X% | ~100K | Xs | X× |
    | Exp 2: +Poly | X% | X% | ~50K | Xs | X× |
    | Exp 3: +Quant | X% | X% | ~30K | Xs | X× |

  - 创建可视化:
    - 训练曲线对比
    - 混淆矩阵对比
    - Pareto前沿: (Accuracy, Proof Time)

#### Day 4-5: 消融研究
- **任务**:
  - 验证各优化技术贡献:
    | Ablation | Setup | Acc | Prove Time |
    |----------|-------|-----|------------|
    | Full | KD+Poly+Quant | X% | X s |
    | w/o Quant | KD+Poly | X% | X s |
    | w/o Poly | KD+Quant | X% | X s |
    | w/o KD | Poly+Quant | X% | X s |

#### Day 6-7: 理论分析与讨论
- **任务**:
  - 分析为什么CNN→MLP蒸馏有效:
    - CNN学习的局部特征提取
    - MLP学习的位置感知加权
    - 知识蒸馏的软目标平滑作用

  - 失败案例分析:
    - 哪些市场状态下模型失效？
    - MLP丢失了哪些CNN捕获的特征？

  - 撰写 `reports/week12_comprehensive_analysis.md`

---

### **Week 13: 最终报告撰写 (Dec 20-26)**

**目标**: 完成项目最终报告(10-15页)

#### Day 1-2: 方法论章节
- **任务**:
  - Introduction (1页):
    - zkML背景与挑战
    - 本研究动机与贡献

  - Related Work (1.5页):
    - 知识蒸馏研究
    - zkML优化技术
    - 量化神经网络

  - Methodology (3页):
    - 问题形式化
    - CNN Teacher架构
    - MLP Student设计
    - 知识蒸馏框架
    - 多项式激活替换
    - 自适应量化策略
    - EZKL编译流程

#### Day 3-4: 实验与结果章节
- **任务**:
  - Experimental Setup (1页):
    - 数据集描述
    - 实验环境
    - 评估指标

  - Results (3页):
    - 表格: 所有实验对比
    - 图表: Pareto前沿、训练曲线
    - 消融研究结果
    - 链上部署性能(如有)

#### Day 5-6: 讨论与结论
- **任务**:
  - Discussion (1.5页):
    - CNN→MLP蒸馏为何有效
    - 各优化技术的trade-off
    - zkML实用性分析
    - 局限性讨论

  - Conclusion (0.5页):
    - 总结贡献
    - 未来工作方向

#### Day 7: 最终审校
- **任务**:
  - 完整性检查
  - 图表质量审核
  - 参考文献格式
  - LaTeX编译测试
  - 生成 `reports/final_report.pdf`

---

### **Week 14: 展示准备 (Dec 27-31, 可选)**

**目标**: 准备项目展示材料

#### Day 1-3: 幻灯片制作
- **任务**:
  - 创建15分钟演讲PPT:
    1. 研究动机 (2分钟)
    2. 技术方案 (5分钟)
    3. 实验结果 (5分钟)
    4. 讨论与展望 (3分钟)

  - 重点突出:
    - Pareto前沿可视化
    - 端到端demo视频
    - 性能对比表

#### Day 4-5: Demo准备
- **任务**:
  - 创建Jupyter Notebook demo:
    - 实时BTC数据获取
    - 模型推理
    - ZK证明生成
    - 结果可视化

  - 录制演示视频 (可选)

#### Day 6-7: 答辩预演
- **任务**:
  - 准备预期问题回答:
    - 为什么选CNN而非Transformer？
    - 如何保证蒸馏后准确率？
    - zkML实际部署成本如何？
    - 未来改进方向？

  - 时间控制练习

---

## 风险管理与应急计划

### **高风险环节**

#### 1. Week 7: CNN性能不达标 (Acc < 60%)
- **原因**: 数据质量问题、标签噪声、模型容量不足
- **应急方案**:
  - 调整标签阈值 (2% → 1.5% 或 2.5%)
  - 增加CNN深度 (添加第三层卷积)
  - 尝试数据增强 (Mixup、Cutout)
  - 最坏情况: 降低准确率目标至55%

#### 2. Week 8: 蒸馏准确率保留 < 85%
- **原因**: 知识蒸馏超参数不当、容量不匹配
- **应急方案**:
  - 扩大MLP宽度 (128→256)
  - 尝试中间层特征蒸馏
  - 延长训练周期
  - 最坏情况: 接受80%保留率，强调性能提升

#### 3. Week 9: 多项式激活准确率骤降 (> 5%)
- **原因**: 激活函数表达能力不足
- **应急方案**:
  - 提高多项式阶数 (Cubic → Quintic)
  - 混合激活策略 (前几层保留ReLU)
  - 更长的微调周期
  - 最坏情况: 跳过此实验，直接量化

#### 4. Week 10: EZKL编译失败
- **原因**: ONNX不兼容、电路过大
- **应急方案**:
  - 简化模型架构 (减少层数)
  - 使用ezkl-python而非CLI
  - 联系EZKL社区寻求帮助
  - 最坏情况: 仅进行理论分析，引用已有benchmark

---

## 成功标准

### **最小可接受目标 (Minimum Viable)**
- ✅ CNN Teacher准确率 ≥ 55%
- ✅ MLP蒸馏准确率保留 ≥ 80%
- ✅ 完成多项式激活或量化之一
- ✅ 完成EZKL编译(即使未部署链上)
- ✅ 提交完整项目报告

### **预期目标 (Expected)**
- ✅ CNN Teacher准确率 ≥ 60%
- ✅ MLP蒸馏准确率保留 ≥ 85%
- ✅ 完成多项式激活 + 量化
- ✅ Proving时间 < 60s (本地)
- ✅ 准确率下降 < 5%
- ✅ 10× 证明加速

### **理想目标 (Stretch)**
- ✅ CNN Teacher准确率 ≥ 65%
- ✅ MLP蒸馏准确率保留 ≥ 90%
- ✅ 完成链上部署
- ✅ Proving时间 < 30s
- ✅ 准确率下降 < 3%
- ✅ 20-50× 证明加速

---

## 工作时间估计

| Week | 核心任务 | 预计工时 | 风险等级 |
|------|----------|----------|----------|
| Week 7 | 数据+CNN | 25h | 中 |
| Week 8 | 蒸馏 | 30h | 高 |
| Week 9 | 优化 | 28h | 高 |
| Week 10 | EZKL | 25h | 极高 |
| Week 11 | 链上部署 | 20h (可选) | 极高 |
| Week 12 | 分析 | 20h | 低 |
| Week 13 | 报告 | 25h | 低 |
| Week 14 | 展示 | 15h (可选) | 低 |
| **总计** | | **188h (核心130h)** | |

**建议**: 每周投入18-25小时，优先保证Week 7-10核心实验完成

---

## 代码结构预览

```
VeriRegime/
├── PROJECT_ROADMAP.md          # 本文档
├── README.md                    # 项目说明
├── requirements.txt             # Python依赖
├── setup.py                     # 安装配置
│
├── docs/
│   ├── proposal_final_cnn.tex  # 最终提案
│   └── figures/                 # 论文图表
│
├── data/
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后数据
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── src/
│   ├── data_collection.py       # 数据获取
│   ├── data_split.py            # 数据分割
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # PyTorch Dataset
│   │   └── transforms.py        # 数据变换
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_teacher.py       # CNN架构
│   │   └── mlp_student.py       # MLP架构
│   │
│   ├── activations/
│   │   ├── __init__.py
│   │   └── polynomial.py        # 多项式激活
│   │
│   ├── distillation/
│   │   ├── __init__.py
│   │   ├── kd_trainer.py        # 知识蒸馏
│   │   └── losses.py            # 蒸馏损失
│   │
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── adaptive_quant.py    # 自适应量化
│   │   └── qat.py               # 量化感知训练
│   │
│   ├── zkml/
│   │   ├── __init__.py
│   │   ├── onnx_export.py       # ONNX导出
│   │   └── ezkl_wrapper.py      # EZKL接口
│   │
│   ├── train_cnn.py             # CNN训练脚本
│   ├── train_mlp.py             # MLP训练脚本
│   ├── distill.py               # 蒸馏脚本
│   ├── optimize.py              # 优化脚本
│   └── evaluate.py              # 评估脚本
│
├── notebooks/
│   ├── 01_eda.ipynb             # 探索性分析
│   ├── 02_cnn_analysis.ipynb    # CNN结果分析
│   ├── 03_distillation.ipynb    # 蒸馏实验
│   ├── 04_optimization.ipynb    # 优化实验
│   └── 05_zkml_demo.ipynb       # zkML演示
│
├── models/                      # 保存的模型
│   ├── cnn_teacher_best.pth
│   ├── mlp_student_kd.pth
│   ├── mlp_student_poly.pth
│   ├── mlp_student_quantized.pth
│   └── mlp_student.onnx
│
├── results/                     # 实验结果
│   ├── figures/                 # 图表
│   ├── tables/                  # 表格
│   └── logs/                    # 训练日志
│
├── reports/                     # 报告文档
│   ├── week7_summary.md
│   ├── week8_distillation.md
│   ├── week9_optimization.md
│   ├── week10_zkml_benchmark.md
│   ├── week12_comprehensive_analysis.md
│   └── final_report.pdf
│
├── tests/                       # 单元测试
│   ├── test_models.py
│   ├── test_distillation.py
│   └── test_quantization.py
│
└── scripts/                     # 辅助脚本
    ├── run_all_experiments.sh
    ├── benchmark.sh
    └── deploy.sh
```

---

## 参考文献与资源

### **知识蒸馏**
- Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2014
- Urban et al., "Do Deep Convolutional Nets Really Need to be Deep?", ICLR 2017

### **zkML优化**
- EZKL Documentation: https://docs.ezkl.xyz/
- Kang et al., "ZKML: Efficient and Privacy-Preserving Neural Network Inference", 2023
- ZKonduit Blog: https://blog.ezkl.xyz/

### **量化神经网络**
- Gholami et al., "A Survey of Quantization Methods for Efficient NN Inference", 2021
- Dong et al., "HAWQ: Hessian Aware Quantization", ICCV 2019

### **技术栈**
- PyTorch: https://pytorch.org/
- CCXT: https://github.com/ccxt/ccxt
- TA-Lib: https://github.com/mrjbq7/ta-lib
- EZKL: https://github.com/zkonduit/ezkl

---

## 联系与支持

**项目负责人**: Lambert Lin
**课程**: DDA4220 Deep Learning
**截止日期**: 2024年12月31日

**遇到问题时**:
1. 检查本路线图的应急方案
2. 查阅相关文献与文档
3. EZKL问题可在GitHub开issue
4. 及时与课程助教沟通

---

**最后更新**: 2024-11-08
**版本**: v1.0
