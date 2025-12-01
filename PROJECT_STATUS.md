# VeriRegime 项目状态

**更新时间**: 2024-12-01  
**版本**: 2.0 (4小时波动率)  
**状态**: ✅ 项目重构完成，已迁移至4小时波动率预测

---

## 🎉 重大变更

### ✅ 从1小时迁移至4小时波动率预测

**理由**:
1. **更符合DeFi应用场景**: 4小时更新周期适合链上协议
2. **更高的信噪比**: 过滤短期噪声，提升预测准确性
3. **更实用的风险管理**: 给协议足够时间调整参数

### 配置对比

| 参数 | 旧版本(1h) | 新版本(4h) |
|------|-----------|-----------|
| 预测窗口 | 60分钟 | 240分钟 |
| 输入序列 | SEQ_LENGTH=60 | SEQ_LENGTH=240 |
| 波动率阈值 | 0.05% | 0.1% |
| 标签分布 | 56% vs 44% | 90% vs 10% |
| 任务性质 | 平衡分类 | 不平衡分类 |
| 应用场景 | 短期交易 | DeFi风控 ⭐ |

---

## 📊 当前数据统计

```
总样本: 973,830
├── 训练集: 681,681 (70%)
│   ├── 低波动(0): 615,008 (90.2%)
│   └── 高波动(1): 66,673 (9.8%)
│
├── 验证集: 146,074 (15%)
│   ├── 低波动(0): 134,101 (91.8%)
│   └── 高波动(1): 11,973 (8.2%)
│
└── 测试集: 146,075 (15%)
    ├── 低波动(0): 129,001 (88.3%)
    └── 高波动(1): 17,074 (11.7%)
```

**特点**: 
- 90:10不平衡分布（高波动是少数类）
- 已使用类别权重处理不平衡
- 评估指标以F1-score为主

---

## 📁 项目结构

```
VeriRegime/
├── train/                      # 可复用训练模块
│   ├── __init__.py
│   ├── models.py              # CNN模型（35K参数）
│   ├── dataset.py             # 数据加载
│   └── trainer.py             # 训练器（支持类别权重）
│
├── notebooks/                  # Jupyter notebooks
│   └── train_volatility.ipynb # 4h波动率训练（已更新）
│
├── results/                    # 训练结果
│   ├── checkpoints/
│   ├── figures/
│   │   └── volatility_distribution.png  # 4h数据分布
│   └── logs/
│
├── scripts/                    # 数据处理
│   ├── relabel_volatility.py  # 4h波动率标注
│   ├── data_split.py          # 数据分割
│   ├── data_collection.py
│   └── feature_engineering_v2.py
│
└── data/                       # 数据文件
    ├── btc_usdt_1m_volatility_4h.csv  # 完整数据
    ├── train.csv              # 训练集
    ├── val.csv                # 验证集
    └── test.csv               # 测试集
```

---

## 🚀 使用方式

### 训练模型

```bash
# 方式1: Jupyter Notebook（推荐）
jupyter lab
# 打开 notebooks/train_volatility.ipynb
# 执行所有cell

# 方式2: Python脚本
python -c "
from train import CNNVolatility, create_dataloaders, Trainer
# ... 训练代码
"
```

### 关键配置

在notebook中：

```python
# 数据配置
SEQ_LENGTH = 240  # 4小时 = 240分钟 ⭐
FEATURE_COLS = ['ema_5', 'ema_10', 'ema_20', 'rsi', 'macd', 'volume_ma_5', 'volume_ma_10']

# 训练配置
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 512

# 数据路径
TRAIN_CSV = '../data/train.csv'
VAL_CSV = '../data/val.csv'
TEST_CSV = '../data/test.csv'
```

---

## 🎯 性能目标与评估

### 基准目标

由于90%的样本是低波动，随机猜测可达90%准确率。因此：

- **准确率**: ≥90% (baseline)，目标92%+
- **F1-score**: ≥0.3 (关键指标) ⭐
- **高波动召回率**: ≥50% (识别真实高波动)
- **高波动精确率**: ≥30% (预测准确性)

### 为什么用F1而非准确率？

在90:10不平衡数据中：
- 模型全预测0（低波动）→ 准确率90%但毫无价值
- F1-score平衡精确率和召回率，更能反映真实性能

---

## ⚠️ 处理类别不平衡

已内置的策略：

1. **类别权重**: 
```python
# train/trainer.py
class_weights = [0.846, 1.222]  # 自动计算
self.criterion = nn.CrossEntropyLoss(weight=class_weights)
```

2. **评估指标**:
```python
# 使用macro F1作为主指标
f1 = f1_score(y_true, y_pred, average='macro')
```

3. **混淆矩阵分析**:
```
              预测LOW  预测HIGH
真实LOW       xxxxx     xxxx    ← 关注假阳性
真实HIGH      xxxx      xxxx    ← 关注召回率 ⭐
```

---

## 🔬 下一步

1. ✅ **项目结构重构完成**
2. ✅ **迁移至4小时波动率**
3. ✅ **数据生成与分割完成**
4. 🔄 **当前**: 准备训练CNN baseline
5. ⏳ 等待训练完成
6. ⏳ 如果F1≥0.3，进入知识蒸馏阶段
7. ⏳ zkML优化与EZKL编译

---

## 📝 训练Tips

### 监控指标

```bash
# 实时查看训练（如果有日志）
tail -f results/logs/*.log
```

### 判断模型好坏

不要只看准确率！重点看：

1. **验证F1-score**: 是否≥0.3
2. **高波动召回率**: 能识别多少真实高波动
3. **混淆矩阵**: 高波动行的表现

### 调优建议

如果F1<0.3:
- 增加高波动类的权重
- 尝试Focal Loss
- 增加训练轮数
- 调整阈值（0.1% → 0.08%增加高波动样本）

---

## 🎓 研究意义

**4小时波动率预测**相比1小时的优势：

1. **学术价值**: 
   - DeFi风控是新兴方向
   - 不平衡分类更有挑战性
   - 可以在报告中对比1h vs 4h

2. **实用价值**:
   - 直接应用于借贷协议（Aave/Compound）
   - 期权定价的输入参数
   - AMM动态手续费调整

3. **zkML契合度**:
   - 4小时更新周期适合链上gas成本
   - 风险管理比投机交易更有价值
   - 可验证性需求更强

---

**重构完成！项目已完全迁移至4小时波动率预测。**

**下一步**: 在notebook中训练模型，观察不平衡数据下的表现！ 🚀
