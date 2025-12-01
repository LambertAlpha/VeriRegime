# VeriRegime 环境配置

**环境**: conda ml  
**Python**: 3.12.12  
**PyTorch**: 2.9.1  
**GPU**: MPS (Apple Silicon) ✅

---

## ✅ 环境已就绪

当前ml环境已包含所有依赖：
- ✅ PyTorch 2.9.1 (with MPS support)
- ✅ pandas, numpy
- ✅ matplotlib, seaborn
- ✅ scikit-learn
- ✅ jupyter, jupyterlab

---

## 🚀 快速启动

### 训练模型

```bash
# 1. 激活环境
conda activate ml

# 2. 进入项目目录
cd /Users/lambertlin/Projects/VeriRegime

# 3. 启动Jupyter
jupyter lab

# 4. 打开 notebooks/train_volatility.ipynb
# 5. 执行所有cell
```

---

## 📊 当前配置

**数据**:
- 预测目标: 未来4小时波动率
- 阈值: 0.05% (平衡分布)
- 时间窗口: 240分钟
- 样本数: 973,351

**模型**:
- 架构: CNN (35K参数)
- 输入: (batch, 240分钟, 7特征)
- 输出: (batch, 2) → [LOW, HIGH]

**训练**:
- 设备: MPS (Apple Silicon GPU)
- 批次: 512
- 学习率: 1e-3
- 预期时间: 20-30分钟

---

## 🔧 如需重新配置

```bash
# 删除并重建ml环境
conda deactivate
conda env remove -n ml
conda create -n ml python=3.12 -y
conda activate ml

# 安装依赖
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn tqdm jupyter jupyterlab
```

---

**一切就绪！开始训练吧！** 🚀

