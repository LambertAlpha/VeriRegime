# Notebooks 使用说明

## 启动方式

```bash
# 1. 激活ml环境
conda activate ml

# 2. 启动Jupyter Lab
jupyter lab

# 3. 打开 train_volatility.ipynb
```

## 当前Notebook

### train_volatility.ipynb
**功能**: 训练4小时波动率预测CNN模型

**配置** (已设置好，可直接运行):
- SEQ_LENGTH = 240 (4小时)
- 波动率阈值: 0.05%
- 批次大小: 512
- MPS GPU加速 ✅

**预期结果**:
- 准确率: 65-70%
- F1-score: 0.63-0.68
- 训练时间: 20-30分钟

**使用方式**: 按顺序执行所有cell即可！
