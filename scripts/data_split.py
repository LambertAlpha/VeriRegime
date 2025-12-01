"""
数据分割脚本 - 4小时波动率版本
按时间顺序分割为训练集/验证集/测试集（70/15/15）
"""

import pandas as pd
from pathlib import Path

# ============================================================
# 配置
# ============================================================

INPUT_FILE = 'data/btc_usdt_1m_volatility_4h.csv'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

OUTPUT_DIR = Path('data')
TRAIN_FILE = OUTPUT_DIR / 'train.csv'
VAL_FILE = OUTPUT_DIR / 'val.csv'
TEST_FILE = OUTPUT_DIR / 'test.csv'


# ============================================================
# 主函数
# ============================================================

def main():
    """按时间顺序分割数据"""
    print("\n" + "="*60)
    print("数据分割 - 4小时波动率预测")
    print("="*60)
    
    # 读取数据
    print(f"\n读取数据: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, index_col=0)
    
    # 重命名列（统一命名）
    df = df.rename(columns={
        'future_volatility_4h': 'future_volatility',
        'label_4h': 'label'
    })
    
    print(f"  总样本数: {len(df):,}")
    
    # 计算分割点
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    # 按时间顺序分割
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"\n分割结果:")
    print(f"  训练集: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
    print(f"  验证集: {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
    print(f"  测试集: {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
    
    # 检查标签分布
    for name, subset in [('训练', train_df), ('验证', val_df), ('测试', test_df)]:
        counts = subset['label'].value_counts().sort_index()
        total = len(subset)
        print(f"\n{name}集标签分布:")
        print(f"  低波动 (0): {counts.get(0, 0):,} ({counts.get(0, 0)/total*100:.1f}%)")
        print(f"  高波动 (1): {counts.get(1, 0):,} ({counts.get(1, 0)/total*100:.1f}%)")
    
    # 保存
    print(f"\n保存文件...")
    train_df.to_csv(TRAIN_FILE)
    val_df.to_csv(VAL_FILE)
    test_df.to_csv(TEST_FILE)
    
    print(f"  ✅ {TRAIN_FILE}")
    print(f"  ✅ {VAL_FILE}")
    print(f"  ✅ {TEST_FILE}")
    
    print("\n" + "="*60)
    print("✅ 数据分割完成！")
    print("="*60)
    print("\n下一步: 在notebook中训练模型")
    print("  配置: SEQ_LENGTH = 240  # 4小时窗口")


if __name__ == '__main__':
    main()

