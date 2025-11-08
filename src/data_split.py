"""
数据分割脚本：按时间顺序分割训练/验证/测试集
避免时间泄漏，确保训练数据在验证/测试数据之前
"""

import pandas as pd
import numpy as np
from pathlib import Path

def split_time_series_data(input_file, output_dir='data',
                           train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    按时间顺序分割时间序列数据

    Args:
        input_file: 处理后的数据文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        train_df, val_df, test_df
    """
    # 验证比例
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须为1.0"

    print(f"读取数据文件: {input_file}")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)

    total_samples = len(df)
    print(f"总样本数: {total_samples:,}")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"特征列: {list(df.columns)}")

    # 计算分割点（按时间顺序）
    train_end_idx = int(total_samples * train_ratio)
    val_end_idx = int(total_samples * (train_ratio + val_ratio))

    # 分割数据
    train_df = df.iloc[:train_end_idx]
    val_df = df.iloc[train_end_idx:val_end_idx]
    test_df = df.iloc[val_end_idx:]

    # 打印统计信息
    print("\n" + "="*60)
    print("数据分割统计:")
    print("="*60)

    for name, subset in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        print(f"\n{name}:")
        print(f"  样本数: {len(subset):,} ({len(subset)/total_samples*100:.2f}%)")
        print(f"  时间范围: {subset.index[0]} 至 {subset.index[-1]}")

        # 标签分布
        if 'label' in subset.columns:
            label_counts = subset['label'].value_counts().sort_index()
            print(f"  标签分布:")
            print(f"    SELL (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(subset)*100:.2f}%)")
            print(f"    HOLD (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(subset)*100:.2f}%)")
            print(f"    BUY  (2): {label_counts.get(2, 0):,} ({label_counts.get(2, 0)/len(subset)*100:.2f}%)")

    # 保存分割后的数据
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    test_path = output_dir / 'test.csv'

    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    test_df.to_csv(test_path)

    print("\n" + "="*60)
    print("✅ 数据分割完成！文件已保存至:")
    print(f"  训练集: {train_path} ({train_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  验证集: {val_path} ({val_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  测试集: {test_path} ({test_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print("="*60)

    return train_df, val_df, test_df


def verify_no_data_leakage(train_df, val_df, test_df):
    """验证数据分割没有时间泄漏"""
    train_end = train_df.index[-1]
    val_start = val_df.index[0]
    val_end = val_df.index[-1]
    test_start = test_df.index[0]

    assert train_end < val_start, "训练集与验证集存在时间重叠！"
    assert val_end < test_start, "验证集与测试集存在时间重叠！"

    print("\n✅ 数据分割验证通过：无时间泄漏")
    print(f"  训练集结束: {train_end}")
    print(f"  验证集开始: {val_start}")
    print(f"  验证集结束: {val_end}")
    print(f"  测试集开始: {test_start}")


if __name__ == '__main__':
    import sys

    # 默认参数
    input_file = 'data/btc_usdt_1m_processed.csv'

    # 支持命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    # 执行分割
    train_df, val_df, test_df = split_time_series_data(
        input_file=input_file,
        output_dir='data',
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # 验证无时间泄漏
    verify_no_data_leakage(train_df, val_df, test_df)

    print("\n🎉 数据准备完成！现在可以进行EDA或模型训练了")
