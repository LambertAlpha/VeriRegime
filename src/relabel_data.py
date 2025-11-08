"""
重新标注数据脚本
分析收益率分布，找到合适的阈值使标签分布相对均衡
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_return_distribution(csv_file):
    """分析未来收益率分布"""
    print(f"读取数据: {csv_file}")
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    returns = df['future_return_1h'].dropna()

    print(f"\n收益率统计:")
    print(f"  样本数: {len(returns):,}")
    print(f"  均值: {returns.mean():.4f}%")
    print(f"  标准差: {returns.std():.4f}%")
    print(f"  中位数: {returns.median():.4f}%")
    print(f"\n分位数:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = returns.quantile(p/100)
        print(f"  {p:2d}%: {val:7.4f}%")

    # 测试不同阈值下的标签分布
    print("\n\n不同阈值下的标签分布:")
    print("="*70)
    print(f"{'阈值':>6} | {'SELL':>8} | {'HOLD':>8} | {'BUY':>8} | {'均衡度':>8}")
    print("-"*70)

    best_threshold = None
    best_balance = 0

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        labels = []
        for r in returns:
            if r > threshold:
                labels.append(2)  # BUY
            elif r < -threshold:
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD

        labels = np.array(labels)
        sell_pct = (labels == 0).sum() / len(labels) * 100
        hold_pct = (labels == 1).sum() / len(labels) * 100
        buy_pct = (labels == 2).sum() / len(labels) * 100

        # 计算均衡度：理想是33.33%，偏差越小越好
        balance = 100 - (abs(sell_pct - 33.33) + abs(hold_pct - 33.33) + abs(buy_pct - 33.33))

        print(f"±{threshold:.1f}% | {sell_pct:7.2f}% | {hold_pct:7.2f}% | {buy_pct:7.2f}% | {balance:7.2f}")

        if balance > best_balance:
            best_balance = balance
            best_threshold = threshold

    print("="*70)
    print(f"\n推荐阈值: ±{best_threshold:.1f}% (均衡度: {best_balance:.2f})")

    return best_threshold


def relabel_data(input_file, output_file, threshold):
    """使用新阈值重新生成标签"""
    print(f"\n使用阈值 ±{threshold}% 重新标注数据...")

    df = pd.read_csv(input_file, index_col=0, parse_dates=True)

    # 重新生成标签
    def classify(return_pct):
        if pd.isna(return_pct):
            return np.nan
        elif return_pct > threshold:
            return 2  # BUY
        elif return_pct < -threshold:
            return 0  # SELL
        else:
            return 1  # HOLD

    df['label'] = df['future_return_1h'].apply(classify)

    # 删除NaN
    df_clean = df.dropna()

    # 统计
    label_counts = df_clean['label'].value_counts().sort_index()
    print(f"\n新标签分布:")
    print(f"  SELL (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df_clean)*100:.2f}%)")
    print(f"  HOLD (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df_clean)*100:.2f}%)")
    print(f"  BUY  (2): {label_counts.get(2, 0):,} ({label_counts.get(2, 0)/len(df_clean)*100:.2f}%)")

    # 保存
    df_clean.to_csv(output_file)
    print(f"\n✅ 重新标注的数据已保存: {output_file}")
    print(f"   样本数: {len(df_clean):,}")

    return df_clean


if __name__ == '__main__':
    input_file = 'data/btc_usdt_1m_processed.csv'
    output_file = 'data/btc_usdt_1m_processed.csv'

    # 分析并找到最佳阈值
    best_threshold = analyze_return_distribution(input_file)

    # 重新标注
    df = relabel_data(input_file, output_file, best_threshold)

    print("\n" + "="*70)
    print("🎉 数据重新标注完成！")
    print("="*70)
