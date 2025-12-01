"""
波动率标注脚本 - 4小时版本
预测未来4小时的波动率水平（高/低）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# 配置参数
# ============================================================

VOLATILITY_WINDOW = 240      # 4小时 = 240分钟
VOLATILITY_THRESHOLD = 0.001   # 阈值：0.1% (识别真正的高波动)
INPUT_FILE = 'data/btc_usdt_1m_volatility_4h.csv'  # 使用已有4h数据
OUTPUT_FILE = 'data/btc_usdt_1m_volatility_4h.csv'


# ============================================================
# 核心函数
# ============================================================

def calculate_future_volatility(df, window=240):
    """
    计算未来N分钟的实际波动率
    
    Args:
        df: 包含'close'列的DataFrame
        window: 窗口大小（分钟），默认240 = 4小时
    
    Returns:
        Series: 未来波动率
    """
    # 计算收益率
    returns = df['close'].pct_change()
    
    # 未来窗口的标准差 = 波动率
    future_vol = returns.shift(-window).rolling(window).std()
    
    return future_vol


def create_volatility_labels_4h(df, threshold=0.001):
    """
    生成4小时波动率二分类标签
    
    Args:
        df: 原始数据
        threshold: 波动率阈值（建议0.1%）
    
    Returns:
        df: 添加了'future_volatility_4h'和'label_4h'列
    """
    print(f"\n{'='*60}")
    print(f"4小时波动率标注配置")
    print(f"{'='*60}")
    print(f"阈值: {threshold:.4f} ({threshold*100:.2f}%)")
    print(f"窗口: {VOLATILITY_WINDOW}分钟 (4小时)")
    
    # 计算未来4小时波动率
    df['future_volatility_4h'] = calculate_future_volatility(df, window=VOLATILITY_WINDOW)
    
    # 生成标签：0=低波动, 1=高波动
    df['label_4h'] = (df['future_volatility_4h'] > threshold).astype(int)
    
    # 统计NaN情况
    initial_len = len(df)
    df_clean = df.dropna(subset=['future_volatility_4h', 'label_4h'])
    removed = initial_len - len(df_clean)
    
    print(f"\n数据清理:")
    print(f"  原始样本: {initial_len:,}")
    print(f"  有效样本: {len(df_clean):,}")
    print(f"  删除NaN: {removed:,} ({removed/initial_len*100:.2f}%)")
    
    # 统计标签分布
    label_counts = df_clean['label_4h'].value_counts().sort_index()
    total = len(df_clean)
    
    print(f"\n标签分布:")
    print(f"  低波动 (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total*100:.2f}%)")
    print(f"  高波动 (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total*100:.2f}%)")
    
    # 波动率统计
    print(f"\n4小时波动率统计:")
    print(f"  均值:   {df_clean['future_volatility_4h'].mean():.6f}")
    print(f"  中位数: {df_clean['future_volatility_4h'].median():.6f}")
    print(f"  标准差: {df_clean['future_volatility_4h'].std():.6f}")
    print(f"  最小值: {df_clean['future_volatility_4h'].min():.6f}")
    print(f"  最大值: {df_clean['future_volatility_4h'].max():.6f}")
    
    return df_clean


def visualize_4h_distribution(df, save_path='results/figures/volatility_4h_distribution.png'):
    """可视化4小时波动率分布"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 波动率直方图
    axes[0, 0].hist(df['future_volatility_4h'], bins=100, alpha=0.7, 
                    edgecolor='black', color='steelblue')
    axes[0, 0].axvline(VOLATILITY_THRESHOLD, color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Threshold={VOLATILITY_THRESHOLD*100:.2f}%')
    axes[0, 0].set_xlabel('4-Hour Volatility', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Future 4-Hour Volatility Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. 标签分布饼图
    label_counts = df['label_4h'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    labels_pie = [f'LOW ({label_counts.get(0, 0):,})', 
                  f'HIGH ({label_counts.get(1, 0):,})']
    axes[0, 1].pie(label_counts, labels=labels_pie, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 11})
    axes[0, 1].set_title('4H Label Distribution', fontsize=12, fontweight='bold')
    
    # 3. 时间序列（前2000个样本）
    sample_size = min(2000, len(df))
    time_idx = range(sample_size)
    axes[1, 0].fill_between(time_idx, 0, df['label_4h'].iloc[:sample_size], 
                             alpha=0.6, color='coral')
    axes[1, 0].set_xlabel('Time (minutes)', fontsize=11)
    axes[1, 0].set_ylabel('Label', fontsize=11)
    axes[1, 0].set_title('4H Volatility Labels Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(['LOW', 'HIGH'])
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 1h vs 4h波动率对比
    if 'future_volatility' in df.columns:
        axes[1, 1].scatter(df['future_volatility'].iloc[:10000], 
                          df['future_volatility_4h'].iloc[:10000],
                          alpha=0.3, s=1)
        axes[1, 1].set_xlabel('1-Hour Volatility', fontsize=11)
        axes[1, 1].set_ylabel('4-Hour Volatility', fontsize=11)
        axes[1, 1].set_title('1H vs 4H Volatility Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, '1H data not available', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化保存到: {save_path}")
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    """主执行流程"""
    print("\n" + "="*60)
    print("VeriRegime - 4小时波动率标注系统")
    print("="*60)
    
    # 1. 读取数据
    print(f"\n[1/4] 读取数据: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, index_col=0)
    print(f"  数据形状: {df.shape}")
    
    # 2. 生成4小时波动率标签
    print(f"\n[2/4] 生成4小时波动率标签...")
    df_labeled = create_volatility_labels_4h(df, threshold=VOLATILITY_THRESHOLD)
    
    # 3. 保存
    print(f"\n[3/4] 保存标注数据...")
    # 只保留必要的列
    output_cols = ['open', 'high', 'low', 'close', 'volume',
                   'ema_5', 'ema_10', 'ema_20', 'rsi', 'macd', 
                   'volume_ma_5', 'volume_ma_10',
                   'future_volatility_4h', 'label_4h']
    df_labeled[output_cols].to_csv(OUTPUT_FILE)
    print(f"  ✅ 保存到: {OUTPUT_FILE}")
    print(f"  文件大小: {Path(OUTPUT_FILE).stat().st_size / 1024 / 1024:.2f} MB")
    
    # 4. 可视化
    print(f"\n[4/4] 生成可视化...")
    visualize_4h_distribution(df_labeled)
    
    print("\n" + "="*60)
    print("✅ 4小时标注完成！")
    print("="*60)
    print(f"\n下一步:")
    print(f"  1. 运行数据分割: python scripts/data_split_volatility.py (需修改为_4h版本)")
    print(f"  2. 在notebook中修改:")
    print(f"     - TRAIN_CSV = '../data/train_volatility_4h.csv'")
    print(f"     - SEQ_LENGTH = 240  # 4小时窗口")
    print(f"  3. 重新训练模型")


if __name__ == '__main__':
    main()

