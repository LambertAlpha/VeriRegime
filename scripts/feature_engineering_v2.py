"""
增强特征工程 v2
添加更多技术指标和价格/成交量特征

新增特征:
1. 价格特征: 多时间尺度收益率、波动率、价格范围
2. 成交量特征: 成交量变化率、相对成交量
3. 技术指标: 布林带、ATR、OBV、Stochastic等
4. 动量特征: ROC、Williams %R等
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_returns(df, periods=[1, 3, 5, 10, 15, 30]):
    """计算多时间尺度收益率"""
    for period in periods:
        df[f'return_{period}m'] = df['close'].pct_change(periods=period)
    return df


def calculate_volatility(df, windows=[5, 10, 20, 30]):
    """计算滚动波动率"""
    for window in windows:
        df[f'volatility_{window}m'] = df['close'].pct_change().rolling(window).std()
    return df


def calculate_price_features(df):
    """计算价格相关特征"""
    # 价格范围 (归一化)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
    # 价格位置 (在当前蜡烛图中的位置)
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # 上下影线比例
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-8)
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # 实体大小
    df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # 趋势强度 (收盘价相对于N周期移动平均的位置)
    for window in [5, 10, 20]:
        ma = df['close'].rolling(window).mean()
        df[f'price_vs_ma{window}'] = (df['close'] - ma) / ma
    
    return df


def calculate_volume_features(df):
    """计算成交量相关特征"""
    # 成交量变化率
    for period in [1, 3, 5, 10]:
        df[f'volume_change_{period}m'] = df['volume'].pct_change(periods=period)
    
    # 相对成交量 (相对于移动平均)
    for window in [5, 10, 20]:
        volume_ma = df['volume'].rolling(window).mean()
        df[f'volume_ratio_{window}m'] = df['volume'] / (volume_ma + 1e-8)
    
    # 成交量加权价格
    df['vwap_5'] = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
    df['vwap_10'] = (df['close'] * df['volume']).rolling(10).sum() / df['volume'].rolling(10).sum()
    
    return df


def calculate_bollinger_bands(df, window=20, num_std=2):
    """计算布林带"""
    ma = df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std()
    
    df[f'bb_upper_{window}'] = ma + (std * num_std)
    df[f'bb_lower_{window}'] = ma - (std * num_std)
    df[f'bb_middle_{window}'] = ma
    
    # 布林带宽度 (归一化)
    df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
    
    # 价格在布林带中的位置 (%B指标)
    df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'] + 1e-8)
    
    return df


def calculate_atr(df, window=14):
    """计算真实波幅 (ATR)"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f'atr_{window}'] = true_range.rolling(window).mean()
    
    # ATR百分比 (归一化)
    df[f'atr_{window}_pct'] = df[f'atr_{window}'] / df['close']
    
    return df


def calculate_obv(df):
    """计算能量潮 (OBV)"""
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    
    df['obv'] = obv
    
    # OBV移动平均
    df['obv_ma_5'] = df['obv'].rolling(5).mean()
    df['obv_ma_10'] = df['obv'].rolling(10).mean()
    
    # OBV变化率
    df['obv_change_5'] = df['obv'].pct_change(periods=5)
    
    return df


def calculate_stochastic(df, k_window=14, d_window=3):
    """计算随机指标 (Stochastic Oscillator)"""
    low_min = df['low'].rolling(k_window).min()
    high_max = df['high'].rolling(k_window).max()
    
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
    df['stoch_d'] = df['stoch_k'].rolling(d_window).mean()
    
    return df


def calculate_roc(df, periods=[5, 10, 20]):
    """计算变化率 (Rate of Change)"""
    for period in periods:
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    
    return df


def calculate_williams_r(df, window=14):
    """计算威廉指标 (Williams %R)"""
    high_max = df['high'].rolling(window).max()
    low_min = df['low'].rolling(window).min()
    
    df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min + 1e-8)
    
    return df


def calculate_cci(df, window=20):
    """计算商品通道指标 (CCI)"""
    tp = (df['high'] + df['low'] + df['close']) / 3  # Typical Price
    tp_ma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: abs(x - x.mean()).mean())
    
    df[f'cci_{window}'] = (tp - tp_ma) / (0.015 * mad + 1e-8)
    
    return df


def calculate_momentum_features(df):
    """计算动量特征"""
    # 简单动量
    for period in [3, 5, 10]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
    
    # 加速度 (动量的变化)
    df['acceleration_5'] = df['momentum_5'] - df['momentum_5'].shift(5)
    
    return df


def add_all_features(input_csv, output_csv):
    """
    添加所有增强特征
    
    Args:
        input_csv: 输入CSV文件路径 (必须包含 OHLCV + 基础技术指标)
        output_csv: 输出CSV文件路径
    """
    print("="*60)
    print("📊 增强特征工程 v2")
    print("="*60)
    
    # 读取数据
    print(f"\n读取数据: {input_csv}")
    df = pd.read_csv(input_csv, index_col=0)
    print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    
    original_cols = len(df.columns)
    
    # 添加各类特征
    print("\n添加特征...")
    
    print("  [1/11] 多时间尺度收益率...")
    df = calculate_returns(df, periods=[1, 3, 5, 10, 15, 30])
    
    print("  [2/11] 滚动波动率...")
    df = calculate_volatility(df, windows=[5, 10, 20, 30])
    
    print("  [3/11] 价格形态特征...")
    df = calculate_price_features(df)
    
    print("  [4/11] 成交量特征...")
    df = calculate_volume_features(df)
    
    print("  [5/11] 布林带...")
    df = calculate_bollinger_bands(df, window=20)
    
    print("  [6/11] ATR (真实波幅)...")
    df = calculate_atr(df, window=14)
    
    print("  [7/11] OBV (能量潮)...")
    df = calculate_obv(df)
    
    print("  [8/11] 随机指标...")
    df = calculate_stochastic(df, k_window=14, d_window=3)
    
    print("  [9/11] ROC (变化率)...")
    df = calculate_roc(df, periods=[5, 10, 20])
    
    print("  [10/11] Williams %R...")
    df = calculate_williams_r(df, window=14)
    
    print("  [11/11] CCI & 动量...")
    df = calculate_cci(df, window=20)
    df = calculate_momentum_features(df)
    
    # 删除NaN值
    print(f"\n删除NaN...")
    before_drop = len(df)
    df = df.dropna()
    after_drop = len(df)
    print(f"  删除了 {before_drop - after_drop} 行 ({(before_drop - after_drop)/before_drop*100:.2f}%)")
    
    new_cols = len(df.columns) - original_cols
    print(f"\n✅ 特征添加完成!")
    print(f"   新增特征: {new_cols} 个")
    print(f"   总特征数: {len(df.columns)} 个")
    print(f"   有效样本: {len(df)} 行")
    
    # 显示所有特征列
    print(f"\n所有特征列:")
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label', 'future_return_1h']]
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {col}")
    
    # 保存
    print(f"\n保存到: {output_csv}")
    df.to_csv(output_csv)
    print(f"✅ 保存完成!")
    
    return df


def main():
    input_csv = 'data/btc_usdt_1m_processed.csv'
    output_csv = 'data/btc_usdt_1m_features_v2.csv'
    
    df = add_all_features(input_csv, output_csv)
    
    print("\n" + "="*60)
    print("📊 特征统计")
    print("="*60)
    print(df.describe())
    

if __name__ == '__main__':
    main()

