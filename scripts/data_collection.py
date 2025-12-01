"""
数据收集脚本：从Binance获取BTC/USDT 1分钟K线数据
用于VeriRegime项目的训练数据准备
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pathlib import Path

def fetch_binance_ohlcv(symbol='BTC/USDT', timeframe='1m', start_date='2023-01-01', end_date='2024-12-31'):
    """
    从Binance获取OHLCV数据

    Args:
        symbol: 交易对
        timeframe: 时间周期 (1m, 5m, 1h等)
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}  # 使用永续合约获取funding rate
    })

    # 转换日期为时间戳
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    all_data = []
    current_ts = start_ts

    print(f"开始获取 {symbol} 数据，从 {start_date} 到 {end_date}")

    while current_ts < end_ts:
        try:
            # 每次请求1000条数据（Binance限制）
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)

            if not ohlcv:
                break

            all_data.extend(ohlcv)
            current_ts = ohlcv[-1][0] + 1  # 更新时间戳

            print(f"已获取 {len(all_data)} 条数据... 当前时间: {datetime.fromtimestamp(current_ts/1000)}")
            time.sleep(exchange.rateLimit / 1000)  # 遵守rate limit

        except Exception as e:
            print(f"错误: {e}")
            time.sleep(60)  # 出错等待1分钟后重试
            continue

    # 转换为DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

    print(f"\n✅ 数据获取完成！总共 {len(df)} 条记录")
    print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")

    return df

def calculate_features(df):
    """
    计算技术指标特征

    Features (8维):
    - EMA(5), EMA(10), EMA(20)
    - RSI(14)
    - MACD
    - Volume MA(5), Volume MA(10)
    - (Funding rate需要单独API获取)
    """
    import pandas_ta as ta

    print("\n开始计算技术指标...")

    # EMA
    df['ema_5'] = ta.ema(df['close'], length=5)
    df['ema_10'] = ta.ema(df['close'], length=10)
    df['ema_20'] = ta.ema(df['close'], length=20)

    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14)

    # MACD (使用histogram作为特征)
    macd_result = ta.macd(df['close'])
    df['macd'] = macd_result['MACDh_12_26_9']

    # Volume MA
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()

    # 计算未来1小时收益率（用于生成标签）
    df['future_return_1h'] = df['close'].pct_change(60).shift(-60) * 100  # 转换为百分比

    print("✅ 特征计算完成")

    return df

def generate_labels(df, buy_threshold=2.0, sell_threshold=-2.0):
    """
    生成交易信号标签

    Args:
        buy_threshold: 涨幅超过此值标记为BUY (%)
        sell_threshold: 跌幅超过此值标记为SELL (%)

    Returns:
        df with 'label' column: 0=SELL, 1=HOLD, 2=BUY
    """
    def classify(return_pct):
        if pd.isna(return_pct):
            return np.nan
        elif return_pct > buy_threshold:
            return 2  # BUY
        elif return_pct < sell_threshold:
            return 0  # SELL
        else:
            return 1  # HOLD

    df['label'] = df['future_return_1h'].apply(classify)

    # 统计标签分布
    label_counts = df['label'].value_counts().sort_index()
    print("\n标签分布:")
    print(f"  SELL (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.2f}%)")
    print(f"  HOLD (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.2f}%)")
    print(f"  BUY  (2): {label_counts.get(2, 0)} ({label_counts.get(2, 0)/len(df)*100:.2f}%)")

    return df

def save_data(df, output_path='data/btc_usdt_1m.csv'):
    """保存处理后的数据"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path)
    print(f"\n✅ 数据已保存到 {output_path}")
    print(f"   数据形状: {df.shape}")
    print(f"   特征列: {list(df.columns)}")

if __name__ == '__main__':
    # 1. 获取原始数据
    df = fetch_binance_ohlcv(
        symbol='BTC/USDT',
        timeframe='1m',
        start_date='2023-01-01',
        end_date='2024-11-08'
    )

    # 2. 计算特征
    df = calculate_features(df)

    # 3. 生成标签
    df = generate_labels(df, buy_threshold=2.0, sell_threshold=-2.0)

    # 4. 删除NaN行（前期指标计算会产生NaN）
    df_clean = df.dropna()
    print(f"\n清理后数据: {len(df_clean)} 条（删除了 {len(df) - len(df_clean)} 条NaN）")

    # 5. 保存
    save_data(df_clean, output_path='data/btc_usdt_1m_processed.csv')

    print("\n" + "="*60)
    print("🎉 数据准备完成！现在可以开始训练模型了")
    print("="*60)
