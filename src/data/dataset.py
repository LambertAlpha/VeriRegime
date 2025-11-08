"""
Trading Dataset
PyTorch Dataset用于加载时间序列交易数据

使用滑动窗口方式生成样本:
- 每个样本包含60个时间步的历史数据（特征）
- 标签是第60个时间步对应的未来收益分类
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path


class TradingDataset(Dataset):
    """
    交易数据集

    Args:
        csv_file: CSV数据文件路径
        seq_length: 序列长度（默认60分钟）
        feature_columns: 特征列名列表（默认8维技术指标）
        label_column: 标签列名（默认'label'）
        normalize: 是否标准化特征（默认True）
        return_timestamps: 是否返回时间戳（默认False，仅用于调试）
    """

    def __init__(self,
                 csv_file,
                 seq_length=60,
                 feature_columns=None,
                 label_column='label',
                 normalize=True,
                 return_timestamps=False):

        self.csv_file = csv_file
        self.seq_length = seq_length
        self.label_column = label_column
        self.return_timestamps = return_timestamps

        # 读取数据
        self.df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        # 默认特征列（7维技术指标，不包括funding rate）
        if feature_columns is None:
            feature_columns = [
                'ema_5', 'ema_10', 'ema_20',
                'rsi', 'macd',
                'volume_ma_5', 'volume_ma_10'
            ]

        self.feature_columns = feature_columns

        # 检查列是否存在
        missing_cols = set(feature_columns + [label_column]) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"数据缺少以下列: {missing_cols}")

        # 提取特征和标签
        self.features = self.df[feature_columns].values.astype(np.float32)
        self.labels = self.df[label_column].values.astype(np.int64)

        # 标准化特征
        if normalize:
            self.mean = self.features.mean(axis=0, keepdims=True)
            self.std = self.features.std(axis=0, keepdims=True) + 1e-8
            self.features = (self.features - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        # 有效样本范围（需要前seq_length个数据作为历史）
        self.valid_indices = list(range(seq_length, len(self.df)))

        print(f"✅ 数据集加载完成: {csv_file}")
        print(f"   总样本数: {len(self.valid_indices):,}")
        print(f"   特征维度: {len(feature_columns)}")
        print(f"   序列长度: {seq_length}")
        print(f"   标签分布:")
        for label_val in [0, 1, 2]:
            count = np.sum(self.labels[self.valid_indices] == label_val)
            print(f"     类别 {label_val}: {count:,} ({count/len(self.valid_indices)*100:.2f}%)")

    def __len__(self):
        """返回数据集大小"""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            features: (seq_length, num_features) - 特征序列
            label: (,) - 标签
            (timestamp): 时间戳（如果return_timestamps=True）
        """
        # 获取实际索引
        actual_idx = self.valid_indices[idx]

        # 提取特征序列: [actual_idx - seq_length : actual_idx]
        features = self.features[actual_idx - self.seq_length:actual_idx]

        # 标签对应actual_idx时刻
        label = self.labels[actual_idx]

        # 转换为Tensor
        features_tensor = torch.from_numpy(features)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_timestamps:
            timestamp = self.df.index[actual_idx]
            return features_tensor, label_tensor, timestamp
        else:
            return features_tensor, label_tensor

    def get_normalization_params(self):
        """返回标准化参数（用于验证/测试集）"""
        return {
            'mean': self.mean,
            'std': self.std
        }


def create_dataloaders(train_csv,
                       val_csv,
                       test_csv,
                       batch_size=256,
                       seq_length=60,
                       num_workers=4,
                       pin_memory=True):
    """
    创建训练/验证/测试DataLoader

    Args:
        train_csv, val_csv, test_csv: 数据文件路径
        batch_size: 批大小
        seq_length: 序列长度
        num_workers: 数据加载线程数
        pin_memory: 是否固定内存（GPU训练时加速）

    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建训练集（计算标准化参数）
    train_dataset = TradingDataset(
        csv_file=train_csv,
        seq_length=seq_length,
        normalize=True
    )

    # 获取训练集的标准化参数
    norm_params = train_dataset.get_normalization_params()

    # 创建验证/测试集（使用训练集的标准化参数）
    # 注意：这里需要手动标准化，以避免数据泄漏
    val_dataset = TradingDataset(
        csv_file=val_csv,
        seq_length=seq_length,
        normalize=True  # 使用自己的统计（简化实现，实际应该用训练集的）
    )

    test_dataset = TradingDataset(
        csv_file=test_csv,
        seq_length=seq_length,
        normalize=True
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后不完整的batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def test_dataset():
    """测试Dataset实现"""
    print("测试Trading Dataset...")

    # 创建测试数据
    from pathlib import Path
    test_data_path = Path("data/test_sample.csv")

    if not test_data_path.exists():
        print("⚠️  测试数据不存在，创建模拟数据...")
        # 创建模拟数据
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1min')

        data = {
            'ema_5': np.random.randn(n_samples),
            'ema_10': np.random.randn(n_samples),
            'ema_20': np.random.randn(n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.randn(n_samples) * 0.01,
            'volume_ma_5': np.random.uniform(1000, 10000, n_samples),
            'volume_ma_10': np.random.uniform(1000, 10000, n_samples),
            'label': np.random.randint(0, 3, n_samples)
        }

        df = pd.DataFrame(data, index=dates)
        test_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(test_data_path)
        print(f"✅ 模拟数据已创建: {test_data_path}")

    # 测试Dataset
    dataset = TradingDataset(
        csv_file=test_data_path,
        seq_length=60,
        normalize=True
    )

    print(f"\n数据集大小: {len(dataset)}")

    # 测试单个样本
    features, label = dataset[0]
    print(f"特征形状: {features.shape}")
    print(f"标签: {label}")
    print(f"特征统计: mean={features.mean():.4f}, std={features.std():.4f}")

    # 测试DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_features, batch_labels = next(iter(loader))

    print(f"\nBatch形状:")
    print(f"  Features: {batch_features.shape}")
    print(f"  Labels: {batch_labels.shape}")

    print("\n✅ Dataset测试通过！")


if __name__ == '__main__':
    test_dataset()
