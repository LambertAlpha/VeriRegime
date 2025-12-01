"""
数据加载模块
处理波动率预测数据集
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np


class VolatilityDataset:
    """波动率数据集加载器"""
    
    def __init__(self, csv_path, feature_cols, seq_length=60):
        """
        Args:
            csv_path: CSV文件路径
            feature_cols: 特征列名列表
            seq_length: 序列长度（默认60分钟）
        """
        self.csv_path = csv_path
        self.feature_cols = feature_cols
        self.seq_length = seq_length
        
        # 读取数据
        df = pd.read_csv(csv_path, index_col=0)
        
        # 提取特征和标签
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.int64)
    
    def normalize(self, mean, std):
        """归一化特征"""
        self.features = (self.features - mean) / (std + 1e-8)
    
    def create_sequences(self):
        """创建滑动窗口序列"""
        X_list, y_list = [], []
        
        for i in range(len(self.features) - self.seq_length):
            X_list.append(self.features[i:i+self.seq_length])
            y_list.append(self.labels[i+self.seq_length])
        
        X = torch.FloatTensor(np.array(X_list))
        y = torch.LongTensor(np.array(y_list))
        
        return TensorDataset(X, y)


def create_dataloaders(train_csv, val_csv, test_csv, 
                      feature_cols, seq_length=60, batch_size=512, num_workers=8):
    """
    创建训练/验证/测试数据加载器
    
    Args:
        train_csv, val_csv, test_csv: 数据文件路径
        feature_cols: 特征列名
        seq_length: 序列长度
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader, test_loader, class_weights, stats
    """
    # 加载数据集
    train_dataset = VolatilityDataset(train_csv, feature_cols, seq_length)
    val_dataset = VolatilityDataset(val_csv, feature_cols, seq_length)
    test_dataset = VolatilityDataset(test_csv, feature_cols, seq_length)
    
    # 计算全局归一化参数
    all_features = np.vstack([
        train_dataset.features,
        val_dataset.features,
        test_dataset.features
    ])
    global_mean = all_features.mean(axis=0)
    global_std = all_features.std(axis=0)
    
    # 归一化
    train_dataset.normalize(global_mean, global_std)
    val_dataset.normalize(global_mean, global_std)
    test_dataset.normalize(global_mean, global_std)
    
    # 创建序列
    train_data = train_dataset.create_sequences()
    val_data = val_dataset.create_sequences()
    test_data = test_dataset.create_sequences()
    
    # 计算类别权重
    _, train_labels = train_data[:]
    unique, counts = np.unique(train_labels.numpy(), return_counts=True)
    total = len(train_labels)
    class_weights = torch.FloatTensor([total / (len(unique) * c) for c in counts])
    
    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    
    # 统计信息
    stats = {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'feature_mean': global_mean,
        'feature_std': global_std,
        'class_weights': class_weights
    }
    
    return train_loader, val_loader, test_loader, class_weights, stats

