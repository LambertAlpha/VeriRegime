"""
模型架构定义
包含CNN波动率预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNVolatility(nn.Module):
    """
    CNN波动率预测模型
    二分类：低波动(0) vs 高波动(1)
    
    架构:
        输入 (60分钟, 7特征)
        → Conv1D(7→64) + BN + ReLU + MaxPool
        → Conv1D(64→128) + BN + ReLU + MaxPool
        → GlobalAvgPool
        → FC(128→64) + ReLU
        → FC(64→2)
    """
    
    def __init__(self, input_channels=7, seq_length=60, dropout_rate=0.3):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        
        # 卷积层1
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 卷积层2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        """前向传播"""
        # (batch, seq, features) → (batch, features, seq)
        x = x.permute(0, 2, 1)
        
        # 卷积块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 卷积块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 全局池化
        x = self.global_pool(x).squeeze(-1)
        
        # 全连接
        x = self.fc1(x)
        features = F.relu(x)
        x = self.dropout_fc(features)
        logits = self.fc2(x)
        
        return {'logits': logits, 'features': features}
    
    def predict(self, x):
        """预测类别和概率"""
        with torch.no_grad():
            output = self.forward(x)
            probs = F.softmax(output['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


class MLPStudent(nn.Module):
    """
    MLP Student模型（用于知识蒸馏）
    二分类：低波动(0) vs 高波动(1)
    
    架构:
        输入 (240分钟 × 7特征 = 1680维)
        → FC(1680→256) + ReLU + Dropout
        → FC(256→128) + ReLU + Dropout
        → FC(128→64) + ReLU + Dropout
        → FC(64→2)
    
    设计原则:
        - 简单3层MLP（后续改为多项式激活）
        - 参数量 << CNN Teacher
        - 适合zkML证明生成
    """
    
    def __init__(self, input_dim=1680, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super().__init__()
        
        self.input_dim = input_dim  # 240 * 7
        
        # 构建全连接层
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(in_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        # (batch, seq_len, features) → (batch, seq_len * features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平
        
        logits = self.network(x)
        
        return {'logits': logits}
    
    def predict(self, x):
        """预测类别和概率"""
        with torch.no_grad():
            output = self.forward(x)
            probs = F.softmax(output['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs

