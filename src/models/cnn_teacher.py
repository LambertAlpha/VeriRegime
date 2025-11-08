"""
CNN Teacher Model
1D CNN用于时间序列特征提取和交易信号分类

Architecture:
    X ∈ R^(60×8) → Conv1D(64, k=5) → ReLU → MaxPool(2)
                 → Conv1D(128, k=3) → ReLU → MaxPool(2)
                 → GlobalAvgPool → FC(128→3) → Softmax

Input: (batch_size, 60, 8) - 60个时间步，8维特征
Output: (batch_size, 3) - 3类分类概率 [SELL, HOLD, BUY]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTeacher(nn.Module):
    """
    CNN Teacher模型用于金融时间序列分类

    Args:
        input_channels: 输入特征维度 (默认8)
        seq_length: 输入序列长度 (默认60)
        num_classes: 分类数量 (默认3: SELL/HOLD/BUY)
        dropout_rate: Dropout比例 (默认0.3)
    """

    def __init__(self,
                 input_channels=8,
                 seq_length=60,
                 num_classes=3,
                 dropout_rate=0.3):
        super(CNNTeacher, self).__init__()

        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes

        # 第一卷积层: 8 → 64
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2  # 保持序列长度
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 60 → 30
        self.dropout1 = nn.Dropout(dropout_rate)

        # 第二卷积层: 64 → 128
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 30 → 15
        self.dropout2 = nn.Dropout(dropout_rate)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 15 → 1

        # 全连接层
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch_size, seq_length, input_channels)
               例如: (256, 60, 8)

        Returns:
            logits: (batch_size, num_classes)
            features: 中间层特征（用于知识蒸馏）
        """
        # PyTorch Conv1d要求输入格式: (batch, channels, length)
        # 所以需要转置: (batch, seq_length, channels) → (batch, channels, seq_length)
        x = x.permute(0, 2, 1)  # (batch, 8, 60)

        # 第一卷积块
        x = self.conv1(x)       # (batch, 64, 60)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)       # (batch, 64, 30)
        x = self.dropout1(x)

        # 第二卷积块
        x = self.conv2(x)       # (batch, 128, 30)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)       # (batch, 128, 15)
        features_2d = x  # 保存用于蒸馏
        x = self.dropout2(x)

        # 全局平均池化
        x = self.global_pool(x)  # (batch, 128, 1)
        x = x.squeeze(-1)        # (batch, 128)

        # 全连接层
        logits = self.fc(x)      # (batch, 3)

        # 返回logits和中间特征（用于知识蒸馏）
        return {
            'logits': logits,
            'features': x,  # 全连接前的特征
            'conv_features': features_2d  # 卷积层特征
        }

    def predict(self, x):
        """
        预测类别

        Args:
            x: (batch_size, seq_length, input_channels)

        Returns:
            predictions: (batch_size,) - 预测的类别索引
            probabilities: (batch_size, num_classes) - 类别概率
        """
        with torch.no_grad():
            output = self.forward(x)
            logits = output['logits']
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions, probabilities

    def get_num_parameters(self):
        """返回模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_architecture(self):
        """打印模型架构摘要"""
        print("="*60)
        print("CNN Teacher Architecture")
        print("="*60)
        print(f"Input Shape: (batch_size, {self.seq_length}, {self.input_channels})")
        print(f"Output Shape: (batch_size, {self.num_classes})")
        print("\nLayers:")
        print(f"  Conv1D(8→64, k=5) + BN + ReLU + MaxPool(2)")
        print(f"  Conv1D(64→128, k=3) + BN + ReLU + MaxPool(2)")
        print(f"  GlobalAvgPool")
        print(f"  FC(128→{self.num_classes})")
        print(f"\nTotal Parameters: {self.get_num_parameters():,}")
        print("="*60)


def test_cnn_teacher():
    """测试CNN Teacher模型"""
    print("测试CNN Teacher模型...")

    # 创建模型
    model = CNNTeacher(
        input_channels=8,
        seq_length=60,
        num_classes=3,
        dropout_rate=0.3
    )

    model.print_architecture()

    # 创建测试输入
    batch_size = 16
    x = torch.randn(batch_size, 60, 8)

    print(f"\n测试输入形状: {x.shape}")

    # 前向传播
    output = model(x)
    logits = output['logits']
    features = output['features']

    print(f"输出logits形状: {logits.shape}")
    print(f"中间特征形状: {features.shape}")

    # 测试预测
    predictions, probabilities = model.predict(x)
    print(f"预测结果形状: {predictions.shape}")
    print(f"概率分布形状: {probabilities.shape}")
    print(f"预测类别: {predictions[:5]}")
    print(f"概率分布示例:\n{probabilities[:3]}")

    print("\n✅ CNN Teacher模型测试通过！")


if __name__ == '__main__':
    test_cnn_teacher()
