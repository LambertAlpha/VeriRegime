"""
VeriRegime训练模块
提供可复用的模型架构、数据加载和训练组件
"""

from .models import CNNVolatility, MLPStudent
from .dataset import VolatilityDataset, create_dataloaders
from .trainer import Trainer
from .distillation import DistillationTrainer, DistillationLoss

__all__ = [
    'CNNVolatility', 'MLPStudent',
    'VolatilityDataset', 'create_dataloaders',
    'Trainer', 'DistillationTrainer', 'DistillationLoss'
]

