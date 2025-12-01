"""
VeriRegime训练模块
提供可复用的模型架构、数据加载和训练组件
"""

from .models import CNNVolatility
from .dataset import VolatilityDataset, create_dataloaders
from .trainer import Trainer

__all__ = ['CNNVolatility', 'VolatilityDataset', 'create_dataloaders', 'Trainer']

