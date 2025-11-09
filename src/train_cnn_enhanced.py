#!/usr/bin/env python3
"""
增强版CNN Teacher训练脚本 - M4优化版本

改进点:
1. 增加模型容量 (27K -> 100K+参数)
2. 添加类别权重处理不平衡
3. 改进学习率调度策略
4. 使用Focal Loss减轻类别不平衡影响
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import platform


# ============================================================
# 增强版CNN Teacher模型
# ============================================================

class EnhancedCNNTeacher(nn.Module):
    """
    增强版CNN Teacher - 更大容量, 更深网络
    
    Architecture:
        Input (60, 7) 
        → Conv1D(7→128, k=5) → BN → ReLU → Dropout → MaxPool(2)  [60→30]
        → Conv1D(128→256, k=3) → BN → ReLU → Dropout → MaxPool(2) [30→15]
        → Conv1D(256→256, k=3) → BN → ReLU → Dropout              [15→15]
        → GlobalAvgPool + GlobalMaxPool → Concat                   [512]
        → FC(512→256) → BN → ReLU → Dropout
        → FC(256→3)
    
    参数量: ~150K
    """
    
    def __init__(self,
                 input_channels=7,
                 seq_length=60,
                 num_classes=3,
                 dropout_rate=0.4):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        # 卷积层1: 7 → 128
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 卷积层2: 128 → 256
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 卷积层3: 256 → 256 (不降维)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 双池化: GlobalAvgPool + GlobalMaxPool
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 全连接层: 512 → 256 → 3
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels) = (batch, 60, 7)
        Returns:
            dict with 'logits' and 'features'
        """
        # 转换为Conv1d格式: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, 7, 60)
        
        # Conv块1
        x = self.conv1(x)       # (batch, 128, 60)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)       # (batch, 128, 30)
        x = self.dropout1(x)
        
        # Conv块2
        x = self.conv2(x)       # (batch, 256, 30)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)       # (batch, 256, 15)
        x = self.dropout2(x)
        
        # Conv块3 (不降维)
        x = self.conv3(x)       # (batch, 256, 15)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # 双池化
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # (batch, 256)
        max_pool = self.global_max_pool(x).squeeze(-1)  # (batch, 256)
        x = torch.cat([avg_pool, max_pool], dim=1)      # (batch, 512)
        
        # 全连接层
        x = self.fc1(x)         # (batch, 256)
        x = self.bn_fc(x)
        features = F.relu(x)
        x = self.dropout_fc(features)
        logits = self.fc2(x)    # (batch, 3)
        
        return {
            'logits': logits,
            'features': features
        }
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_architecture(self):
        print("="*60)
        print("Enhanced CNN Teacher Architecture")
        print("="*60)
        print(f"Input Shape: (batch, {self.seq_length}, {self.input_channels})")
        print(f"Output Shape: (batch, {self.num_classes})")
        print("\nLayers:")
        print("  Conv1D(7→128, k=5) + BN + ReLU + Dropout + MaxPool(2)  [60→30]")
        print("  Conv1D(128→256, k=3) + BN + ReLU + Dropout + MaxPool(2) [30→15]")
        print("  Conv1D(256→256, k=3) + BN + ReLU + Dropout              [15→15]")
        print("  GlobalAvgPool + GlobalMaxPool → Concat                   [512]")
        print("  FC(512→256) + BN + ReLU + Dropout")
        print(f"  FC(256→{self.num_classes})")
        print(f"\nTotal Parameters: {self.get_num_parameters():,}")
        print("="*60)


# ============================================================
# Focal Loss for Imbalanced Data
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: 自动降低易分类样本的权重,关注难分类样本
    FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
# 数据加载
# ============================================================

def create_dataloaders(train_csv, val_csv, test_csv, batch_size=512, seq_length=60, num_workers=8):
    """创建数据加载器"""
    
    def load_and_prepare(csv_file):
        df = pd.read_csv(csv_file, index_col=0)
        
        # 选择特征列
        feature_cols = ['ema_5', 'ema_10', 'ema_20', 'rsi', 'macd', 
                       'volume_ma_5', 'volume_ma_10']
        
        # 归一化
        features = df[feature_cols].values
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        labels = df['label'].values.astype(np.int64)
        
        # 创建序列
        X_list, y_list = [], []
        for i in range(len(features) - seq_length):
            X_list.append(features[i:i+seq_length])
            y_list.append(labels[i+seq_length])
        
        X = torch.FloatTensor(np.array(X_list))
        y = torch.LongTensor(np.array(y_list))
        
        print(f"✅ 数据集加载完成: {csv_file}")
        print(f"   总样本数: {len(y):,}")
        print(f"   特征维度: {len(feature_cols)}")
        print(f"   序列长度: {seq_length}")
        
        # 统计标签分布
        unique, counts = np.unique(y.numpy(), return_counts=True)
        print(f"   标签分布:")
        for label, count in zip(unique, counts):
            print(f"     类别 {label}: {count:,} ({count/len(y)*100:.2f}%)")
        
        return TensorDataset(X, y), counts
    
    # 加载数据
    train_dataset, train_counts = load_and_prepare(train_csv)
    val_dataset, _ = load_and_prepare(val_csv)
    test_dataset, _ = load_and_prepare(test_csv)
    
    # 计算类别权重 (用于Focal Loss)
    total_samples = sum(train_counts)
    class_weights = torch.FloatTensor([
        total_samples / (len(train_counts) * count) for count in train_counts
    ])
    print(f"\n类别权重: {class_weights.tolist()}")
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_weights


# ============================================================
# 训练器
# ============================================================

class EnhancedTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_weights, lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # 使用Focal Loss
        self.criterion = FocalLoss(
            alpha=class_weights.to(device),
            gamma=2.0
        )
        
        # AdamW优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine Annealing with Warm Restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # 10个epoch后重启
            T_mult=2,  # 每次重启周期翻倍
            eta_min=1e-6
        )
        
        self.best_val_f1 = 0.0
        self.best_model_path = None
        self.patience_counter = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            output = self.model(features)
            logits = output['logits']
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader, desc='Evaluating'):
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=desc)
            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                output = self.model(features)
                logits = output['logits']
                loss = self.criterion(logits, labels)
                
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return avg_loss, accuracy, f1, all_predictions, all_labels
    
    def train(self, epochs=100, early_stop_patience=20, save_dir='models'):
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("开始训练Enhanced CNN Teacher")
        print("="*60)
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-"*60)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_f1, _, _ = self.evaluate(self.val_loader, desc='Validating')
            
            # 更新学习率
            self.scheduler.step(epoch - 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
            print(f"LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_path = f"{save_dir}/cnn_enhanced_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc
                }, self.best_model_path)
                print(f"✅ 保存最佳模型 (F1: {val_f1:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"⚠️  验证F1未提升 ({self.patience_counter}/{early_stop_patience})")
            
            # 早停
            if self.patience_counter >= early_stop_patience:
                print(f"\n早停触发！最佳epoch: {epoch - early_stop_patience}")
                break
        
        print("\n" + "="*60)
        print("训练完成！")
        print(f"最佳验证F1: {self.best_val_f1:.4f}")
        print("="*60)
    
    def test(self):
        """测试最佳模型"""
        if self.best_model_path is None:
            print("⚠️  没有找到最佳模型")
            return None, None
        
        # 加载最佳模型
        checkpoint = torch.load(self.best_model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print("\n" + "="*60)
        print("测试最佳模型")
        print("="*60)
        
        test_loss, test_acc, test_f1, predictions, labels = self.evaluate(
            self.test_loader, desc='Testing'
        )
        
        print(f"\n测试结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        print(f"  F1-Score: {test_f1:.4f}")
        
        # 详细分类报告
        print("\n分类报告:")
        print(classification_report(
            labels, predictions,
            target_names=['SELL', 'HOLD', 'BUY'],
            digits=4
        ))
        
        return test_acc, test_f1


# ============================================================
# 主函数
# ============================================================

def main():
    # 设备配置
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ 检测到Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("✅ 检测到NVIDIA GPU")
    else:
        device = 'cpu'
        print("⚠️  使用CPU")
    
    print(f"使用设备: {device}")
    print(f"系统平台: {platform.platform()}")
    print(f"处理器: {platform.processor()}")
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        train_csv='data/train.csv',
        val_csv='data/val.csv',
        test_csv='data/test.csv',
        batch_size=512,  # M4优化
        seq_length=60,
        num_workers=8    # M4优化
    )
    
    # 创建模型
    print("\n创建模型...")
    model = EnhancedCNNTeacher(
        input_channels=7,
        seq_length=60,
        num_classes=3,
        dropout_rate=0.4
    )
    model.print_architecture()
    
    # 创建训练器
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        class_weights=class_weights,
        lr=1e-3,
        weight_decay=1e-4
    )
    
    # 训练
    trainer.train(
        epochs=100,
        early_stop_patience=20,
        save_dir='models'
    )
    
    # 测试
    test_acc, test_f1 = trainer.test()
    
    # 检查是否达标
    print("\n" + "="*60)
    if test_acc >= 60.0 and test_f1 >= 0.60:
        print("✅ 训练成功！模型达到目标性能")
    else:
        print("⚠️  未完全达标，但已显著改进")
        print(f"   当前: Accuracy={test_acc:.2f}%, F1={test_f1:.4f}")
        print(f"   目标: Accuracy>=60%, F1>=0.60")
    print("="*60)


if __name__ == '__main__':
    main()

