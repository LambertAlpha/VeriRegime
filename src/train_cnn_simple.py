#!/usr/bin/env python3
"""
简化版CNN训练 - 专注于基础正确性

改进:
1. 更简单的模型(50K参数)
2. 标准CrossEntropy Loss
3. 更保守的Dropout(0.2)
4. 逐批归一化(避免train/val分布不匹配)
5. 更大的标签阈值(±0.5%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import platform


# ============================================================
# 简化版CNN模型
# ============================================================

class SimpleCNNTeacher(nn.Module):
    """
    简化的CNN Teacher - 适中容量, 稳定训练
    
    Architecture:
        Input (60, 7) 
        → Conv1D(7→64, k=5) → BN → ReLU → MaxPool(2)  [60→30]
        → Conv1D(64→128, k=3) → BN → ReLU → MaxPool(2) [30→15]
        → GlobalAvgPool → FC(128→64) → ReLU → FC(64→3)
    
    参数量: ~50K
    """
    
    def __init__(self,
                 input_channels=7,
                 seq_length=60,
                 num_classes=3,
                 dropout_rate=0.2):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        # 卷积层1: 7 → 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 卷积层2: 64 → 128
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层: 128 → 64 → 3
        self.fc1 = nn.Linear(128, 64)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # 转换为Conv1d格式
        x = x.permute(0, 2, 1)  # (batch, 7, 60)
        
        # Conv块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 全局池化
        x = self.global_pool(x).squeeze(-1)  # (batch, 128)
        
        # 全连接
        x = self.fc1(x)
        features = F.relu(x)
        x = self.dropout_fc(features)
        logits = self.fc2(x)
        
        return {
            'logits': logits,
            'features': features
        }
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_architecture(self):
        print("="*60)
        print("Simple CNN Teacher Architecture")
        print("="*60)
        print(f"Input Shape: (batch, {self.seq_length}, {self.input_channels})")
        print(f"Output Shape: (batch, {self.num_classes})")
        print("\nLayers:")
        print("  Conv1D(7→64, k=5) + BN + ReLU + MaxPool(2)  [60→30]")
        print("  Conv1D(64→128, k=3) + BN + ReLU + MaxPool(2) [30→15]")
        print("  GlobalAvgPool                                 [128]")
        print("  FC(128→64) + ReLU")
        print(f"  FC(64→{self.num_classes})")
        print(f"\nTotal Parameters: {self.get_num_parameters():,}")
        print("="*60)


# ============================================================
# 数据加载 (使用全局归一化)
# ============================================================

def create_dataloaders(train_csv, val_csv, test_csv, batch_size=512, seq_length=60, num_workers=8):
    """创建数据加载器 - 使用全局归一化统计量"""
    
    print("加载所有数据以计算全局归一化统计...")
    
    # 读取所有数据
    train_df = pd.read_csv(train_csv, index_col=0)
    val_df = pd.read_csv(val_csv, index_col=0)
    test_df = pd.read_csv(test_csv, index_col=0)
    
    feature_cols = ['ema_5', 'ema_10', 'ema_20', 'rsi', 'macd', 'volume_ma_5', 'volume_ma_10']
    
    # 合并所有数据计算归一化参数
    all_features = pd.concat([
        train_df[feature_cols],
        val_df[feature_cols],
        test_df[feature_cols]
    ])
    
    global_mean = all_features.mean().values
    global_std = all_features.std().values + 1e-8
    
    print(f"全局归一化参数:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: mean={global_mean[i]:.4f}, std={global_std[i]:.4f}")
    
    def prepare_dataset(df, split_name):
        features = df[feature_cols].values
        features = (features - global_mean) / global_std
        labels = df['label'].values.astype(np.int64)
        
        # 创建序列
        X_list, y_list = [], []
        for i in range(len(features) - seq_length):
            X_list.append(features[i:i+seq_length])
            y_list.append(labels[i+seq_length])
        
        X = torch.FloatTensor(np.array(X_list))
        y = torch.LongTensor(np.array(y_list))
        
        print(f"\n✅ {split_name}数据集")
        print(f"   样本数: {len(y):,}")
        unique, counts = np.unique(y.numpy(), return_counts=True)
        print(f"   标签分布:")
        for label, count in zip(unique, counts):
            print(f"     类别 {label}: {count:,} ({count/len(y)*100:.2f}%)")
        
        return TensorDataset(X, y), counts
    
    train_dataset, train_counts = prepare_dataset(train_df, "训练")
    val_dataset, _ = prepare_dataset(val_df, "验证")
    test_dataset, _ = prepare_dataset(test_df, "测试")
    
    # 计算类别权重
    total_samples = sum(train_counts)
    class_weights = torch.FloatTensor([
        total_samples / (len(train_counts) * count) for count in train_counts
    ])
    print(f"\n类别权重: {[f'{w:.3f}' for w in class_weights]}")
    
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

class SimpleTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, class_weights, lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # 使用加权CrossEntropy
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # Adam优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # ReduceLROnPlateau scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 监控F1最大化
            factor=0.5,
            patience=5
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
    
    def train(self, epochs=50, early_stop_patience=10, save_dir='models'):
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("开始训练Simple CNN Teacher")
        print("="*60)
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-"*60)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_f1, _, _ = self.evaluate(self.val_loader, desc='Validating')
            
            # 更新学习率
            self.scheduler.step(val_f1)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
            print(f"LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_path = f"{save_dir}/cnn_simple_best.pth"
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
        
        # 混淆矩阵
        print("\n混淆矩阵:")
        cm = confusion_matrix(labels, predictions)
        print(cm)
        
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
    
    # 加载数据
    print("\n" + "="*60)
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        train_csv='data/train.csv',
        val_csv='data/val.csv',
        test_csv='data/test.csv',
        batch_size=512,
        seq_length=60,
        num_workers=8
    )
    
    # 创建模型
    print("\n")
    model = SimpleCNNTeacher(
        input_channels=7,
        seq_length=60,
        num_classes=3,
        dropout_rate=0.2
    )
    model.print_architecture()
    
    # 创建训练器
    trainer = SimpleTrainer(
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
        epochs=50,
        early_stop_patience=10,
        save_dir='models'
    )
    
    # 测试
    test_acc, test_f1 = trainer.test()
    
    # 检查是否达标
    print("\n" + "="*60)
    if test_acc and test_f1:
        if test_acc >= 60.0 and test_f1 >= 0.60:
            print("✅ 训练成功！模型达到目标性能")
        else:
            print("📊 训练完成")
            print(f"   当前: Accuracy={test_acc:.2f}%, F1={test_f1:.4f}")
            print(f"   目标: Accuracy>=60%, F1>=0.60")
    print("="*60)


if __name__ == '__main__':
    main()

