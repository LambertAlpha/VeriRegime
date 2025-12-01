"""
训练器模块
封装训练/验证/测试流程
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, test_loader,
                 device, class_weights, config):
        """
        Args:
            model: PyTorch模型
            train_loader, val_loader, test_loader: 数据加载器
            device: 训练设备 (cpu/cuda/mps)
            class_weights: 类别权重
            config: 配置字典
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # 学习率调度
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 5)
        )
        
        # 训练状态
        self.best_f1 = 0
        self.best_epoch = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for features, labels in pbar:
            features, labels = features.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(features)
            loss = self.criterion(output['logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(output['logits'], dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{correct/total*100:.2f}%'})
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc='Validating'):
                features, labels = features.to(self.device), labels.to(self.device)
                
                output = self.model(features)
                loss = self.criterion(output['logits'], labels)
                total_loss += loss.item()
                
                preds = torch.argmax(output['logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, f1
    
    def train(self, epochs, early_stop_patience=10, save_dir='results/checkpoints'):
        """完整训练流程"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        no_improve = 0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 60)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            
            self.scheduler.step(val_f1)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            print(f"训练 - Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
            print(f"验证 - Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | F1: {val_f1:.4f}")
            
            # 保存最佳模型
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_epoch = epoch
                best_path = save_dir / 'best_model.pth'
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': self.best_f1,
                    'history': self.history
                }, best_path)
                
                print(f"✅ 最佳模型已保存 (F1: {val_f1:.4f})")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    print(f"\n早停触发！最佳epoch: {self.best_epoch}")
                    break
        
        print(f"\n训练完成！最佳F1: {self.best_f1:.4f}")
        return self.best_f1
    
    def test(self, checkpoint_path='results/checkpoints/best_model.pth'):
        """测试最佳模型"""
        # 加载最佳模型
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for features, labels in tqdm(self.test_loader, desc='Testing'):
                features = features.to(self.device)
                output = self.model(features)
                preds = torch.argmax(output['logits'], dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 评估
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"\n{'='*60}")
        print("测试结果")
        print(f"{'='*60}")
        print(f"准确率: {acc*100:.2f}%")
        print(f"F1分数: {f1:.4f}")
        print(f"\n混淆矩阵:")
        print(f"              预测LOW  预测HIGH")
        print(f"真实LOW       {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"真实HIGH      {cm[1,0]:6d}    {cm[1,1]:6d}")
        print(f"\n分类报告:")
        print(classification_report(all_labels, all_preds, 
                                   target_names=['LOW', 'HIGH'], digits=4))
        
        # 可视化
        self.plot_results(cm)
        
        return acc, f1
    
    def plot_results(self, cm, save_path='results/figures/training_results.png'):
        """绘制训练曲线和混淆矩阵"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 训练曲线
        epochs = range(1, len(self.history['train_acc']) + 1)
        axes[0].plot(epochs, self.history['train_acc'], label='Train', marker='o')
        axes[0].plot(epochs, self.history['val_acc'], label='Val', marker='s')
        axes[0].axhline(y=0.6, color='r', linestyle='--', label='Target')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Training Curves')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                   xticklabels=['LOW', 'HIGH'], yticklabels=['LOW', 'HIGH'])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"✅ 结果图已保存: {save_path}")

