"""
知识蒸馏训练器
实现从CNN Teacher到MLP Student的知识转移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import os


# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    
    组合两种损失：
    1. 蒸馏损失（KL散度）：学习Teacher的软标签
    2. 硬标签损失（交叉熵）：学习真实标签
    
    总损失 = alpha * 蒸馏损失 + (1 - alpha) * 硬标签损失
    """
    
    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Args:
            temperature: 温度参数（T越大，分布越平滑）
            alpha: 蒸馏损失权重（0-1之间）
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        计算蒸馏损失
        
        Args:
            student_logits: Student模型输出 (batch, num_classes)
            teacher_logits: Teacher模型输出 (batch, num_classes)
            labels: 真实标签 (batch,)
        
        Returns:
            total_loss: 总损失
            distill_loss: 蒸馏损失（用于监控）
            hard_loss: 硬标签损失（用于监控）
        """
        # 1. 蒸馏损失（KL散度）
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distill_loss = F.kl_div(
            student_soft, 
            teacher_soft, 
            reduction='batchmean'
        ) * (self.temperature ** 2)  # 温度平方缩放
        
        # 2. 硬标签损失（交叉熵）
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 3. 组合损失
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, distill_loss, hard_loss


class DistillationTrainer:
    """知识蒸馏训练器"""
    
    def __init__(self, teacher_model, student_model, train_loader, val_loader, 
                 test_loader, device, config):
        """
        Args:
            teacher_model: 预训练的CNN Teacher
            student_model: 待训练的MLP Student
            train_loader, val_loader, test_loader: 数据加载器
            device: 训练设备
            config: 训练配置字典
        """
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Teacher设为评估模式（不更新）
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 蒸馏损失
        self.criterion = DistillationLoss(
            temperature=config.get('temperature', 4.0),
            alpha=config.get('alpha', 0.7)
        )
        
        # 优化器
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, 
            patience=5
        )
        
        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        self.best_f1 = 0.0
        # 使用项目根目录的绝对路径
        self.checkpoint_dir = PROJECT_ROOT / 'results' / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.student.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Teacher前向（不计算梯度）
            with torch.no_grad():
                teacher_output = self.teacher(features)
                teacher_logits = teacher_output['logits']
            
            # Student前向
            student_output = self.student(features)
            student_logits = student_output['logits']
            
            # 计算蒸馏损失
            loss, distill_loss, hard_loss = self.criterion(
                student_logits, teacher_logits, labels
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(student_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            acc = (preds == labels).float().mean().item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'distill': f'{distill_loss.item():.4f}',
                'hard': f'{hard_loss.item():.4f}',
                'acc': f'{acc*100:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = accuracy_score(all_labels, all_preds)
        
        return avg_loss, avg_acc
    
    def validate(self):
        """验证"""
        self.student.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating')
            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Teacher和Student前向
                teacher_output = self.teacher(features)
                student_output = self.student(features)
                
                loss, _, _ = self.criterion(
                    student_output['logits'],
                    teacher_output['logits'],
                    labels
                )
                
                total_loss += loss.item()
                preds = torch.argmax(student_output['logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = accuracy_score(all_labels, all_preds)
        avg_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, avg_acc, avg_f1
    
    def test(self):
        """测试最佳模型"""
        # 加载最佳模型
        checkpoint = torch.load(
            self.checkpoint_dir / 'best_student.pth'
        )
        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.student.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for features, labels in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                output = self.student(features)
                preds = torch.argmax(output['logits'], dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return acc, f1, all_preds, all_labels
    
    def train(self, epochs, early_stop_patience=10):
        """完整训练流程"""
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 60)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_f1 = self.validate()
            
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
                best_epoch = epoch
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': self.best_f1,
                    'history': self.history
                }, self.checkpoint_dir / 'best_student.pth')
                
                print(f"✅ 最佳模型已保存 (F1: {val_f1:.4f})")
            else:
                patience_counter += 1
            
            # 学习率调度
            self.scheduler.step(val_f1)
            
            # 早停
            if patience_counter >= early_stop_patience:
                print(f"\n⚠️ 早停触发（{early_stop_patience}轮无提升）")
                print(f"最佳F1: {self.best_f1:.4f} (Epoch {best_epoch})")
                break
        
        return self.best_f1

