"""
CNN Teacher训练脚本 - Mac Mini M4优化版
针对Mac Mini M4的高性能配置进行了优化

主要改进:
- Batch size: 256 -> 512 (利用更大内存)
- Num workers: 4 -> 8 (利用更多CPU核心)
- Epochs: 50 -> 100 (更充分训练)
- 添加了混合精度训练支持(可选)

目标:
- Test Accuracy ≥ 60%
- Test F1 Score ≥ 60%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
from tqdm import tqdm
import json
import platform

from models.cnn_teacher import CNNTeacher
from data.dataset import create_dataloaders


class CNNTrainer:
    """CNN Teacher训练器 - M4优化版"""

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 device='cuda',
                 lr=1e-3,
                 weight_decay=1e-4,
                 label_smoothing=0.1,
                 use_amp=False):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.use_amp = use_amp

        # 优化器和损失函数
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        # 混合精度训练
        self.scaler = torch.amp.GradScaler('cpu') if use_amp else None

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'lr': []
        }

        self.best_val_f1 = 0.0
        self.best_model_path = None

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # 混合精度训练
            if self.use_amp:
                with torch.amp.autocast('cpu'):
                    output = self.model(features)
                    logits = output['logits']
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准训练
                output = self.model(features)
                logits = output['logits']
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def evaluate(self, data_loader, desc='Evaluating'):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(data_loader, desc=desc):
                features = features.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                output = self.model(features)
                logits = output['logits']
                loss = self.criterion(logits, labels)

                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))

        # 计算F1 score
        f1 = f1_score(all_labels, all_predictions, average='macro')

        return avg_loss, accuracy, f1, all_predictions, all_labels

    def train(self, epochs=100, early_stop_patience=15, save_dir='models'):
        """
        完整训练流程

        Args:
            epochs: 最大训练轮数 (M4优化: 50->100)
            early_stop_patience: 早停耐心值 (M4优化: 10->15)
            save_dir: 模型保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("="*60)
        print("开始训练CNN Teacher模型 (Mac Mini M4优化版)")
        print("="*60)
        if self.use_amp:
            print("✅ 混合精度训练已启用")

        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 60)

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc, val_f1, _, _ = self.evaluate(
                self.val_loader, desc='Validating'
            )

            # 更新学习率
            self.scheduler.step(val_f1)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)

            # 打印统计
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
            print(f"LR: {current_lr:.6f}")

            # 保存最佳模型
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0

                self.best_model_path = save_dir / 'cnn_teacher_best_m4.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                }, self.best_model_path)

                print(f"✅ 最佳模型已保存 (F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"⚠️  验证F1未提升 ({patience_counter}/{early_stop_patience})")

            # 早停
            if patience_counter >= early_stop_patience:
                print(f"\n早停触发！最佳epoch: {best_epoch}")
                break

        print("\n" + "="*60)
        print("训练完成！")
        print(f"最佳验证F1: {self.best_val_f1:.4f} (Epoch {best_epoch})")
        print("="*60)

    def test(self):
        """测试最佳模型"""
        if self.best_model_path is None:
            print("⚠️  没有找到最佳模型，跳过测试")
            return

        # 加载最佳模型
        checkpoint = torch.load(self.best_model_path)
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
        print(f"  F1 Score: {test_f1:.4f}")

        # 分类报告
        print("\n分类报告:")
        print(classification_report(
            labels, predictions,
            target_names=['SELL', 'HOLD', 'BUY'],
            digits=4
        ))

        # 保存混淆矩阵
        self.save_confusion_matrix(labels, predictions)

        # 保存训练曲线
        self.save_training_curves()

        # 保存训练历史
        self.save_history()

        return test_acc, test_f1

    def save_confusion_matrix(self, labels, predictions, save_path='results/figures/cnn_confusion_matrix_m4.png'):
        """保存混淆矩阵"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['SELL', 'HOLD', 'BUY'],
                    yticklabels=['SELL', 'HOLD', 'BUY'])
        plt.title('CNN Teacher Confusion Matrix (Mac Mini M4)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"\n✅ 混淆矩阵已保存: {save_path}")

    def save_training_curves(self, save_path='results/figures/cnn_training_curves_m4.png'):
        """保存训练曲线"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy曲线
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Val')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1 Score曲线
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', color='green')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning Rate曲线
        axes[1, 1].plot(self.history['lr'], label='LR', color='red')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✅ 训练曲线已保存: {save_path}")

    def save_history(self, save_path='results/logs/cnn_training_history_m4.json'):
        """保存训练历史"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"✅ 训练历史已保存: {save_path}")


def main():
    """主函数 - Mac Mini M4优化版"""
    # 检测设备
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ 检测到Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("✅ 检测到CUDA GPU")
    else:
        device = 'cpu'
        print("⚠️  使用CPU训练")

    print(f"使用设备: {device}")
    print(f"系统平台: {platform.platform()}")
    print(f"处理器: {platform.processor()}")

    # 创建DataLoader - M4优化配置
    print("\n加载数据...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv='data/train.csv',
        val_csv='data/val.csv',
        test_csv='data/test.csv',
        batch_size=512,      # M4优化: 256 -> 512
        seq_length=60,
        num_workers=8        # M4优化: 4 -> 8
    )

    print(f"✅ Batch size: 512 (针对M4大内存优化)")
    print(f"✅ Num workers: 8 (充分利用M4多核)")

    # 创建模型
    print("\n创建模型...")
    model = CNNTeacher(
        input_channels=7,
        seq_length=60,
        num_classes=3,
        dropout_rate=0.3
    )
    model.print_architecture()

    # 创建训练器
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.1,
        use_amp=False  # 可以在M4上启用混合精度训练
    )

    # 训练 - M4优化配置
    trainer.train(
        epochs=100,              # M4优化: 50 -> 100
        early_stop_patience=15,  # M4优化: 10 -> 15
        save_dir='models'
    )

    # 测试
    test_acc, test_f1 = trainer.test()

    # 检查是否达标
    print("\n" + "="*60)
    if test_acc >= 60.0 and test_f1 >= 0.60:
        print("✅ 训练成功！CNN Teacher达到目标性能")
    else:
        print("⚠️  未达到目标性能，建议:")
        print("  1. 调整超参数（学习率、batch size等）")
        print("  2. 增加模型容量")
        print("  3. 调整标签阈值")
        print("  4. 使用更多数据")
    print("="*60)


if __name__ == '__main__':
    main()
