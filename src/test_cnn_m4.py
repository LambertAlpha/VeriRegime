#!/usr/bin/env python3
"""
CNN Teacher M4 测试脚本
只加载已保存的最佳模型并在测试集上评估
"""

import torch
import platform
from train_cnn_m4 import CNNTeacher, create_dataloaders
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np


def test_model(model_path='models/cnn_teacher_best_m4.pth'):
    """加载保存的模型并测试"""
    
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
    
    # 加载测试数据
    print("\n加载测试数据...")
    _, _, test_loader = create_dataloaders(
        train_csv='data/train.csv',
        val_csv='data/val.csv',
        test_csv='data/test.csv',
        batch_size=512,
        seq_length=60,
        num_workers=8
    )
    
    # 创建模型
    print("\n创建模型...")
    model = CNNTeacher(
        input_channels=7,
        seq_length=60,
        num_classes=3,
        dropout_rate=0.3
    )
    
    # 加载模型权重
    print(f"\n加载模型: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型已加载 (来自Epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"   训练时验证F1: {checkpoint.get('val_f1', 'unknown'):.4f}")
    
    # 测试
    print("\n" + "="*60)
    print("开始测试...")
    print("="*60)
    
    all_predictions = []
    all_labels = []
    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(device)
            labels = labels.to(device)
            
            output = model(features)
            logits = output['logits']  # 从字典中提取logits
            loss = criterion(logits, labels)
            
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            test_loss += loss.item()
            
            # 更新进度条
            if (batch_idx + 1) % 10 == 0:
                current_acc = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))
                pbar.set_postfix({'acc': f'{current_acc:.2f}%'})
    
    # 计算最终指标
    test_loss /= len(test_loader)
    test_acc = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))
    test_f1 = f1_score(all_labels, all_predictions, average='macro')
    
    # 打印结果
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1-Score: {test_f1:.4f}")
    print("="*60)
    
    # 检查是否达标
    print()
    if test_acc >= 60.0 and test_f1 >= 0.60:
        print("✅ 模型达到目标性能！")
    else:
        print("⚠️  模型未达到目标性能")
        print(f"   目标: Accuracy >= 60%, F1 >= 0.60")
        print(f"   当前: Accuracy = {test_acc:.2f}%, F1 = {test_f1:.4f}")
        print()
        print("建议:")
        print("  1. 调整超参数重新训练")
        print("  2. 增加模型容量")
        print("  3. 使用数据增强")
        print("  4. 调整标签生成阈值")
    print("="*60)
    
    return test_acc, test_f1


if __name__ == '__main__':
    import sys
    
    # 支持命令行指定模型路径
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/cnn_teacher_best_m4.pth'
    
    test_model(model_path)

