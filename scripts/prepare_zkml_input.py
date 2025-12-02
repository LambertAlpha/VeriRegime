#!/usr/bin/env python3
"""
准备zkML证明的输入数据
从测试集中提取真实样本作为EZKL输入
"""

import json
import numpy as np
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def prepare_zkml_input(
    test_csv='data/test.csv',
    feature_cols=['ema_5', 'ema_10', 'ema_20', 'rsi', 'macd', 'volume_ma_5', 'volume_ma_10'],
    seq_length=240,
    output_file='results/zkml/input.json',
    sample_idx=0
):
    """
    从测试集提取样本作为zkML输入
    
    Args:
        test_csv: 测试集路径
        feature_cols: 特征列名
        seq_length: 序列长度
        output_file: 输出JSON路径
        sample_idx: 样本索引
    """
    print("=" * 60)
    print("准备zkML输入数据")
    print("=" * 60)
    
    # 1. 加载测试数据
    print(f"\n1. 加载测试数据: {test_csv}")
    df = pd.read_csv(test_csv)
    print(f"   测试集大小: {len(df):,} 样本")
    
    # 2. 提取特征
    print(f"\n2. 提取特征")
    features = df[feature_cols].values
    labels = df['label'].values
    
    # 3. 标准化（使用训练集统计）
    print(f"\n3. 标准化特征（Z-score）")
    # 简单标准化（实际使用时应加载训练集的mean/std）
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features_normalized = (features - mean) / std
    
    # 4. 创建序列
    print(f"\n4. 创建序列 (seq_length={seq_length})")
    if sample_idx + seq_length > len(features_normalized):
        sample_idx = 0
        print(f"   警告：sample_idx过大，使用第0个样本")
    
    input_sequence = features_normalized[sample_idx:sample_idx+seq_length]
    true_label = labels[sample_idx + seq_length - 1] if sample_idx + seq_length <= len(labels) else labels[-1]
    
    print(f"   样本索引: {sample_idx}")
    print(f"   序列形状: {input_sequence.shape}")
    print(f"   真实标签: {true_label} ({'HIGH' if true_label == 1 else 'LOW'} Volatility)")
    
    # 5. 转换为EZKL格式
    print(f"\n5. 转换为EZKL输入格式")
    # EZKL期望: [batch_size, seq_length, features]
    input_data = input_sequence.reshape(1, seq_length, len(feature_cols)).tolist()
    
    ezkl_input = {
        "input_data": [input_data]
    }
    
    # 6. 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(ezkl_input, f, indent=2)
    
    print(f"\n✅ 输入数据已保存: {output_file}")
    
    # 7. 保存元数据
    metadata = {
        'sample_idx': sample_idx,
        'seq_length': seq_length,
        'features': feature_cols,
        'true_label': int(true_label),
        'label_name': 'HIGH' if true_label == 1 else 'LOW',
        'input_shape': [1, seq_length, len(feature_cols)],
        'normalization': {
            'method': 'z-score',
            'mean': mean.tolist(),
            'std': std.tolist()
        }
    }
    
    metadata_file = output_file.replace('.json', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ 元数据已保存: {metadata_file}")
    
    print(f"\n{'=' * 60}")
    print("🎉 zkML输入准备完成！")
    print(f"{'=' * 60}")
    print(f"\n下一步:")
    print(f"  1. 确保EZKL已安装: ezkl --version")
    print(f"  2. 运行证明生成: ./scripts/zkml_generate_proof.sh")
    
    return ezkl_input, metadata

if __name__ == '__main__':
    # 准备输入数据
    ezkl_input, metadata = prepare_zkml_input(
        test_csv='data/test.csv',
        sample_idx=1000  # 使用第1000个样本（可调整）
    )
    
    print(f"\n输入数据预览:")
    print(f"  形状: {metadata['input_shape']}")
    print(f"  真实标签: {metadata['label_name']}")
    print(f"  特征: {', '.join(metadata['features'])}")

