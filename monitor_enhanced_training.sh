#!/bin/bash
# 增强版CNN训练监控脚本

echo "======================================================================"
echo "Enhanced CNN Teacher 训练监控"
echo "======================================================================"
echo ""

# 检查训练是否在运行
if ps aux | grep -q "[p]ython src/train_cnn_enhanced.py"; then
    echo "✅ 训练进程正在运行"
    PID=$(ps aux | grep "[p]ython src/train_cnn_enhanced.py" | awk '{print $2}')
    echo "   进程ID: $PID"
    echo ""
else
    echo "⚠️  训练进程未运行"
    echo ""
fi

# 显示最新进度
echo "最新训练日志 (关键信息):"
echo "----------------------------------------------------------------------"
if [ -f "cnn_enhanced_training.log" ]; then
    # 提取Epoch信息、训练/验证指标
    tail -100 cnn_enhanced_training.log | grep -E "(Epoch|Train Loss|Val Loss|✅|⚠️|早停|最佳)" | tail -20
else
    echo "⏳ 日志文件尚未生成..."
fi
echo ""

# 检查是否有已保存的模型
if [ -f "models/cnn_enhanced_best.pth" ]; then
    echo "✅ 最佳模型已保存: models/cnn_enhanced_best.pth"
    ls -lh models/cnn_enhanced_best.pth | awk '{print "   大小: "$5"  修改时间: "$6" "$7" "$8}'
else
    echo "⏳ 等待最佳模型保存..."
fi

echo ""
echo "======================================================================"
echo "实时监控命令: tail -f cnn_enhanced_training.log"
echo "停止训练: kill -9 $PID"
echo "======================================================================"

