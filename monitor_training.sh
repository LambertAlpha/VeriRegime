#!/bin/bash
# 训练监控脚本

echo "======================================================================"
echo "VeriRegime CNN Teacher 训练监控"
echo "======================================================================"
echo ""

# 检查训练是否在运行
if ps aux | grep -q "[p]ython src/train_cnn.py"; then
    echo "✅ 训练进程正在运行"
    echo ""
else
    echo "⚠️  训练进程未运行"
    echo ""
    exit 1
fi

# 显示最新进度
echo "最新训练日志 (最后30行):"
echo "----------------------------------------------------------------------"
tail -30 cnn_training.log | grep -E "(Epoch|Train|Val|Best|Early|✅)"
echo ""

# 检查是否有已保存的模型
if [ -f "models/cnn_teacher_best.pth" ]; then
    echo "✅ 最佳模型已保存: models/cnn_teacher_best.pth"
    ls -lh models/cnn_teacher_best.pth
else
    echo "⏳ 等待最佳模型保存..."
fi

echo ""
echo "======================================================================"
echo "实时监控命令: tail -f cnn_training.log"
echo "======================================================================"
