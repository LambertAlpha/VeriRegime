#!/bin/bash
# VeriRegime - EZKL证明生成脚本

set -e

echo "=========================================="
echo "VeriRegime - zkML证明生成"
echo "=========================================="

# 进入项目根目录
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# 路径配置
ONNX_MODEL="${PROJECT_ROOT}/results/onnx/student_model.onnx"
INPUT_FILE="${PROJECT_ROOT}/results/zkml/input.json"
SETTINGS_FILE="${PROJECT_ROOT}/results/zkml/settings/settings.json"
COMPILED_MODEL="${PROJECT_ROOT}/results/zkml/compiled/network.ezkl"
PK_FILE="${PROJECT_ROOT}/results/zkml/compiled/pk.key"
VK_FILE="${PROJECT_ROOT}/results/zkml/compiled/vk.key"
PROOF_FILE="${PROJECT_ROOT}/results/zkml/proof/proof.json"
WITNESS_FILE="${PROJECT_ROOT}/results/zkml/proof/witness.json"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查ONNX模型是否存在
if [ ! -f "$ONNX_MODEL" ]; then
    echo -e "${YELLOW}⚠️ ONNX模型不存在，请先运行 notebooks/export_onnx.ipynb${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 找到ONNX模型: ${ONNX_MODEL}${NC}"

# Step 1: 生成设置文件
echo ""
echo "=========================================="
echo "Step 1: 生成EZKL设置"
echo "=========================================="

ezkl gen-settings \
    -M ${ONNX_MODEL} \
    -O ${SETTINGS_FILE} \
    --input-visibility "public" \
    --param-visibility "fixed"

echo -e "${GREEN}✅ 设置文件已生成${NC}"

# Step 2: 校准设置
echo ""
echo "=========================================="
echo "Step 2: 校准设置（优化电路参数）"
echo "=========================================="

ezkl calibrate-settings \
    -M ${ONNX_MODEL} \
    -D ${INPUT_FILE} \
    -O ${SETTINGS_FILE}

echo -e "${GREEN}✅ 设置校准完成${NC}"

# Step 3: 编译电路
echo ""
echo "=========================================="
echo "Step 3: 编译ZK电路"
echo "=========================================="

ezkl compile-circuit \
    -M ${ONNX_MODEL} \
    -S ${SETTINGS_FILE} \
    --compiled-circuit ${COMPILED_MODEL}

echo -e "${GREEN}✅ 电路编译完成${NC}"

# Step 4: 生成证明密钥和验证密钥
echo ""
echo "=========================================="
echo "Step 4: 生成密钥（这可能需要几分钟）"
echo "=========================================="

ezkl setup \
    --compiled-circuit ${COMPILED_MODEL} \
    --pk-path ${PK_FILE} \
    --vk-path ${VK_FILE}

echo -e "${GREEN}✅ 密钥生成完成${NC}"

# Step 5: 生成见证
echo ""
echo "=========================================="
echo "Step 5: 生成见证（Witness）"
echo "=========================================="

ezkl gen-witness \
    -M ${ONNX_MODEL} \
    -D ${INPUT_FILE} \
    -O ${WITNESS_FILE}

echo -e "${GREEN}✅ 见证生成完成${NC}"

# Step 6: 生成证明
echo ""
echo "=========================================="
echo "Step 6: 生成ZK证明"
echo "=========================================="

START_TIME=$(date +%s)

ezkl prove \
    --witness ${WITNESS_FILE} \
    --compiled-circuit ${COMPILED_MODEL} \
    --pk-path ${PK_FILE} \
    --proof-path ${PROOF_FILE}

END_TIME=$(date +%s)
PROOF_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}✅ 证明生成完成（用时: ${PROOF_TIME}秒）${NC}"

# Step 7: 验证证明
echo ""
echo "=========================================="
echo "Step 7: 验证ZK证明"
echo "=========================================="

START_TIME=$(date +%s)

ezkl verify \
    --proof-path ${PROOF_FILE} \
    --vk-path ${VK_FILE} \
    --settings-path ${SETTINGS_FILE}

END_TIME=$(date +%s)
VERIFY_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}✅ 证明验证成功（用时: ${VERIFY_TIME}秒）${NC}"

# 总结
echo ""
echo "=========================================="
echo "🎉 zkML证明生成完成！"
echo "=========================================="
echo ""
echo "性能统计:"
echo "  证明生成时间: ${PROOF_TIME}秒"
echo "  验证时间: ${VERIFY_TIME}秒"
echo ""
echo "输出文件:"
echo "  证明: ${PROOF_FILE}"
echo "  见证: ${WITNESS_FILE}"
echo "  编译电路: ${COMPILED_MODEL}"
echo "  证明密钥: ${PK_FILE}"
echo "  验证密钥: ${VK_FILE}"
echo ""

