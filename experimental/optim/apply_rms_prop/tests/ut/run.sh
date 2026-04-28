#!/bin/bash
# apply_rms_prop 算子 UT 测试执行脚本

set -e

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CLEAN_BUILD=true
VERBOSE=""
RUN_OP_HOST=true
RUN_OP_API=false

# ============================================================================
# 帮助信息
# ============================================================================
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "UT 测试（单元测试）：测试 Tiling、InferShape、op_api 逻辑"
    echo ""
    echo "Options:"
    echo "  -c, --clean      清理构建目录（默认: true）"
    echo "  -v, --verbose    显示详细输出"
    echo "  --ophost         仅运行 op_host 测试"
    echo "  --opapi          仅运行 op_api 测试"
    echo "  -h, --help       显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  # 默认模式（清理 + 编译 + 运行所有测试）"
    echo "  $0"
    echo ""
    echo "  # 仅运行 op_host 测试"
    echo "  $0 --ophost"
    echo ""
    echo "  # 仅运行 op_api 测试"
    echo "  $0 --opapi"
    echo ""
    echo "  # 不清理构建目录（增量编译）"
    echo "  $0 -c false"
    echo ""
    echo "  # 详细输出"
    echo "  $0 -v"
}

# ============================================================================
# 解析参数
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            if [ "$2" = "false" ]; then
                CLEAN_BUILD=false
                shift 2
            else
                CLEAN_BUILD=true
                shift
            fi
            ;;
        -v|--verbose)
            VERBOSE="VERBOSE=1"
            shift
            ;;
        --ophost)
            RUN_OP_HOST=true
            RUN_OP_API=false
            shift
            ;;
        --opapi)
            RUN_OP_HOST=false
            RUN_OP_API=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# 显示配置信息
# ============================================================================
echo "========================================"
echo "apply_rms_prop 算子 UT 测试"
echo "========================================"
echo "清理构建: ${CLEAN_BUILD}"
echo "工作目录: ${SCRIPT_DIR}"
echo "测试范围: op_host=${RUN_OP_HOST}, op_api=${RUN_OP_API}"
echo "========================================"
echo ""

# ============================================================================
# 检查依赖
# ============================================================================
echo "检查依赖..."

# 检查 cmake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake"
    exit 1
fi

# 检查 g++
if ! command -v g++ &> /dev/null; then
    echo "错误: 未找到 g++"
    exit 1
fi

echo "依赖检查完成"
echo ""

# ============================================================================
# 设置环境变量
# ============================================================================
echo "设置环境变量..."

# 检查 CANN 环境
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "警告: 未设置 ASCEND_HOME_PATH 环境变量"
    echo "使用默认路径: /home/developer/Ascend/cann"
    export ASCEND_HOME_PATH=/home/developer/Ascend/cann
fi

# 设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH: ${ASCEND_HOME_PATH}/lib64"

echo "环境变量设置完成"
echo ""

# ============================================================================
# 清理构建目录
# ============================================================================
if [ "$CLEAN_BUILD" = true ]; then
    echo "清理构建目录..."
    rm -rf "${BUILD_DIR}"
fi

# ============================================================================
# 创建构建目录
# ============================================================================
echo "创建构建目录..."
mkdir -p "${BUILD_DIR}"

# ============================================================================
# CMake 配置
# ============================================================================
echo ""
echo "CMake 配置..."
cd "${BUILD_DIR}"

cmake .. ${VERBOSE}

if [ $? -ne 0 ]; then
    echo "错误: CMake 配置失败"
    exit 1
fi

# ============================================================================
# 编译
# ============================================================================
echo ""
echo "编译 UT 测试..."
make -j$(nproc) ${VERBOSE}

if [ $? -ne 0 ]; then
    echo "错误: 编译失败"
    exit 1
fi

echo "编译成功"
echo ""

# ============================================================================
# 执行 UT 测试
# ============================================================================
echo "========================================"
echo "执行 UT 测试"
echo "========================================"

FAILED_TESTS=()
PASSED_TESTS=()

# 执行 op_host 测试
if [ "$RUN_OP_HOST" = true ]; then
    echo ""
    echo ">>> 运行 op_host 测试 <<<"
    echo ""
    cd "${BUILD_DIR}/op_host"
    
    if [ ! -f "./apply_rms_prop_op_host_ut" ]; then
        echo "错误: op_host UT 可执行文件不存在"
        FAILED_TESTS+=("op_host")
    else
        if ./apply_rms_prop_op_host_ut; then
            PASSED_TESTS+=("op_host")
            echo "[PASS] op_host 测试通过"
        else
            FAILED_TESTS+=("op_host")
            echo "[FAIL] op_host 测试失败"
        fi
    fi
fi

# 执行 op_api 测试
if [ "$RUN_OP_API" = true ]; then
    echo ""
    echo ">>> 运行 op_api 测试 <<<"
    echo ""
    echo "提示: apply_rms_prop 当前未实现 op_api UT，且 UT CMake 默认未启用 op_api 编译。"
    echo "如需启用，请先在 CMakeLists.txt 中开启 op_api 子目录并补齐对应用例。"
    echo "[SKIP] op_api 测试跳过"
fi

# ============================================================================
# 输出结果汇总
# ============================================================================
echo ""
echo "========================================"
echo "测试结果汇总"
echo "========================================"
echo ""

if [ ${#PASSED_TESTS[@]} -gt 0 ]; then
    echo "✓ 通过的测试:"
    for test in "${PASSED_TESTS[@]}"; do
        echo "  - ${test}"
    done
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "✗ 失败的测试:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - ${test}"
    done
    echo ""
    echo "========================================"
    echo "测试结果: FAIL ✗"
    echo "========================================"
    exit 1
else
    echo ""
    echo "========================================"
    echo "测试结果: PASS ✓"
    echo "========================================"
    exit 0
fi
