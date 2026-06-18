#!/bin/bash
# apply_ftrl 算子 UT 执行脚本（standalone）
#   - op_host UT：CPU 侧 Tiling/InferShape（gtest + CANN registry，无需 NPU）
#   - op_kernel UT：CPU 孪生 tikicpulib（ICPU_RUN_KF 直跑主线 kernel，无需 NPU）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CLEAN_BUILD=true
VERBOSE=""
RUN_OP_HOST=true
RUN_OP_KERNEL=true
KERNEL_SOC="Ascend910B3"   # CPU 孪生芯片型号（DAV_2201）
SIM_ARCH="dav_2201"        # libpem_davinci.so 所在 simulator 目录

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean) if [ "$2" = "false" ]; then CLEAN_BUILD=false; shift 2; else CLEAN_BUILD=true; shift; fi ;;
        -v|--verbose) VERBOSE="VERBOSE=1"; shift ;;
        --ophost) RUN_OP_HOST=true; RUN_OP_KERNEL=false; shift ;;
        --opkernel) RUN_OP_HOST=false; RUN_OP_KERNEL=true; shift ;;
        -h|--help) echo "Usage: $0 [-c false] [-v] [--ophost|--opkernel]"; exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "apply_ftrl UT  (op_host=${RUN_OP_HOST}, op_kernel=${RUN_OP_KERNEL})"
echo "工作目录: ${SCRIPT_DIR}   清理构建: ${CLEAN_BUILD}"
echo "========================================"

if [ -z "$ASCEND_HOME_PATH" ]; then echo "错误: 未设置 ASCEND_HOME_PATH"; exit 1; fi

# 运行期库路径：CANN lib64 + tikicpulib(含 per-soc libcpudebug) + simulator(libpem_davinci)
ARCH_TRIPLE="$(uname -m)-linux"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/tools/tikicpulib/lib:${ASCEND_HOME_PATH}/tools/tikicpulib/lib/${KERNEL_SOC}:${ASCEND_HOME_PATH}/${ARCH_TRIPLE}/simulator/${SIM_ARCH}/lib:${LD_LIBRARY_PATH}"

if [ "$CLEAN_BUILD" = true ]; then rm -rf "${BUILD_DIR}"; fi
mkdir -p "${BUILD_DIR}"; cd "${BUILD_DIR}"

echo ">>> CMake 配置..."
cmake .. ${VERBOSE}
echo ">>> 编译..."
make -j"$(nproc)" ${VERBOSE}

FAILED=()

if [ "$RUN_OP_HOST" = true ]; then
    echo ""; echo ">>> 运行 op_host UT (Tiling + InferShape) <<<"
    if ( cd "${BUILD_DIR}/op_host" && ./apply_ftrl_op_host_ut ); then
        echo "[PASS] op_host UT"
    else
        echo "[FAIL] op_host UT"; FAILED+=("op_host")
    fi
fi

if [ "$RUN_OP_KERNEL" = true ]; then
    echo ""; echo ">>> 运行 op_kernel UT (CPU twin) <<<"
    if ( cd "${BUILD_DIR}/op_kernel" && ./apply_ftrl_op_kernel_ut ); then
        echo "[PASS] op_kernel UT"
    else
        echo "[FAIL] op_kernel UT"; FAILED+=("op_kernel")
    fi
fi

echo ""
echo "========================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "测试结果: PASS"; echo "========================================"; exit 0
else
    echo "测试结果: FAIL (${FAILED[*]})"; echo "========================================"; exit 1
fi
