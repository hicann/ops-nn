#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. CANN Open Software License 2.0.
#
# ApplyAdamD examples 编译 / 运行脚本。
#
# 本脚本处理三类示例：
#
#   [1] test_aclnn_apply_adam_d.cpp —— aclnn 两段式调用示例。
#       需先构建并部署 ApplyAdamD 自定义算子包，设置 ASCEND_CUSTOM_OPP_PATH，
#       并确保自定义 op_api 库路径在 LD_LIBRARY_PATH 中。
#
#   [2] test_geir_apply_adam_d.cpp  —— 图模式（GE/IR 构图）标准调用形态示例。
#       编译形态与 build.sh --run_example <op> graph 一致（g++ + GE 头/库）。
#       默认仅做「编译检查」（COMPILE-ONLY）：端到端 GE 建图运行需先把 ApplyAdamD 自定义算子包
#       部署到 OPP（设置 ASCEND_CUSTOM_OPP_PATH）。如已部署，传 --run-geir 让脚本尝试运行。
#
#   [3] test_kernel_direct_apply_adam_d.asc —— kernel 直调（<<<>>>）示例。
#       ASC/bisheng 编译（--npu-arch=dav-2201 = ascend910b），在真实 NPU(910B3) 上调用真实内核
#       NsApplyAdamD::ApplyAdamD<T,NES>，与 host(FP32) golden 比对，覆盖 use_nesterov=false/true。
#       这是本环境下「示例编译运行通过」的真机验证路径。
#
# 用法:
#   bash run.sh              # 编译并运行 aclnn + 编译 test_geir（仅编译）+ 编译并真机运行 kernel 直调示例
#   bash run.sh --skip-aclnn  # 跳过 aclnn 示例，仅保留图模式编译检查 + kernel 直调
#   bash run.sh --run-geir   # 额外尝试端到端运行 test_geir（需已部署 ApplyAdamD 自定义算子包）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RUN_GEIR=0
RUN_ACLNN=1
for arg in "$@"; do
    case "${arg}" in
        --run-geir) RUN_GEIR=1 ;;
        --skip-aclnn) RUN_ACLNN=0 ;;
        *) echo "Unknown argument: ${arg}"; exit 1 ;;
    esac
done

# ---- 环境 ----
if [ -z "${ASCEND_HOME_PATH}" ]; then
    for f in /usr/local/Ascend/cann-9.0.0/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
        [ -f "$f" ] && source "$f" && break
    done
fi
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
ARCH_INFO="$(uname -m)"

GRAPH_INCLUDE_PATH="${ASCEND_HOME_PATH}/include/graph"
GE_INCLUDE_PATH="${ASCEND_HOME_PATH}/include/ge"
INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
EXTERNAL_INCLUDE_PATH="${ASCEND_HOME_PATH}/include/external"
LINUX_INCLUDE_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include"
INC_INCLUDE_PATH="${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc"
GRAPH_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"

mkdir -p "${BUILD_DIR}"

# ---- [1/3] aclnn 两段式示例 ----
if [ "${RUN_ACLNN}" = "1" ]; then
    echo "==== [1/3] 编译并运行 aclnn 示例 test_aclnn_apply_adam_d.cpp ===="
    CUSTOM_OPP_ROOT="${ASCEND_CUSTOM_OPP_PATH:-${ASCEND_HOME_PATH}/opp/vendors/custom_nn}"
    CUSTOM_OPAPI_INC="${CUSTOM_OPP_ROOT}/op_api/include"
    CUSTOM_OPAPI_LIB="${CUSTOM_OPP_ROOT}/op_api/lib"
    g++ "${SCRIPT_DIR}/test_aclnn_apply_adam_d.cpp" \
        -std=c++17 \
        -I "${ASCEND_HOME_PATH}/aarch64-linux/include" \
        -I "${CUSTOM_OPAPI_INC}" \
        -L "${ASCEND_HOME_PATH}/lib64" \
        -L "${CUSTOM_OPAPI_LIB}" \
        -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${CUSTOM_OPAPI_LIB}" \
        -lcust_opapi -lnnopbase -lascendcl \
        -o "${BUILD_DIR}/test_aclnn_apply_adam_d"
    "${BUILD_DIR}/test_aclnn_apply_adam_d"
else
    echo "==== [1/3] 跳过 aclnn 示例（--skip-aclnn） ===="
fi

# ---- [2/3] 图模式示例 test_geir（编译检查；与 --run_example graph 编译形态一致） ----
echo "==== [2/3] 编译图模式示例 test_geir_apply_adam_d.cpp (标准调用形态) ===="
g++ "${SCRIPT_DIR}/test_geir_apply_adam_d.cpp" \
    -std=c++17 \
    -I "${GRAPH_INCLUDE_PATH}" -I "${GE_INCLUDE_PATH}" -I "${INCLUDE_PATH}" \
    -I "${EXTERNAL_INCLUDE_PATH}" -I "${LINUX_INCLUDE_PATH}" -I "${INC_INCLUDE_PATH}" \
    -L "${GRAPH_LIBRARY_PATH}" \
    -lgraph -lge_runner -lgraph_base -lge_compiler \
    -o "${BUILD_DIR}/test_geir_apply_adam_d"
echo "[OK] test_geir 编译通过 -> ${BUILD_DIR}/test_geir_apply_adam_d"

if [ "${RUN_GEIR}" = "1" ]; then
    echo "---- 尝试端到端运行 test_geir (需已部署 ApplyAdamD 自定义算子包) ----"
    cd "${BUILD_DIR}"
    set +e
    ./test_geir_apply_adam_d 0 | tail -n 20
    echo "test_geir run exit=$?"
    set -e
else
    echo "[SKIP] test_geir 端到端运行（COMPILE-ONLY）。"
    echo "       端到端运行需部署 ApplyAdamD 自定义算子包并 source 其 set_env，再执行 bash run.sh --run-geir。"
fi

# ---- [3/3] kernel 直调示例（真实 NPU 运行） ----
echo "==== [3/3] 编译并真机运行 kernel 直调示例 test_kernel_direct_apply_adam_d.asc ===="
KDIR="${BUILD_DIR}/kernel_direct"
rm -rf "${KDIR}"; mkdir -p "${KDIR}"; cd "${KDIR}"
cmake "${SCRIPT_DIR}" > cmake_cfg.log 2>&1
make -j"$(nproc)"
echo "---- 真实 NPU 执行 ----"
"${KDIR}/test_kernel_direct_apply_adam_d"
echo "==== 完成 ===="
