#!/bin/bash
# ============================================================================
# apply_adam_d 算子 op_host UT（Tiling + InferShape）执行脚本
#
# 独立自包含工程（不依赖 build.sh -u）：
#   配置 -> 编译 -> 运行 gtest（--gtest_output=json）-> 转换为 tests/ut/test-report.json
#
# test-report.json 来自真实 UT 运行（gtest JSON），非脚本伪造。
# 退出码：UT 全通过=0，否则非 0。
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RESULTS_DIR="${SCRIPT_DIR}/results"
GTEST_JSON="${RESULTS_DIR}/gtest_op_host.json"
REPORT_JSON="${SCRIPT_DIR}/test-report.json"
CLEAN_BUILD=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--clean)   CLEAN_BUILD=true;  shift ;;
        --no-clean)   CLEAN_BUILD=false; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "apply_adam_d op_host UT (Tiling + InferShape)"
echo "工作目录: ${SCRIPT_DIR}"
echo "========================================"

# ---------------------------------------------------------------------------
# CANN 环境
# ---------------------------------------------------------------------------
if [ -z "${ASCEND_HOME_PATH}" ]; then
    for f in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh; do
        if [ -f "$f" ]; then echo "source $f"; source "$f"; break; fi
    done
fi
if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "错误: 未设置 ASCEND_HOME_PATH，且未找到 set_env.sh"
    exit 1
fi
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}"

# ---------------------------------------------------------------------------
# 解析 nlohmann/json.hpp 路径（CANN 不提供，取仓内 third_party）
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
NLOHMANN_JSON_INCLUDE=""
for c in \
    "${REPO_ROOT}/third_party/json/single_include" \
    "${REPO_ROOT}/third_party/json/include" \
    "/usr/include"; do
    if [ -f "${c}/nlohmann/json.hpp" ]; then NLOHMANN_JSON_INCLUDE="${c}"; break; fi
done
if [ -z "${NLOHMANN_JSON_INCLUDE}" ]; then
    echo "错误: 未找到 nlohmann/json.hpp（请确保 third_party/json 已就绪，运行一次 build.sh 可自动拉取）"
    exit 1
fi
echo "NLOHMANN_JSON_INCLUDE=${NLOHMANN_JSON_INCLUDE}"

# ---------------------------------------------------------------------------
# 配置 + 编译
# ---------------------------------------------------------------------------
mkdir -p "${RESULTS_DIR}"
if [ "${CLEAN_BUILD}" = true ]; then rm -rf "${BUILD_DIR}"; fi
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DSOC_VERSION=Ascend910B \
    -DNLOHMANN_JSON_INCLUDE="${NLOHMANN_JSON_INCLUDE}"

make -j"$(nproc)"

# ---------------------------------------------------------------------------
# 运行 UT（真实执行 + gtest JSON 输出）
# ---------------------------------------------------------------------------
UT_BIN="${BUILD_DIR}/op_host/apply_adam_d_op_host_ut"
if [ ! -x "${UT_BIN}" ]; then
    echo "错误: 未生成 UT 可执行文件 ${UT_BIN}"
    exit 1
fi

echo ""
echo ">>> 运行 op_host UT <<<"
rm -f "${GTEST_JSON}"
set +e
( cd "${BUILD_DIR}/op_host" && "${UT_BIN}" --gtest_output="json:${GTEST_JSON}" )
UT_RC=$?
set -e

# ---------------------------------------------------------------------------
# gtest JSON -> test-report.json（validator 强校验格式）
# ---------------------------------------------------------------------------
python3 "${SCRIPT_DIR}/gen_report.py" "${GTEST_JSON}" "${REPORT_JSON}"
GEN_RC=$?

echo ""
echo "========================================"
if [ "${UT_RC}" -eq 0 ] && [ "${GEN_RC}" -eq 0 ]; then
    echo "测试结果: PASS"
    echo "report: ${REPORT_JSON}"
    exit 0
else
    echo "测试结果: FAIL (ut_rc=${UT_RC}, gen_rc=${GEN_RC})"
    echo "report: ${REPORT_JSON}"
    exit 1
fi
