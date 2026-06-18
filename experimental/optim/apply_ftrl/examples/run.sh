#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# ApplyFtrl examples 一键编译 + 真实 NPU 运行（Kernel 直调 <<<>>> 样例）
#
# 编译 test_kernel_direct_apply_ftrl.asc（复用本算子 op_kernel/apply_ftrl_kernel.h），
# 在 ascend910b 真实 NPU 上运行并与 fp64 CPU golden 比对三输出 var/accum/linear。
#
# 用法:
#   bash run.sh                # 完整流程（编译 + 运行）
#   bash run.sh --skip-build   # 复用已编译产物
#
# 设备选择：默认逻辑 device 0；若 device 0 被占满，请用 ASCEND_RT_VISIBLE_DEVICES 指定，例如:
#   ASCEND_RT_VISIBLE_DEVICES=1 bash run.sh
#
# 注：GE 图模式样例 test_geir_apply_ftrl.cpp 经仓库统一流程构建运行（需先把算子安装进 OPP）:
#   bash build.sh --pkg --experimental --soc=ascend910b --ops=apply_ftrl   # 编译 + 安装算子
#   bash build.sh --run_example apply_ftrl graph                           # 构图下发样例
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_BIN="test_kernel_direct_apply_ftrl"

SKIP_BUILD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1 ;;
        *) echo "[WARN] unknown arg: $1" ;;
    esac
    shift
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/3] 设置 CANN 环境 ==="
# 需先 source <CANN 安装路径>/set_env.sh，使 ASCEND_HOME_PATH 指向本机 CANN 安装（不硬编码容器路径）。
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置，请先 source <CANN 安装路径>/set_env.sh"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh 执行失败"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_BIN}" ] || die "--skip-build 指定但 build/${OP_BIN} 不存在"
    echo "=== [2/3] 跳过编译 ==="
else
    echo "=== [2/3] 编译（主线 kernel via ASC, dav-2201） ==="
    rm -rf build
    mkdir -p build
    cd build
    cmake .. || die "cmake 配置失败"
    make -j8 || die "make 编译失败"
    cd ..
fi

echo "=== [3/3] 真实 NPU 运行 + 三输出精度比对 (device=${ASCEND_RT_VISIBLE_DEVICES:-0}) ==="
"./build/${OP_BIN}"
RC=$?

echo ""
if [ "${RC}" -eq 0 ]; then
    echo "ApplyFtrl Kernel 直调样例: RESULT PASS"
else
    echo "ApplyFtrl Kernel 直调样例: RESULT FAIL (rc=${RC})"
fi
exit "${RC}"
