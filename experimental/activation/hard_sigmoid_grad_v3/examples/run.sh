#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CANN_ROOT=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0-beta.1}

source "${CANN_ROOT}/set_env.sh"
export LD_LIBRARY_PATH="${CANN_ROOT}/opp/vendors/customize_nn/op_api/lib:${LD_LIBRARY_PATH:-}"

BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

g++ -std=c++17 -O2 \
    "${SCRIPT_DIR}/test_aclnn_hard_sigmoid_grad_v3.cpp" \
    -I"${CANN_ROOT}/aarch64-linux/include" \
    -I"${CANN_ROOT}/opp/vendors/customize_nn/op_api/include" \
    -L"${CANN_ROOT}/lib64" \
    -L"${CANN_ROOT}/opp/vendors/customize_nn/op_api/lib" \
    -Wl,-rpath,"${CANN_ROOT}/lib64:${CANN_ROOT}/opp/vendors/customize_nn/op_api/lib" \
    -lcust_opapi -lnnopbase -lascendcl \
    -o "${BUILD_DIR}/test_aclnn_hard_sigmoid_grad_v3"

"${BUILD_DIR}/test_aclnn_hard_sigmoid_grad_v3"
