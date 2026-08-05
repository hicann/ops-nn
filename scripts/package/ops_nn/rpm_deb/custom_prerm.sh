#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

sourcedir="${INSTALL_PATH}"
WHL_INSTALL_DIR_PATH="${sourcedir}/python/site-packages"
export PYTHONPATH="${WHL_INSTALL_DIR_PATH}"
export PIP_BREAK_SYSTEM_PACKAGES=1

run_pip() { python3 -m pip "$@" || pip3 "$@"; }
run_pip uninstall -y cann-ops-nn >/dev/null 2>&1 || true
run_pip uninstall -y es-nn >/dev/null 2>&1 || true

rm -rf "${WHL_INSTALL_DIR_PATH}/cann_ops_nn" 2>/dev/null
rm -rf "${WHL_INSTALL_DIR_PATH}"/cann_ops_nn-*.dist-info 2>/dev/null
rm -rf "${WHL_INSTALL_DIR_PATH}/es_nn" 2>/dev/null
rm -rf "${WHL_INSTALL_DIR_PATH}"/es_nn-*.dist-info 2>/dev/null

rm -f "${sourcedir}"/ops_nn/es_packages/whl/*.whl 2>/dev/null
rmdir "${sourcedir}"/ops_nn/es_packages/whl 2>/dev/null
rmdir "${sourcedir}"/ops_nn/es_packages 2>/dev/null

rm -f "${WHL_INSTALL_DIR_PATH}"/cann_ops_nn-*.whl 2>/dev/null
rmdir "${WHL_INSTALL_DIR_PATH}" 2>/dev/null
parent=$(dirname "${WHL_INSTALL_DIR_PATH}")
[ -d "${parent}" ] && rmdir "${parent}" 2>/dev/null
