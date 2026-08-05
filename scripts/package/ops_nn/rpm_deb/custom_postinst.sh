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
unset PYTHONPATH
export PIP_BREAK_SYSTEM_PACKAGES=1

run_pip() { python3 -m pip "$@" || pip3 "$@"; }

whl_dir="${sourcedir}/ops_nn/es_packages/whl"
if [ -d "${whl_dir}" ]; then
    for whl in "${whl_dir}"/*.whl; do
        if [ -f "${whl}" ]; then
            echo "[ops-nn] installing ${whl}"
            run_pip install --disable-pip-version-check --upgrade --no-deps --force-reinstall -t "${WHL_INSTALL_DIR_PATH}" "${whl}" \
                && rm -f "${whl}" || true
        fi
    done
fi

cann_ops_nn_whl="${sourcedir}/python/site-packages/cann_ops_nn-*.whl"
for whl in ${cann_ops_nn_whl}; do
    if [ -f "${whl}" ]; then
        echo "[ops-nn] installing ${whl}"
        run_pip install --disable-pip-version-check --upgrade --no-deps --force-reinstall -t "${WHL_INSTALL_DIR_PATH}" "${whl}" \
            && rm -f "${whl}" || true
    fi
done
