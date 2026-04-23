#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
export BASE_PATH=$(pwd)
export BUILD_PATH="${BASE_PATH}/build"

pr_file=$(realpath "${1:-pr_filelist.txt}")

mkdir -p "${BUILD_PATH}"
cd "${BUILD_PATH}" && rm -f CMakeCache.txt && cmake -DENABLE_EXPERIMENTAL=TRUE -DPREPROCESS_ONLY=ON .. &>/dev/null

cd "${BASE_PATH}"
{
    result=$(python3 ${BASE_PATH}/scripts/util/parse_compile_changed_files.py ${pr_file} TRUE)
} || {
    echo $result && exit 1
}
result=(${result//:/ })
ops=${result[0]}

if [[ "${ops}" != "" ]]; then
    cd "${BASE_PATH}"
    bash build.sh --pkg --experimental --ops=${ops} --vendor_name=ci_test --no_force
fi

# 检测镜像文件更新，执行experimental构建
mirror_updated=0
mapfile -t lines < ${pr_file}
for file_path in "${lines[@]}"
do
    file_path=$(echo "$file_path" | xargs)
    if [[ -z "$file_path" ]]; then
        continue
    fi
    if [[ "$file_path" == "scripts/ci/mirror_update_time.txt" ]]; then
        mirror_updated=1
        break
    fi
done

if [[ "${mirror_updated}" -eq 1 ]]; then
    cd "${BASE_PATH}"
    echo "[EXECUTE_COMMAND] bash scripts/torch_extension/build_torch_extension.sh --soc=ascend910b"
    bash scripts/torch_extension/build_torch_extension.sh --soc=ascend910b
    status=$?
    if [ $status -ne 0 ]; then
        echo "build_torch_extension.sh --soc=ascend910b failed"
        exit 1
    fi
fi