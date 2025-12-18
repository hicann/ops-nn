#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

curpath=$(dirname $(readlink -f "$0"))
SCENE_FILE="${curpath}""/../scene.info"
OPP_COMMON="${curpath}""/opp_common.sh"
common_func_path="${curpath}/common_func.inc"
. "${OPP_COMMON}"
. "${common_func_path}"

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --latest-dir=*)
        latest_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    -*)
        shift
        ;;
    *)
        break
        ;;
    esac
done
if [ -d "$install_path/$version_dir/opp" ]; then
    ln -srfn "$install_path/$version_dir/opp" "$install_path/$latest_dir/ops"
fi

get_version_dir "ops_nn_version_dir" "$install_path/$latest_dir/ops_nn/version.info"

# if [ -z "$latest_version_dir" ]; then
#     # create op_api soft link
#     logandprint "[INFO]: Start create opapi latest softlinks."
#     createOpapiLatestSoftlink ${install_path}/${version_dir}
#     return_code=$?
#     if [ ${return_code} -eq 0 ]; then
#         logandprint "[INFO]: Create opapi latest softlinks successfully!"
#     elif [ ${return_code} -eq 3 ]; then
#         logandprint "[WARNING]: opapi source file does not exist!"
#     else
#         logandprint "[ERROR]: Create opapi latest softlinks failed!"
#     fi
# fi

# if [ -n "$latest_version_dir" ] && [ -d "$install_path/$latest_version_dir/opp" ]; then
#     if [ "${latest_version_dir}" != "${version_dir}" ]; then
#         logandprint "[INFO]: opp_kernel version is $latest_version_dir, the pkg version is $version_dir, and the opp_latest soft link is created."
#         ln -srfn "$install_path/$latest_version_dir/opp" "$install_path/$latest_dir/opp_latest"
#     else
#         logandprint "[INFO]: opp_kernel version is $latest_version_dir, the pkg version is $version_dir, and the opp_latest soft link is deleted."
#         rm -rf "$install_path/$latest_dir/opp_latest"
#     fi
# fi
