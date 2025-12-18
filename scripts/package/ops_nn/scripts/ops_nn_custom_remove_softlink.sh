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

architecture=$(uname -m)
architectureDir="${architecture}-linux"

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

remove_empty_file() {
    target_dir="$1"
    if [ -d "$target_dir" ] && [ -z "$(ls -A $target_dir)" ]; then
        dir="$target_dir"
        while [ -n "$dir" ] && [ "$dir" != "/" ]; do
            if [ -z "$(ls -A "$dir")" ]; then
                rmdir "$dir"
            else
                break
            fi
            dir=$(dirname "$dir")
            if [[ $dir == $install_path/latest ]]; then
                return 0
            fi
        done
    fi
}

remove_ops_nn_opp_link() {
    opp_path="$1"
    chmod u+w "${opp_path}" 2> /dev/null
    if [ -f "${opp_path}" ] || [ -L "${opp_path}" ] || [ -d "${opp_path}" ]; then
        find ${opp_path} -type l -lname "*/ops_nn/*" -delete
    fi
    logandprint "[INFO]: Delete the latest ops_nn link ("${opp_path}")."
    remove_empty_file ${opp_path}
}

get_version_dir "ops_nn_version_dir" "$install_path/$version_dir/ops_nn/version.info"

if [ -n "$ops_nn_version_dir" ]; then
    logandprint "[INFO]: Start remove latest ops_nn ."
    opp_original_perms="555"
    arch_original_perms="555"

    opp_dst_dir=$install_path/latest/opp
    ops_nn_dst_dir=$install_path/latest/ops_nn
    arch_dst_dir=$install_path/latest/${architectureDir}

    if [ -d "${opp_dst_dir}" ]; then
        opp_original_perms=$(stat -c "%a" "$opp_dst_dir")
        chmod -R 770 ${opp_dst_dir} 2> /dev/null
    fi

    if [ -d "${arch_dst_dir}" ]; then
        arch_original_perms=$(stat -c "%a" "$arch_dst_dir")
        chmod -R 770 ${arch_dst_dir} 2> /dev/null
    fi

    opp_path=$install_path/latest
    remove_ops_nn_opp_link ${opp_path} "ops_nn"

    PATTERN="ascend*"
    opp_aic_info_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/config
    files=( $opp_aic_info_path/$PATTERN )
    for file_path in "${files[@]}"; do
        filename=$(basename "$file_path")
        remove_ops_nn_opp_link ${opp_aic_info_path}/${filename} "aic-${filename}-ops-info-nn.json"
    done

    opp_dynamic_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/impl/dynamic
    remove_ops_nn_opp_link ${opp_dynamic_path}

    opp_ascendc_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/impl/ascendc
    remove_ops_nn_opp_link ${opp_ascendc_path}

    opp_impl_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/impl
    remove_ops_nn_opp_link ${opp_impl_path}

    opp_kernel_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/kernel
    files=( $opp_kernel_path/$PATTERN )
    for file_path in "${files[@]}"; do
        filename=$(basename "$file_path")
        remove_ops_nn_opp_link ${opp_kernel_path}/${filename}
    done

    opp_config_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/kernel/config
    files=( $opp_config_path/$PATTERN )
    for file_path in "${files[@]}"; do
        filename=$(basename "$file_path")
        remove_ops_nn_opp_link ${opp_config_path}/${filename}
    done

    opp_host_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/x86_64
    remove_ops_nn_opp_link ${opp_host_path} "libophost_nn.so"

    opp_graph_inc_path=${opp_dst_dir}/built-in/op_graph/inc
    remove_ops_nn_opp_link ${opp_graph_inc_path} "nn_ops.h"

    latest_api_inc_path=${arch_dst_dir}/include/aclnnop
    opp_api_inc_path=${opp_dst_dir}/include/aclnnop
    opp_api_level2_inc_path=${opp_dst_dir}/include/aclnnop/level2
    remove_ops_nn_opp_link ${latest_api_inc_path}
    remove_ops_nn_opp_link ${opp_api_inc_path}
    remove_ops_nn_opp_link ${opp_api_level2_inc_path}

    latest_api_lib_path=${arch_dst_dir}/lib64
    opp_api_lib_path=${opp_dst_dir}/lib64
    remove_ops_nn_opp_link ${latest_api_lib_path}
    remove_ops_nn_opp_link ${opp_api_lib_path}

    if [ -d "${opp_dst_dir}" ]; then
        chmod "${opp_original_perms}" "$opp_dst_dir"
    fi

    if [ -d "${arch_dst_dir}" ]; then
        chmod "${arch_original_perms}" "$arch_dst_dir"
    fi

    logandprint "[INFO]: Remove latest ops_nn successfully."
fi
