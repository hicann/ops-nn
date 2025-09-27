#!/bin/bash
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

curpath=$(dirname $(readlink -f "$0"))
SCENE_FILE="${curpath}""/../scene.info"
OPP_COMMON="${curpath}""/opp_common.sh"
common_func_path="${curpath}/common_func.inc"
. "${OPP_COMMON}"
. "${common_func_path}"
# init arch
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

create_ops_nn_opp_link() {
    opp_path="$1"
    link_to_file="$2"
    ops_path="$3"
    link_from_file="$4"
    if [ ! -d "${opp_path}" ]; then
        mkdir -p "${opp_path}" || { echo "Failed to create directory $opp_path" >&2; return 1;}
    fi

    if [ -f ${opp_path}/${link_to_file} ] || [ -L ${opp_path}/${link_to_file} ] || [ -d ${opp_path}/${link_to_file} ]; then
        rm -rf ${opp_path}/${link_to_file}
    fi
    if [ -f ${ops_path}/${link_from_file} ] || [ -d ${ops_path}/${link_from_file} ]; then
        createrelativelysoftlink "${ops_path}""/""${link_from_file}" "${opp_path}""/""${link_to_file}"
        logwitherrorlevel "$?" "warn" "[WARNING]: Create soft link list for "${opp_path}""/""${link_to_file}"."
    fi
}

create_ops_nn_opp_link_list() {
    opp_path="$1"
    ops_path="$2"
    if [ ! -d "${opp_path}" ]; then
        mkdir -p "${opp_path}"
    fi
    if [ ! -f "${ops_path}" ] && [ ! -d "${ops_path}" ]; then
        return
    fi
    for file in $(find "$ops_path" -maxdepth 1 | grep -v "$ops_path$"); do
        filename=$(basename "$file")
        if [ -f ${opp_path}/${filename} ] || [ -L ${opp_path}/${filename} ] || [ -d ${opp_path}/${filename} ]; then
            if [ "$filename" == "binary_info_config.json" ]; then
                local binary_json_mode=$(stat -c %a ${opp_path}/${filename})
                if [ "$(id -u)" != 0 ] && [ ! -w "${opp_path}/${filename}" ]; then
                    chmod u+w "${opp_path}/${filename}" 2>/dev/null
                fi
                python3 ${curpath}/merge_binary_info_config.py --base-file ${opp_path}/${filename} \
                    --update-file ${file} --output-file ${opp_path}/${filename}

                chmod ${binary_json_mode} ${opp_path}/${filename} 2>/dev/null
                continue
            fi
            rm -rf ${opp_path}/${filename}
        fi

        createrelativelysoftlink "$file" "$opp_path/$filename"
        logwitherrorlevel "$?" "warn" "[WARNING]: Create soft link list for ("${opp_path}""/""${filename}"). "
    done
}

get_version_dir "ops_nn_version_dir" "$install_path/$version_dir/ops_nn/version.info"

if [ -n "$ops_nn_version_dir" ]; then
    logandprint "[INFO]: Start create latest ops_nn."
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

    create_ops_nn_opp_link $install_path/latest "ops_nn" $install_path/$version_dir "ops_nn"

    opp_aic_info_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/config/ascend910b
    ops_aic_info_path=${ops_nn_dst_dir}/built-in/op_impl/ai_core/tbe/config/ascend910b
    create_ops_nn_opp_link ${opp_aic_info_path} "aic-ascend910b-ops-info-nn.json" ${ops_aic_info_path} "aic-ascend910b-ops-info-nn.json"

    opp_impl_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/impl
    ops_impl_path=${ops_nn_dst_dir}/built-in/op_impl/ai_core/tbe
    create_ops_nn_opp_link ${opp_impl_path} "ops_nn" ${ops_impl_path} "impl"

    opp_dynamic_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/impl/dynamic
    ops_dynamic_path=${ops_nn_dst_dir}/built-in/op_impl/ai_core/tbe/impl/dynamic
    create_ops_nn_opp_link_list ${opp_dynamic_path} ${ops_dynamic_path}

    opp_ascendc_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/impl/ascendc
    ops_ascendc_path=${ops_nn_dst_dir}/built-in/op_impl/ai_core/tbe/impl/ascendc
    create_ops_nn_opp_link_list ${opp_ascendc_path} ${ops_ascendc_path}

    opp_kernel_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/kernel/ascend910b
    ops_kernel_path=${ops_nn_dst_dir}/built-in/op_impl/ai_core/tbe/kernel
    create_ops_nn_opp_link ${opp_kernel_path} "ops_nn" ${ops_kernel_path} "ascend910b"
    create_ops_nn_opp_link_list ${opp_kernel_path} "${ops_kernel_path}/ascend910b"

    opp_config_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b
    ops_config_path=${ops_nn_dst_dir}/built-in/op_impl/ai_core/tbe/kernel/config
    create_ops_nn_opp_link ${opp_config_path} "ops_nn" ${ops_config_path} "ascend910b"
    create_ops_nn_opp_link_list ${opp_config_path} "${ops_config_path}/ascend910b"

    opp_host_path=${opp_dst_dir}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${architecture}
    ops_host_path=${ops_nn_dst_dir}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${architecture}
    create_ops_nn_opp_link ${opp_host_path} "libophost_nn.so" ${ops_host_path} "libophost_nn.so"

    opp_graph_inc_path=${opp_dst_dir}/built-in/op_graph/inc
    ops_graph_inc_path=${ops_nn_dst_dir}/built-in/op_graph/inc
    create_ops_nn_opp_link ${opp_graph_inc_path} "nn_ops.h" ${ops_graph_inc_path} "nn_ops.h"

    latest_api_inc_path=${arch_dst_dir}/include/aclnnop
    opp_api_inc_path=${opp_dst_dir}/include/aclnnop
    opp_api_level2_inc_path=${opp_dst_dir}/include/aclnnop/level2
    ops_api_inc_path=${ops_nn_dst_dir}/built-in/op_api/include/aclnnop
    create_ops_nn_opp_link_list ${latest_api_inc_path} ${ops_api_inc_path}
    create_ops_nn_opp_link_list ${opp_api_inc_path} ${ops_api_inc_path}
    create_ops_nn_opp_link_list ${opp_api_level2_inc_path} ${ops_api_inc_path}

    latest_api_lib_path=${arch_dst_dir}/lib64
    opp_api_lib_path=${opp_dst_dir}/lib64
    ops_api_lib_path=${ops_nn_dst_dir}/built-in/op_api/lib64/${architecture}
    create_ops_nn_opp_link_list ${opp_api_lib_path} ${ops_api_lib_path}
    create_ops_nn_opp_link_list ${latest_api_lib_path} ${ops_api_lib_path}
    chmod "${opp_original_perms}" "$opp_dst_dir"
    chmod "${arch_original_perms}" "$arch_dst_dir"
    logandprint "[INFO]:Create latest ops_nn successfully."
fi