#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

# 查找符合条件的二级目录：
#    1. 一级目录由cmake/variables.cmake中的OP_CATEGORY_LIST直接提供
#    2. 二级目录下需要包含/tests/ut/op_kernel文件夹
#    3. 构建出valid_dirs清单

current_dir=$(pwd)
variables_cmake="cmake/variables.cmake"
pr_file=$(realpath "${1:-pr_filelist.txt}")
# 提取 OP_CATEGORY_LIST 的值
op_category_list=$(grep -oP 'set\(OP_CATEGORY_LIST\s*\K".*"' $current_dir/$variables_cmake | sed 's/"//g')
IFS=' ' read -r -a op_categories <<< "$op_category_list"

#op_categories=("activation" "conv" "foreach" "vfusion" "index" "loss" "matmul" "norm" "optim" "pooling" "quant" "rnn" "control")

valid_dirs=()
for category in "${op_categories[@]}"
do
    category_path="$current_dir/$category"
    if [ -d "$category_path" ]; then
        for dir in "$category_path"/*/
        do
            if [ -d "$dir/tests/ut/op_kernel" ]; then
                dir_name=$(basename "$dir")
                if [[ "$dir_name" != *"common"* ]]; then
                    valid_dirs+=("$dir_name")
                fi
            fi
        done
    fi
done


#搜索变更文件清单，将存在于valid_dirs清单中的算子保存到ops_name中，路径中需要出现'op_kernel'

ops_name=()
ops_name_mirror=()

mapfile -t lines < ${pr_file}
for file_path in "${lines[@]}"
do
    # 去除前后空格
    file_path=$(echo "$file_path" | xargs)
    # 跳过空行
    if [ -z "$file_path" ]; then
        continue
    fi
    # 跳过 .md 后缀的文件
    if [[ "$file_path" == *.md ]]; then
        continue
    fi
    # 检查该路径是否包含 valid_dirs 中的任意一个目录名
    for dir in "${valid_dirs[@]}"
    do
        if [[ "$file_path" == *"/$dir/"* ]]; then
            if [[ "$file_path" == *"op_kernel"* ]]; then
                # 去重后添加到 ops_name 数组中
                if [[ ! " ${ops_name[@]} " =~ " $dir " ]]; then
                    ops_name+=("$dir")
                fi
            fi
            break
        fi
    done

    #如果file_path为scripts/ci/mirror_update_time.txt，则将准备好的需验证算子列表加到ops_name_mirror中
    if [[ "$file_path" == "scripts/ci/mirror_update_time.txt" ]]; then
        operator_list=("conv2d_v2" "conv3d_v2")
        ops_name_mirror=("${operator_list[@]}")
    fi

done

supportedSocVersion=("ascend910b" "ascend310p" "ascend950")

for name in "${ops_name[@]}"
do
    for soc_version in "${supportedSocVersion[@]}"
    do
        echo "[EXECUTE_COMMAND] bash build.sh -u --opkernel --ops=$name --cov --soc=$soc_version"
        bash build.sh -u --opkernel --ops=$name --cov --soc=$soc_version --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        status=$?
        if [ $status -ne 0]; then
            echo "${name} kernel ut fail"
            exit 1
        fi
    done
done

#新增对需验证算子列表的ut校验，包括host,api,kernel
for name in "${ops_name_mirror[@]}"
do
    for soc_version in "${supportedSocVersion[@]}"
    do
        echo "[EXECUTE_COMMAND] bash build.sh -u --ops=$name --cov --soc=$soc_version"
        bash build.sh -u --ops=$name --cov --soc=$soc_version --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        status=$?
        if [ $status -ne 0]; then
            echo "${name} ut fail"
            exit 1
        fi
    done
done