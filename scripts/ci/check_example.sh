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
#    1. 二级目录下存在examples文件夹
#    2. examples文件夹中存在前缀为test_aclnn_的文件
#    3. 构建出valid_dirs清单

current_dir=$(pwd)
variables_cmake="cmake/variables.cmake"
op_category_list=$(grep -oP 'set\(OP_CATEGORY_LIST\s*\K".*"' $current_dir/$variables_cmake | sed 's/"//g')
IFS=' ' read -r -a op_categories <<< "$op_category_list"
valid_dirs=()
pr_file=$(realpath "${1:-pr_filelist.txt}")

for category in "${op_categories[@]}"
do  
    first_level="$current_dir/$category"
    if [ -d "$first_level" ]; then
        for second_level in "$first_level"/*/
        do
            examples_dir="$second_level""examples"
            # 检查 examples 目录是否存在
            if [ -d "$examples_dir" ]; then
                # 检查 examples 目录中是否有以 test_aclnn_ 开头的文件
                if ls "$examples_dir"/test_aclnn_* 1> /dev/null 2> /dev/null; then
                    dir_name=$(basename "$second_level")
                    valid_dirs+=("$dir_name")
                fi
            fi
        done
    fi
done


#搜索变更的文件名，将存在于valid_dirs清单中的算子保存到ops_name中

ops_name=()
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
            # 去重后添加到 ops_name 数组中
            if [[ ! " ${ops_name[@]} " =~ " $dir " ]]; then
                ops_name+=("$dir")
            fi
            break
        fi
    done

    #如果file_path为scripts/ci/mirror_update_time.txt，则将准备好的需验证算子列表加到ops_name中
    if [[ "$file_path" == "scripts/ci/mirror_update_time.txt" ]]; then

        operator_list=("conv3d_v2" "conv2d_v2") 
        
        for op in "${operator_list[@]}"; do
            op_exists=0
            for existing_op in "${ops_name[@]}"; do
                if [ "$existing_op" = "$op" ]; then
                    op_exists=1
                    break
                fi
            done
            if [ "$op_exists" -eq 0 ]; then
                ops_name+=("$op")
            fi
        done
    fi

done


#自定义算子包验证run_example
for name in "${ops_name[@]}"
do
    #安装指定路径的自定义算子包
    echo "--------------------------------"
    echo "${name}"
    ./single/cann-ops-nn-${name}-linux.*.run
    echo "[EXECUTE_COMMAND] bash build.sh --run_example $name eager cust --vendor_name=$name"
    bash build.sh --run_example $name eager cust --vendor_name=$name
    status=$?
    if [ $status -ne 0 ]; then
        echo "${name} example fail"
        exit 1
    fi
done