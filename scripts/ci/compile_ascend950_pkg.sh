#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

declare -a ALL_INCLUDE_FILES=()
declare -a TODO_QUEUE=()
TARGET_H_FILENAME=""
INCLUDE_TYPES=("--include=*.c" "--include=*.h" "--include=*.cpp" "--include=*.CPP" "--include=*.C" "--include=*.H")

run_build_command() {
    local cmd=$1
    echo "$cmd"
    if ! eval "$cmd"; then
        echo "build pkg error"
        exit 1
    fi
}

execute_run_file() {
    local pkg_type=$1
    local run_files=(./build_out/*.run)
    if [ ! -f "${run_files[0]}" ];then
        echo "no run pkg found"
        return 1
    fi
    chmod +x "${run_files[0]}"
    cmd=""${run_files[0]}" --install-path="/tmp""
    if [ $pkg_type == "builtin" ];then
        cmd+=" --full"
    fi
    echo $cmd
    if ! eval "$cmd"; then
        echo "execute pkg error"
        exit 1
    fi
}

is_in_array() {
    local elem="$1"
    local arr_name="$2"
    local -n arr="$arr_name"
    for item in "${arr[@]}"; do
        if [[ "$item" == "$elem" ]]; then
            return 0
        fi
    done
    return 1
}

recursive_find_includes() {
    local match_filename=$1
    local category_path=$2
    # 递归查找当前文件的所有包含文件
    local new_files=($(grep -rl "${INCLUDE_TYPES[@]}" "#include.*"$match_filename"" $category_path | sort -u))
    # 遍历本轮找到的新文件，过滤重复后加入队列和全局数组
    for file in "${new_files[@]}"; do
        local abs_file=$(realpath --relative-to=$(pwd) "$file")
        if ! is_in_array "$abs_file" "ALL_INCLUDE_FILES"; then
            ALL_INCLUDE_FILES+=("$abs_file")
            TODO_QUEUE+=("$abs_file")
        fi
    done

    # 递归终止
    if [[ ${#new_files[@]} -eq 0 ]]; then
        return
    fi

    local current_file
    while [[ ${#TODO_QUEUE[@]} -gt 0 ]]; do
        current_file="${TODO_QUEUE[0]}"
        TODO_QUEUE=("${TODO_QUEUE[@]:1}")
        local curr_filename=$(basename "$current_file")
        recursive_find_includes "$curr_filename" "$category_path"
    done
}

set -euo pipefail
# 获取算子类别列表
current_dir=$(pwd)
variables_cmake="cmake/variables.cmake"
op_category_list=$(
    perl -0777 -ne '
        if (/set\(OP_CATEGORY_LIST\s*(.*?)\)/s) {
            print $1
        }' \
        "$current_dir/$variables_cmake" \
    | grep -oP '"[^"]+"' \
    | sed 's/"//g' \
    | xargs
)
IFS=' ' read -r -a op_categories <<< "$op_category_list"
builtin_dirs=()
experimental_dirs=()
force_jit="false"
pr_file="pr_filelist.txt"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -force_jit)
            force_jit="true"
            if [[ ${2+x} && -n "$2" && "$2" != --* ]]; then
                shift
            fi
            shift
            ;;
        -pr_file)
            if [[ ${2+x} && -n "$2" && "$2" != --* ]]; then
                pr_file="$2"
                shift
            else
                echo "-pr_file use default value: $pr_file"
            fi
            shift
            ;;
        *)
            shift
            ;;
    esac
done


# 遍历每个分类 统计builtin_dirs和experimental_dirs算子目录列表
for category in "${op_categories[@]}"
do
    builtin_category_path="$current_dir/$category"
    if [ -d "$builtin_category_path" ]; then
        for dir in "$builtin_category_path"/*/
        do
            dir_name=$(basename "$dir")
            # 如果子目录名中有common或utils这类pattern，则表示不是算子名，跳过
            if [[ "$dir_name" != *"common"* && "$dir_name" != *"utils"* ]]; then
                builtin_dirs+=("$dir_name")
            fi
        done
    fi
    exper_category_path="$current_dir/experimental/$category"
    if [ -d "$exper_category_path" ]; then
        for dir in "$exper_category_path"/*/
        do
            dir_name=$(basename "$dir")
            if [[ "$dir_name" != *"common"* && "$dir_name" != *"utils"* ]]; then
                experimental_dirs+=("$dir_name")
            fi
        done
    fi
done

builtin_ops_name=()
experimental_ops_name=()
build_all=${build_all:-0}

mapfile -t lines < ${pr_file}

for file_path in "${lines[@]}"
do
    file_path=$(echo "$file_path" | xargs)
    if [ -z "$file_path" ]; then
        continue
    fi
    if [[ "$file_path" == *.md ]]; then
        continue
    fi
    for category in "${op_categories[@]}"
    do
        REGEX_COMMON="^$category/(.*/)?[^/]*common[^/]*/.*\.[hH]$"
        REGEX_UTIL="^$category/(.*/)?[^/]*utils[^/]*/.*\.[hH]$"
        # 如果修改的文件时算子类别下的common或util等公用头文件 就追溯使用该文件的对应算子文件
        if [[ "$file_path" =~ $REGEX_COMMON || "$file_path" =~ $REGEX_UTIL ]]; then
            file_name=$(basename "$file_path")
            recursive_find_includes "$file_name" "$current_dir/$category"
            break
        fi
    done
done

for file_path in "${ALL_INCLUDE_FILES[@]}"; do
    if ! is_in_array "$file_path" "lines"; then
        lines+=("$file_path")
    fi
done

for file_path in "${lines[@]}"
do
    file_path=$(echo "$file_path" | xargs)
    if [ -z "$file_path" ]; then
        continue
    fi
    if [[ "$file_path" == *.md ]]; then
        continue
    fi
    for dir in "${builtin_dirs[@]}"
    do
        # 如果算子目录下的arch35子目录中的文件被修改 则触发A5 kernel编译
        if [[ "$file_path" == *"/$dir/"*"/arch35/"* ]]; then
            if [[ ! " ${builtin_ops_name[@]} " =~ " $dir " ]]; then
                builtin_ops_name+=("$dir")
            fi
            break
        fi
    done
    for dir in "${experimental_dirs[@]}"
    do
        if [[ "$file_path" == "experimental/"*"/$dir/"*"/arch35/"* ]]; then
            if [[ ! " ${experimental_ops_name[@]} " =~ " $dir " ]]; then
                experimental_ops_name+=("$dir")
            fi
            break
        fi
    done
    # 如果被修改的文件在common或cmake目录下 则触发整仓jit编译
    if [[ "$file_path" == "common/"* || "$file_path" == "cmake/"* ]]; then
        build_all=1
    fi
done

echo "related op: ${builtin_ops_name[*]}"
echo "related experimental op: ${experimental_ops_name[*]}"
echo "need build all: ${build_all}"

cann_3rd_lib_path="/home/jenkins/opensource"
a5_soc="ascend950"

if [[ ${#builtin_ops_name[@]} -gt 0 && "$force_jit" = "false" ]]; then
    builtin_ops_str=$(IFS=,; echo "${builtin_ops_name[*]}")
    build_cmd="bash build.sh --pkg --ops=$builtin_ops_str --soc=$a5_soc -j16 --cann_3rd_lib_path=$cann_3rd_lib_path"
    run_build_command "$build_cmd"
    execute_run_file "custom"
fi

if [[ ${#experimental_ops_name[@]} -gt 0 && "$force_jit" = "false" ]]; then
    experimental_ops_str=$(IFS=,; echo "${experimental_ops_name[*]}")
    build_cmd="bash build.sh --pkg --experimental --ops=$experimental_ops_str --soc=$a5_soc -j16 --cann_3rd_lib_path=$cann_3rd_lib_path"
    run_build_command "$build_cmd"
    execute_run_file "custom"
fi

if [ ${build_all} -eq 1 ]; then
    build_cmd="bash build.sh --pkg --jit --soc=$a5_soc -j16 --cann_3rd_lib_path=$cann_3rd_lib_path"
    run_build_command "$build_cmd"
    execute_run_file "builtin"
fi

exit 0
