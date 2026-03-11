#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
workspace_dir="$(dirname "$(dirname "$SCRIPT_DIR")")"
export WORKSPACE=${workspace_dir}
export ASCEND_3RD_LIB_PATH=${workspace_dir}/third_party

ENABLE_OPHOST=false
ENABLE_UTEST=false
ENABLE_SMOKE=false
PR_FILELIST=""

show_usage() {
cat << EOF
Usage: $0 [OPTIONS]

Options:
  -h, --help                    Show this help message and exit
  -u                            Build utest
  -s                            Build smoke test workflow:
                                 - build single op
                                 - build ascend950
                                 - if Ascend 910B: run A2 smoke
  -f <file>, --file <file>      Specify filelist for build
  --jit                         Enable ophost build

  Note: Options can be combined.
EOF
}

if [ "$#" -eq 0 ]; then
    ENABLE_OPHOST=true
    ENABLE_UTEST=true
    ENABLE_SMOKE=true
else
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --jit)
                ENABLE_OPHOST=true
                shift
                ;;
            -u)
                ENABLE_UTEST=true
                shift
                ;;
            -s)
                ENABLE_SMOKE=true
                shift
                ;;
            -f)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "[ERROR] -f requires an argument."
                    show_usage
                    exit 1
                fi
                PR_FILELIST="$2"
                shift 2
                ;;
            --file)
                if [[ "$2" == *=* ]]; then
                    PR_FILELIST="${2#*=}"
                    shift
                elif [ -n "$2" ] && [[ "$2" != -* ]]; then
                    PR_FILELIST="$2"
                    shift 2
                else
                    echo "[ERROR] --file requires an argument."
                    show_usage
                    exit 1
                fi
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                echo "Invalid option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
fi

if [[ "$ENABLE_OPHOST" == "false" && "$ENABLE_UTEST" == "false" && "$ENABLE_SMOKE" == "false" ]]; then
    ENABLE_OPHOST=true
    ENABLE_UTEST=true
    ENABLE_SMOKE=true
fi

get_pkg_name(){
    compile_package_name=$(ls "${workspace_dir}/build_out/" | grep -E "*.run$" | head -n1)
    echo "${compile_package_name}"
    return 0
}

set -e
if [[ -z "$PR_FILELIST" ]] || [[ ! -s "$PR_FILELIST" ]]; then
    PR_FILELIST="pr_filelist.txt"
    TARGET_URL="https://gitcode.com/cann/ops-nn.git"
    BASE_BRANCH_NAME=${1:-master}   # 默认对比 master 分支
    LOCAL_BRANCH=${2:-HEAD}         # 默认对比当前本地分支

    REMOTE_NAME=""
    for remote in $(git remote); do
        url=$(git remote get-url "$remote" 2>/dev/null)
        # 支持 https 和 git 协议，以及去掉末尾的 .git 进行模糊匹配
        if [[ "$url" == *"$TARGET_URL"* ]] || [[ "$url" == *"${TARGET_URL%.git}"* ]]; then
            REMOTE_NAME="$remote"
            break
        fi
    done

    # 如果没找到匹配的 URL，默认使用 'origin' (兼容直接 clone 主仓的情况)
    if [ -z "$REMOTE_NAME" ]; then
        echo "Warning: Specific remote URL not found. Defaulting to 'origin'."
        if ! git remote | grep -q "^origin$"; then
            echo "Error: No 'origin' remote found either. Please check your remotes."
            exit 1
        fi
        REMOTE_NAME="origin"
    fi

    echo "Detected Remote: $REMOTE_NAME (URL: $(git remote get-url $REMOTE_NAME))"

    echo "Fetching latest from $REMOTE_NAME..."
    git fetch "$REMOTE_NAME" --quiet --prune

    # 构建远程分支引用名 (例如: origin/master)
    REMOTE_REF="${REMOTE_NAME}/${BASE_BRANCH_NAME}"

    # 检查远程分支是否存在
    if ! git rev-parse --verify "$REMOTE_REF" >/dev/null 2>&1; then
        echo "Error: Remote branch '$REMOTE_REF' does not exist."
        echo "Available branches from $REMOTE_NAME:"
        git branch -r --list "${REMOTE_NAME}/*"
        exit 1
    fi

    MERGE_BASE_COMMIT=$(git merge-base "$REMOTE_REF" "$LOCAL_BRANCH")

    if [ -z "$MERGE_BASE_COMMIT" ]; then
        echo "Error: Could not find a common ancestor between $REMOTE_REF and $LOCAL_BRANCH."
        exit 1
    fi

    TARGET_COMMIT=$(git rev-parse "$LOCAL_BRANCH")

    echo "Calculating changed files..."
    CHANGED_FILES=$(git diff --name-only "$MERGE_BASE_COMMIT" "$TARGET_COMMIT" | sort -u)

    if [ -z "$CHANGED_FILES" ]; then
        echo "[warning] The file for the change cannot be found. Please check whether the code has been committed."
        exit 0
    fi

    # 输出结果
    echo "$CHANGED_FILES" > $PR_FILELIST
    echo "Saved changed files to $PR_FILELIST"
else
    echo "Using custom file: $PR_FILELIST"
fi

echo "Total: $(wc -l < $PR_FILELIST) files"
echo ""
echo "Preview:"
cat $PR_FILELIST

export BASE_PATH=$(pwd)
export BUILD_PATH="${BASE_PATH}/build"
rm -rf $BUILD_PATH
rm -rf $BASE_PATH/build_out

THREAD_NUM=$(grep -c ^processor /proc/cpuinfo)
if [[ "$ENABLE_OPHOST" == "true" ]]; then
    echo "==============================build jit============================================="
    echo "exec cmd: [bash build.sh --pkg --jit -j${THREAD_NUM} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}]"
    bash build.sh --pkg --jit -j${THREAD_NUM} --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH}
fi

if [[ "$ENABLE_UTEST" == "true" ]]; then
    echo "==============================build utest start======================================"
    echo "--------------------------build ophost ut start-----------------------------------"
    echo "exec cmd: [bash build.sh -u --ophost -f $PR_FILELIST --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j${THREAD_NUM}]"
    bash build.sh -u --ophost -f "$PR_FILELIST" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j${THREAD_NUM}
    echo "--------------------------build opapi ut start------------------------------------"
    echo "exec cmd: [bash build.sh -u --opapi -f $PR_FILELIST --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j${THREAD_NUM}]"
    bash build.sh -u --opapi -f "$PR_FILELIST" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j${THREAD_NUM}
    if [ "$BASE_BRANCH_NAME" = "master" ]; then
        echo "--------------------------build opgraph ut start-----------------------------------"
        echo "exec cmd: [bash build.sh -u --opgraph -f $PR_FILELIST --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j${THREAD_NUM}]"
        bash build.sh -u --opgraph -f "$PR_FILELIST" --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j${THREAD_NUM}
        echo "--------------------------build opkernel ut start-----------------------------------"
        echo "exec cmd: [bash scripts/ci/check_kernel_ut.sh $PR_FILELIST --no_cov]"
        bash scripts/ci/check_kernel_ut.sh $PR_FILELIST --no_cov | tee output.txt
        if grep -q "error happened" output.txt; then
            echo "[ERROR] Error happened in output check log"
            exit 1
        fi
    fi
    rm -rf build && rm -rf build_out
    echo "==============================build utest end===================================="
fi

rm -f run_test.log
if [[ "$ENABLE_SMOKE" == "true" ]]; then
    echo "==============================build single op===================================="
        SINGLE_FILE="single.tar.gz"
        need_check_example="false"
        rm -rf single/*
        echo "exec cmd: [bash scripts/ci/check_pkg.sh $PR_FILELIST -j${THREAD_NUM}]"
        bash scripts/ci/check_pkg.sh "$PR_FILELIST" -j${THREAD_NUM}
        if [[ -f "$SINGLE_FILE" && -s "$SINGLE_FILE" ]]; then
            need_check_example="true"
            rm single.tar.gz
        fi
        rm -rf build && rm -rf build_out

    echo "==============================compile ascend950==================================="
    # 获取机器架构
    ARCH=$(uname -m)
    # 判断是 x86 还是 ARM
    if [[ "$ARCH" == "x86_64" ]]; then
        echo "exec cmd: [bash scripts/ci/compile_ascend950_pkg.sh $PR_FILELIST -j${THREAD_NUM}]"
        bash scripts/ci/compile_ascend950_pkg.sh $PR_FILELIST -j${THREAD_NUM}
    elif [[ "$ARCH" == "aarch64" ]]; then
        echo "exec cmd: [bash scripts/ci/compile_ascend950_pkg.sh $PR_FILELIST -force_jit -j${THREAD_NUM}]"
        bash scripts/ci/compile_ascend950_pkg.sh $PR_FILELIST -force_jit -j${THREAD_NUM}
    else
        echo "[ERROR] Unsupported architecture: $ARCH"
        exit 1
    fi
    check_res=$?
    if [[ $check_res -ne 0 ]]; then
        echo "[ERROR] compile ascend950 failed"
        exit $check_res
    fi
    rm -rf build && rm -rf build_out

    if ! asys info -r=status 2>/dev/null | grep -q "Ascend 910B"; then
        echo "[Warning] The current platform does not support smoke tests, skipping A2 smoke"
    else
        echo "==============================build A2 smoke==================================="
        if [[ ${need_check_example} == "true" ]]; then
            # 执行受影响的算子
            echo "exec cmd: [bash scripts/ci/check_example.sh $PR_FILELIST]"
            bash scripts/ci/check_example.sh $PR_FILELIST  2>&1 | tee -a ./run_test.log
            if grep -w -e "FAIL" -e "errors" -e "fail" -e "failed" -e "error" -e "ERROR:" -e "Error" -e "error:" "./run_test.log"; then
                echo "[ERROR] run test case failed"
                exit 1
            fi
        fi
    fi
fi  

echo "==============================check experimental===================================="
rm -rf build && rm -rf build_out
echo "exec cmd: [bash scripts/ci/check_experimental_pkg.sh $PR_FILELIST]"
bash scripts/ci/check_experimental_pkg.sh "$PR_FILELIST"

if [ -f "${workspace_dir}/build_out/"*.run ]; then
    compile_package_name=$(get_pkg_name)
    if [[ -z "${compile_package_name}" ]]; then
        echo "[ERROR] Not find *.run in  ${workspace_dir}/build_out !"
        exit 1 
    fi
    chmod +x ./build_out/${compile_package_name}
    echo "exec cmd: [bash scripts/ci/check_experimental_example.sh $PR_FILELIST]"
    echo 'y' | bash ${compile_package_name} --quiet
    bash scripts/ci/check_experimental_example.sh $PR_FILELIST 2>&1 | tee -a ./run_test.log
    if grep -w -e "FAIL" -e "errors" -e "fail" -e "failed" -e "error" -e "ERROR:" -e "Error" -e "error:" "./run_test.log"; then
        echo "[ERROR] run test case failed"
        exit 1
    fi
fi
