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

get_pkg_name(){
    compile_package_name=$(ls "${workspace_dir}/build_out/" | grep -E "*.run$" | head -n1)
    echo "${compile_package_name}"
    return 0
}

set -e
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
echo "$CHANGED_FILES" > pr_filelist.txt

echo "Done! Changed files saved to pr_filelist.txt"
echo "Total: $(wc -l < pr_filelist.txt) files"
echo ""
echo "Preview:"
cat pr_filelist.txt

export BASE_PATH=$(pwd)
export BUILD_PATH="${BASE_PATH}/build"
rm -rf $BUILD_PATH
rm -rf $BASE_PATH/build_out

THREAD_NUM=$(grep -c ^processor /proc/cpuinfo)
echo "==============================pre build========================================="
rm -rf build && rm -rf build_out
echo "exec cmd: [bash build.sh --pkg --ops=fatrelu_mul --vendor_name=fatrelu_mul -j${THREAD_NUM}]"
bash build.sh --pkg --ops="fatrelu_mul" --vendor_name="fatrelu_mul"
mkdir -p ${SCRIPT_DIR}/tmp && rm -rf ${script_dir}/tmp/*
mv build_out/cann-ops-nn-fatrelu_mul*.run ${SCRIPT_DIR}/tmp/cann-ops-nn-fatrelu_mul_linux-aarch64.run
rm -rf build && rm -rf build_out

echo "==============================build jit============================================="
echo "exec cmd: [bash build.sh --pkg --jit -j${THREAD_NUM}]"
bash build.sh --pkg --jit -j ${THREAD_NUM}
rm -rf build && rm -rf build_out

echo "==============================build single op===================================="
SINGLE_FILE="single.tar.gz"
need_check_example="false"
rm -rf single/*
echo "exec cmd: [bash scripts/ci/check_pkg.sh pr_filelist.txt]"
bash scripts/ci/check_pkg.sh "pr_filelist.txt"
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
    echo "exec cmd: [bash scripts/ci/compile_ascend950_pkg.sh pr_filelist.txt]"
    bash scripts/ci/compile_ascend950_pkg.sh
elif [[ "$ARCH" == "aarch64" ]]; then
    echo "exec cmd: [bash scripts/ci/compile_ascend950_pkg.sh pr_filelist.txt -force_jit]"
    bash scripts/ci/compile_ascend950_pkg.sh "-force_jit"
fi
check_res=$?
if [[ $check_res -ne 0 ]]; then
    echo "[ERROR] compile ascend950 failed"
    exit $check_res
fi
rm -rf build && rm -rf build_out

echo "==============================build utest start======================================"
echo "--------------------------build ophost ut start-----------------------------------"
echo "exec cmd: [bash build.sh -u --ophost -f pr_filelist.txt -j${THREAD_NUM}]"
bash build.sh -u --ophost -f "pr_filelist.txt" -j ${THREAD_NUM}
echo "--------------------------build opapi ut start------------------------------------"
echo "exec cmd: [bash build.sh -u --opapi -f pr_filelist.txt -j${THREAD_NUM}]"
bash build.sh -u --opapi -f "pr_filelist.txt" -j ${THREAD_NUM}
if [ "$BASE_BRANCH_NAME" = "master" ]; then
    echo "--------------------------build opgraph ut start-----------------------------------"
    echo "exec cmd: [bash build.sh -u --opgraph -f pr_filelist.txt -j${THREAD_NUM}]"
    bash build.sh -u --opgraph -f "pr_filelist.txt" -j ${THREAD_NUM}
    echo "--------------------------build opkernel ut start-----------------------------------"
    echo "exec cmd: [bash scripts/ci/check_kernel_ut.sh pr_filelist.txt --no_cov]"
    bash scripts/ci/check_kernel_ut.sh pr_filelist.txt --no_cov | tee output.txt
    if grep -q "error happened" output.txt; then
        echo "[ERROR] Error happened in output check log"
        exit 1
    fi
fi
rm -rf build && rm -rf build_out
echo "==============================build utest end===================================="

if ! asys info -r=status | grep "Ascend 910B"; then
    echo "[Warning] The current platform does not support smoke tests, and the process ends"
    exit 0
fi

rm -f run_test.log
if [[ ${need_check_example} == "true" ]]; then
    echo "==============================build A2 smoke==================================="
    # 执行受影响的算子
    echo "exec cmd: [bash scripts/ci/check_example.sh pr_filelist.txt]"
    bash scripts/ci/check_example.sh pr_filelist.txt  2>&1 | tee -a ./run_test.log
    # 单独执行nn仓的fatrelu_mul
    chmod +x ${SCRIPT_DIR}/tmp/* && ${SCRIPT_DIR}/tmp/*.run 2>&1 | tee -a ./run_test.log
    echo "exec cmd: [bash build.sh --run_example fatrelu_mul eager cust --vendor_name=fatrelu_mul]"
    bash build.sh --run_example fatrelu_mul eager cust --vendor_name=fatrelu_mul 2>&1 | tee -a ./run_test.log
    if grep -w -e "FAIL" -e "errors" -e "fail" -e "failed" -e "error" -e "ERROR:" -e "Error" -e "error:" "./run_test.log"; then
        echo "[error] run test case failed"
        exit 1
    fi
fi

echo "==============================check experimental===================================="
rm -rf build && rm -rf build_out
echo "exec cmd: [bash scripts/ci/check_experimental_pkg.sh pr_filelist.txt]"
bash scripts/ci/check_experimental_pkg.sh "pr_filelist.txt"
if [ -f "${workspace_dir}/build_out/"*.run ]; then
    compile_package_name=$(get_pkg_name)
    if [[ -z "${compile_package_name}" ]]; then
        echo "[ERROR] Not find *.run in  ${workspace_dir}/build_out !"
        exit 1 
    fi
    chmod +x ./build_out/${compile_package_name}
    echo "exec cmd: [bash scripts/ci/check_experimental_example.sh pr_filelist.txt]"
    echo 'y' | bash ${compile_package_name} --quiet && bash scripts/ci/check_experimental_example.sh pr_filelist.txt 2>&1 | tee -a ./run_test.log
    if grep -w -e "FAIL" -e "errors" -e "fail" -e "failed" -e "error" -e "ERROR:" -e "Error" -e "error:" "./run_test.log"; then
        echo "[ERROR] run test case failed"
        exit 1
    fi
fi
