#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

BRed="\033[1;31m"
Color_Off="\033[0m"

# Log error
function LOG_ERROR() {
    local assert_msg=${1}
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BRed}[ERROR] ${date_time} ${assert_msg}${Color_Off}"
}

function DP_ASSERT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    local log_flag=${4:-"true"}
    local log_path=${5}
    if [ "${actual_value}" != "${expect_value}" ]; then
        if [ -n "${log_path}" ] && [ -f "${log_path}" ]; then
            cat "${log_path}"
        fi
        LOG_ERROR "${assert_msg} is failed."
        exit 1
    else
        if [ "${log_flag}" = "true" ]; then
            echo "${assert_msg} is success."
        fi
    fi
}


echo "WORKSPACE: ${WORKSPACE}"
echo "TARGET_BRANCH: ${TARGET_BRANCH}"
echo "OS_TYPE: ${OS_TYPE}"
echo "task_name: ${task_name}"

cd ${WORKSPACE}
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" == *ubuntu24* ]]; then
    if [[ "${TARGET_BRANCH}" = "master" ]]; then
        sudo update-alternatives --set gcc /usr/bin/gcc-15
    else
        sudo update-alternatives --set gcc /usr/bin/gcc-14
    fi
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
fi

if gcc --version | head -n1 | grep -q "15\."; then
    rm -rf /home/jenkins/opensource/lib_cache
    if [ -d  /home/jenkins/opensource/gcc15 ];then
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15/lib_cache /home/jenkins/opensource/lib_cache
    elif [ -d  /home/jenkins/opensource/gcc15x86 ];then
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/abseil-cpp
        rm -rf /home/jenkins/opensource/gcc15x86/lib_cache/device/abseil-cpp
        ln -s /home/jenkins/opensource/gcc15x86/lib_cache /home/jenkins/opensource/lib_cache
    fi
else
    gcc --version
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi

gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e
if [[ "${task_name}" =~ x86_compile_ubuntu24 ]] && [ -f "build_out/"*.run ] && [ "${TARGET_BRANCH}" == master ]; then
    echo "api-check=compile" >> "${ATOMGIT_OUTPUT}"
else
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
fi
non_skip_count=$(grep -vE '(\.md$|^tests/)' "${WORKSPACE}/pr_filelist.txt" | grep -cv '^$')
if [ "${non_skip_count}" -eq 0 ]; then
    echo "pr_filelist.txt only contains .md or tests/ files, skip build"
    mkdir -p build_out
    touch build_out/skip_build.run
    touch single.tar.gz
    exit 0
fi
if [[ "${task_name}" == compile_single* ]]; then
    echo "buildout_package=single.tar.gz" >> $ATOMGIT_OUTPUT
else
    echo "buildout_package=build_out/*.run" >> $ATOMGIT_OUTPUT
fi

case "${task_name}" in
    x86_compile)
        bash build.sh --pkg --jit -f "pr_filelist.txt" --cann_3rd_lib_path=/home/jenkins/opensource -j16
        DP_ASSERT_EQUAL $? 0 "build ${task_name}"
        echo "exec cmd: [bash build.sh --pkg --jit -f --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        ;;
    x86_compile_ubuntu24)
        sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
        bash build.sh --pkg --jit -f "pr_filelist.txt" --cann_3rd_lib_path=/home/jenkins/opensource -j16
        DP_ASSERT_EQUAL $? 0 "build ${task_name}"
        echo "exec cmd: [bash build.sh --pkg --jit -f --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        ;;
    X86_monitor_910b)
        if [ "${TARGET_BRANCH}" = "master" ];then
            bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -f "pr_filelist.txt" -j16 --soc=ascend910b
            DP_ASSERT_EQUAL $? 0 "build ${task_name}"
            echo "exec cmd: [bash build.sh --pkg -f --jit -j16 --soc=ascend910b]"
        else
            echo "not need build monitor"
            mkdir build_out
            touch build_out/cann-ops-nn_linux-x86_64.run
        fi
        ;;
    X86_monitor_910c)
        if [ "${TARGET_BRANCH}" = "master" ];then
            bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -f "pr_filelist.txt" -j16 --soc=ascend910_93
            DP_ASSERT_EQUAL $? 0 "build ${task_name}"
            echo "exec cmd: [bash build.sh --pkg -f --jit -j16 --soc=ascend910_93]"
        else
            echo "not need build monitor"
            mkdir build_out
            touch build_out/cann-ops-nn_linux-x86_64.run
        fi
        ;;
    X86_monitor_950)
        if [ "${TARGET_BRANCH}" = "master" ];then
            bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -f "pr_filelist.txt" -j16 --soc=ascend950
            DP_ASSERT_EQUAL $? 0 "build ${task_name}"
            echo "exec cmd: [bash build.sh --pkg -f --jit -j16 --soc=ascend950]"
        else
            echo "not need build monitor"
            mkdir build_out
            touch build_out/cann-ops-nn_linux-x86_64.run
        fi
        ;;
    Compile_Ascend_X86_950*)
        export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
        bash scripts/ci/compile_ascend950_pkg.sh "pr_filelist.txt" "-j32" "--no_force"
        DP_ASSERT_EQUAL $? 0 "build ${task_name}"
        compile_package_name=$(ls "${WORKSPACE}/build_out/" |grep -E "*.run$"|head -n1)
        if [[ -z "${compile_package_name}" ]]; then
            echo "not need build 950"
            mkdir build_out
            touch build_out/cann-ops-nn-950_linux-x86_64.run
        fi
        ;;
    Pre_compile)
        bash build.sh --pkg --ops="fatrelu_mul" --cann_3rd_lib_path=/home/jenkins/opensource
        DP_ASSERT_EQUAL $? 0 "build ${task_name}"
        echo "build fatrelu_mul"
        ls build_out
        mv build_out/*.run ${WORKSPACE}/build_out/cann-ops-nn-fatrelu_mul_linux-aarch64.run
        ls build_out
        ;;
    compile_single*)
        if [ "${TARGET_BRANCH}" = "master" ];then
            export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
            bash scripts/ci/check_pkg.sh "pr_filelist.txt" "-j16"
            DP_ASSERT_EQUAL $? 0 "build ${task_name}"
            echo "exec cmd: [bash scripts/ci/check_pkg.sh pr_filelist.txt]"
        fi
        if [ ! -f ${WORKSPACE}/single.tar.gz ];then
            echo "not need build single"
            touch single.tar.gz
        fi
        ;;
    arm_compile*)
        bash build.sh --pkg --jit -f "pr_filelist.txt" --cann_3rd_lib_path=/home/jenkins/opensource -j16
        DP_ASSERT_EQUAL $? 0 "build ${task_name}"
        echo "exec cmd: [bash build.sh --pkg --jit -f --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        ;;
    Compile_Ascend_experimental)
        sh scripts/ci/check_experimental_pkg.sh "pr_filelist.txt"
        DP_ASSERT_EQUAL $? 0 "build ${task_name}"
        echo "exec cmd: [sh scripts/ci/check_experimental_pkg.sh pr_filelist.txt]"
        if [ ! -f "build_out/"*.run ]; then
            mkdir -p build_out
            touch build_out/cann-ops-nn-experimental_linux-aarch64.run
        fi
        ;;
    Compile_Ascend_ARM_950)
        export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
        bash scripts/ci/compile_ascend950_pkg.sh "pr_filelist.txt" "-j16" "-force_jit" "--no_force"
        DP_ASSERT_EQUAL $? 0 "build ${task_name}"
        compile_package_name=$(ls "${WORKSPACE}/build_out/" |grep -E "*.run$"|head -n1)
        if [[ -z "${compile_package_name}" ]]; then
            echo "not need build 950"
            mkdir build_out
            touch build_out/cann-ops-nn-950_linux-aarch64.run
        fi
        ;;
    Compile_Ascend_X86_mobile_station)
        if [ "${TARGET_BRANCH}" = "master" ];then
            bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=/home/jenkins/opensource -j16
            DP_ASSERT_EQUAL $? 0 "build ${task_name}"
            echo "exec cmd: [bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        else
            echo "not need build mobile_station"
            mkdir build_out
            touch build_out/cann-ops-nn-kirinx90_linux-x86_64.run
            exit 0
        fi
        ;;
    Compile_Ascend_X86_mobile_station_ubuntu24)
        if [ "${TARGET_BRANCH}" = "master" ];then
            bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=/home/jenkins/opensource -j16
            DP_ASSERT_EQUAL $? 0 "build ${task_name}"
            echo "exec cmd: [bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        else
            echo "not need build mobile_station"
            mkdir build_out
            touch build_out/cann-ops-nn-kirinx90_linux-x86_64.run
            exit 0
        fi
        ;;
    Compile_Ascend_X86_mobile_station_9030_ubuntu24)
        if [ "${TARGET_BRANCH}" = "master" ];then
            bash build.sh --pkg --soc=kirin9030 --cann_3rd_lib_path=/home/jenkins/opensource -j16
            DP_ASSERT_EQUAL $? 0 "build ${task_name}"
            echo "exec cmd: [bash build.sh --pkg --soc=kirin9030 --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        else
            echo "not need build mobile_station"
            mkdir build_out
            touch build_out/cann-ops-nn-kirin9030_linux-x86_64.run
            exit 0
        fi
        ;;
esac


if [ ! -f "build_out/"*.run ]; then
    mkdir -p build_out
    touch build_out/cann-ops-nn-test_linux-aarch64.run
fi
