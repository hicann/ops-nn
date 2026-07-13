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

cd ${WORKSPACE}
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" == *ubuntu24* ]]; then
    sudo update-alternatives --set gcc /usr/bin/gcc-14
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
fi
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e
case "${task_name}" in
    x86_compile*)
        bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16
        echo "exec cmd: [bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16]"
        exit 0
        ;;
    x86_compile_ubuntu24)
        sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
        bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16
        echo "exec cmd: [bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        exit 0
        ;;
    X86_monitor_910b)
        bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16 --soc=ascend910b
        echo "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend910b]"
        exit 0
        ;;
    X86_monitor_910c)
        bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16 --soc=ascend910_93
        echo "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend910_93]"
        exit 0
        ;;
    X86_monitor_950)
        bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16 --soc=ascend950
        echo "exec cmd: [bash build.sh --pkg --jit -j16 --soc=ascend950]"
        ;;
    Compile_Ascend_X86_950*)
        export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
        bash scripts/ci/compile_ascend950_pkg.sh "pr_filelist.txt" "-j16" "--no_force"
        compile_package_name=$(ls "${WORKSPACE}/build_out/" |grep -E "*.run$"|head -n1)
        if [[ -z "${compile_package_name}" ]]; then
            echo "not need build 950"
            mkdir build_out
            touch build_out/cann-ops-nn-950_linux-x86_64.run
        fi
        exit 0
        ;;
esac

case "${task_name}|${GIT_TARGET_BRANCH}" in
    Compile_Ascend_X86_mobile_station|master)
        wget -nv https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.beta005/cann-bisheng-compiler_9.0.0_linux-x86_64.run
        chmod +x *.run
        sudo -u jenkins ./*.run --full --quiet --install-path=/home/jenkins/Ascend
        bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=/home/jenkins/opensource -j16
        echo "exec cmd: [bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        ;;
    Compile_Ascend_X86_mobile_station|*)
        echo "not need build single"
        mkdir single
        touch single.tar.gz
        ;;
esac
case "${task_name}|${GIT_TARGET_BRANCH}" in
    Compile_Ascend_X86_mobile_station_ubuntu24|master)
        wget -nv https://kiri-obs.obs.cn-north-4.myhuaweicloud.com/Cann%20Large%20Model%20Foundation%208.5.0.beta005/cann-bisheng-compiler_9.0.0_linux-x86_64.run
        chmod +x *.run
        sudo -u jenkins ./*.run --full --quiet --install-path=/home/jenkins/Ascend
        bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=/home/jenkins/opensource -j16
        echo "exec cmd: [bash build.sh --pkg --soc=kirinx90 --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        ;;
    Compile_Ascend_X86_mobile_station|*)
        echo "not need build single"
        mkdir single
        touch single.tar.gz
        ;;
esac

if [ ! -f "build_out/"*.run ]; then
    mkdir -p build_out
    touch build_out/cann-ops-nn-test_linux-aarch64.run
fi
