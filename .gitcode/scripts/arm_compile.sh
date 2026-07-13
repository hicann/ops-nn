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

if [[ "${task_name}" == compile_single* ]]; then
    echo "buildout_package=single.tar.gz" >> $ATOMGIT_OUTPUT
else
    echo "buildout_package=build_out/*.run" >> $ATOMGIT_OUTPUT
fi

case "${task_name}" in
    Pre_compile)
        bash build.sh --pkg --ops="fatrelu_mul" --cann_3rd_lib_path=/home/jenkins/opensource
        echo "build fatrelu_mul"
        ls build_out
        mv build_out/*.run ${WORKSPACE}/build_out/cann-ops-nn-fatrelu_mul_linux-aarch64.run
        ls build_out
        exit 0
        ;;
    compile_single*)
        if [ "${target_branch}" = "master" ];then
            export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
            bash scripts/ci/check_pkg.sh "pr_filelist.txt" "-j16"
            echo "exec cmd: [bash scripts/ci/check_pkg.sh pr_filelist.txt]"
        fi
        if [ ! -f ${WORKSPACE}/single.tar.gz ];then
            echo "not need build single"
            touch single.tar.gz
        fi
        ;;
    arm_compile*)
        bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16
        echo "exec cmd: [bash build.sh --pkg --jit --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
        exit 0
        ;;
    Compile_Ascend_experimental)
        sh scripts/ci/check_experimental_pkg.sh "pr_filelist.txt"
        echo "exec cmd: [sh scripts/ci/check_experimental_pkg.sh pr_filelist.txt]"
        if [ ! -f "build_out/"*.run ]; then
            mkdir -p build_out
            touch build_out/cann-ops-nn-experimental_linux-aarch64.run
        fi
        exit 0
        ;;
    Compile_Ascend_ARM_950)
        export ASCEND_3RD_LIB_PATH=/home/jenkins/opensource
        bash scripts/ci/compile_ascend950_pkg.sh "pr_filelist.txt" "-j16" "-force_jit" "--no_force"
        compile_package_name=$(ls "${WORKSPACE}/build_out/" |grep -E "*.run$"|head -n1)
        if [[ -z "${compile_package_name}" ]]; then
            echo "not need build 950"
            mkdir build_out
            touch build_out/cann-ops-nn-950_linux-aarch64.run
        fi
        exit 0
        ;;
esac

if [ ! -f "build_out/"*.run ]; then
    mkdir -p build_out
    touch build_out/cann-ops-nn-test_linux-aarch64.run
fi
