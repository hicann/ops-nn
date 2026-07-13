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
set -x
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
sudo update-alternatives --set gcc /usr/bin/gcc-14
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e
if [ "$TARGET_BRANCH" = "master" ];then
    case "${ut_type}" in
        ophost)
            bash build.sh -u --cov --ophost --cann_3rd_lib_path=/home/jenkins/opensource -f "pr_filelist.txt" -j16
            ret=$?
            coverage_save="true"
            ;;
        opapi)
            bash build.sh -u --cov --opapi --cann_3rd_lib_path=/home/jenkins/opensource -f "pr_filelist.txt" -j16
            ret=$?
            coverage_save="true"
            ;;
        opgraph)
            bash build.sh -u --opgraph --cov --cann_3rd_lib_path=/home/jenkins/opensource -f "pr_filelist.txt" -j16
            ret=$?
            coverage_save="true"
            ;;
        opkernel)
            bash scripts/ci/check_kernel_ut.sh "pr_filelist.txt" "${repo_name}" "-j16" | tee output.txt
            ret=$?
            coverage_save="true"
            ;;
    esac
else
    case "${ut_type}" in
        ophost)
            bash build.sh -u --ophost --cann_3rd_lib_path=/home/jenkins/opensource -f "pr_filelist.txt" -j16
            ret=$?
            coverage_save="true"
            ;;
        opapi)
            bash build.sh -u --opapi --cann_3rd_lib_path=/home/jenkins/opensource -f "pr_filelist.txt" -j16
            ret=$?
            coverage_save="true"
            ;;
        *)
            echo "Skip UT test execution for ${ut_type} on non-master branch"
            exit 0
            ;;

    esac
fi

if [ $ret -ne 200 ] && [ $ret -ne 0 ]; then
    echo "run ut fail"
    exit 1
fi
if [ $ret -eq 0 ]; then
    if [ "$coverage_save" = "true" ];then
    echo "ut_process=coverage" >> $ATOMGIT_OUTPUT
    else
    echo "ut_process=ut_cov" >> $ATOMGIT_OUTPUT
    fi
fi
exit 0
