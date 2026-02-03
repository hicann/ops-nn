#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
fi

source $_ASCEND_INSTALL_PATH/set_env.sh
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/devlib

function main {
    # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm ./input/*.bin
    rm ./output/*.bin

    # 2. 生成输入数据和真值数据
    cd $CURRENT_DIR
    python3 scripts/gen_data.py
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Generate input data failed!"
        return 1
    fi
    echo "[INFO]: Generate input data success!"

    # 3. 编译acl可执行文件
    cd $CURRENT_DIR
    rm -rf build
    mkdir -p build
    cd build
    cmake ../src -DCMAKE_SKIP_RPATH=TRUE
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Cmake failed!"
        return 1
    fi
    echo "[INFO]: Cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Make failed!"
        return 1
    fi
    echo "[INFO]: Make success!"

    # 4. 运行可执行文件
    export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/custom_nn/op_api/lib:$LD_LIBRARY_PATH
    cd $CURRENT_DIR/output
    echo "[INFO]: Execute op!"
    file_path=output_msg.txt
    ./execute_weight_quant_batch_matmul_experiment_op | tee $file_path
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Acl executable run failed! please check your project!"
        return 1
    fi
    echo "[INFO]: Acl executable run success!"

    # 5. 比较真值文件
    cd $CURRENT_DIR
    python3 scripts/verify_result.py output/output_z.bin output/golden.bin
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Verify result failed!"
        return 1
    fi
    # 6. 生成算子性能数据
    cd $CURRENT_DIR/output
    msprof --output=./msprof_result ./execute_weight_quant_batch_matmul_experiment_op
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Msprof failed to collect the performance data of the acl execution program! please check your project!"
        return 1
    fi
    echo "[INFO]: Msprof successfully collected the performance data of the acl execution program!"
}

main
