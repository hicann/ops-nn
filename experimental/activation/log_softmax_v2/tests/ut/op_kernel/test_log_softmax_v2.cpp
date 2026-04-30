/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif
#include "../../../op_kernel/log_softmax_v2.cpp"
#include "../../../op_kernel/log_softmax_v2_tiling_data.h"
#include <cstdint>

using namespace std;

class log_softmax_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "log_softmax_v2_test SetUp\n" << endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./logsoftmax_data/");
    }
    static void TearDownTestCase()
    {
        cout << "log_softmax_v2_test TearDown\n" << endl;
    }
private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string log_softmax_v2_test::rootPath = "../../../../";
const std::string log_softmax_v2_test::dataPath = rootPath + "experimental/activation/log_softmax_v2/tests/ut/op_kernel/logsoftmax_data";

TEST_F(log_softmax_v2_test, test_case_0)
{
    system("cd ./logsoftmax_data/ && python3 gen_data.py '(32, 64)' 'float32'");
    std::string fileName = "./logsoftmax_data/float32_input_logsoftmax.bin";
    size_t xByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(LogSoftmaxV2TilingData);
    uint32_t blockDim = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(fileName, xByteSize, x, xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    LogSoftmaxV2TilingData* tilingDatafromBin = reinterpret_cast<LogSoftmaxV2TilingData*>(tiling);

    tilingDatafromBin->axis = 1;
    tilingDatafromBin->dims = 2;
    tilingDatafromBin->shape[0] = 32;
    tilingDatafromBin->shape[1] = 64;

    auto LogSoftmaxV2Kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::log_softmax_v2<0>(x, y, workspace, tiling);
    };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(LogSoftmaxV2Kernel,
        blockDim,
        x,
        y,
        workspace,
        (uint8_t *)(tilingDatafromBin));
    fileName = "./logsoftmax_data/float32_output_logsoftmax.bin";
    WriteFile(fileName, y, yByteSize);
    
    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./logsoftmax_data/ && python3 compare_data.py 'float32'");
}