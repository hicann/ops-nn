/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

#ifndef DTYPE_INPUT0
#define DTYPE_INPUT0 float
#endif

#include "../../../op_kernel/adam_apply_one.cpp"
#include "../../../op_kernel/adam_apply_one_tiling_key.h"
#include "../../../op_kernel/adam_apply_one_tiling_data.h"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void adam_apply_one(GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR input3,
                                                     GM_ADDR input4, GM_ADDR mul0_x, GM_ADDR mul1_x, GM_ADDR mul2_x,
                                                     GM_ADDR mul3_x, GM_ADDR add2_y, GM_ADDR output0, GM_ADDR output1,
                                                     GM_ADDR output2, GM_ADDR workspace, GM_ADDR tiling);

class adam_apply_one_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "adam_apply_one SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "adam_apply_one TearDown\n" << endl; }
};

TEST_F(adam_apply_one_test, test_adam_apply_one_dynamic)
{
    size_t size = 32 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(AdamApplyOneTilingData);
    uint32_t blockDim = 1;

    uint8_t* input0 = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* input1 = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* input2 = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* input3 = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* input4 = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* mul0_x = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* mul1_x = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* mul2_x = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* mul3_x = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* add2_y = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* output0 = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* output1 = (uint8_t*)AscendC::GmAlloc(size);
    uint8_t* output2 = (uint8_t*)AscendC::GmAlloc(size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AdamApplyOneTilingData* tilingDatafromBin = reinterpret_cast<AdamApplyOneTilingData*>(tiling);

    tilingDatafromBin->smallCoreDataNum = 128;
    tilingDatafromBin->bigCoreDataNum = 136;
    tilingDatafromBin->tileDataNum = 8176;
    tilingDatafromBin->smallTailDataNum = 128;
    tilingDatafromBin->bigTailDataNum = 136;
    tilingDatafromBin->finalSmallTileNum = 1;
    tilingDatafromBin->finalBigTileNum = 1;
    tilingDatafromBin->tailBlockNum = 0;

    auto AdamApplyOneKernel = [](GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR input3, GM_ADDR input4,
                                 GM_ADDR mul0_x, GM_ADDR mul1_x, GM_ADDR mul2_x, GM_ADDR mul3_x, GM_ADDR add2_y,
                                 GM_ADDR output0, GM_ADDR output1, GM_ADDR output2, GM_ADDR workspace, GM_ADDR tiling) {
        ::adam_apply_one<0>(input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, add2_y, output0,
                            output1, output2, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(AdamApplyOneKernel, blockDim, input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x,
                add2_y, output0, output1, output2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input0);
    AscendC::GmFree(input1);
    AscendC::GmFree(input2);
    AscendC::GmFree(input3);
    AscendC::GmFree(input4);
    AscendC::GmFree(mul0_x);
    AscendC::GmFree(mul1_x);
    AscendC::GmFree(mul2_x);
    AscendC::GmFree(mul3_x);
    AscendC::GmFree(add2_y);
    AscendC::GmFree(output0);
    AscendC::GmFree(output1);
    AscendC::GmFree(output2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}