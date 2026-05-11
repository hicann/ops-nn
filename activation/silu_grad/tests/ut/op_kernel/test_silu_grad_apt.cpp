/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void silu_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling);

#define SILU_GRAD_BF16_NDDMA_WITHOUT_LOOPS_TILING_KEY 200000001000100
#define SILU_GRAD_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY 300000001000100

class silu_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "silu_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "silu_grad_test TearDown\n" << endl;
    }
};

TEST_F(silu_grad_test, test_case_fp32_1)
{
    size_t inputByteSize = 256 * sizeof(float);
    size_t xByteSize = 256 * sizeof(float);
    size_t outputByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(SiluGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    SiluGradTilingData* tilingDatafromBin = reinterpret_cast<SiluGradTilingData*>(tiling);

    tilingDatafromBin->blockFormer = 256;
    tilingDatafromBin->ubFormer = 256;
    tilingDatafromBin->ubOuter = 1;
    tilingDatafromBin->ubTail = 256;
    tilingDatafromBin->blockTail = 256;
    tilingDatafromBin->shapeLen = 1;
    tilingDatafromBin->ubSplitAxis = 0;
    tilingDatafromBin->dimProductBeforeUbInner = 1;
    tilingDatafromBin->elemNum = 256;
    tilingDatafromBin->input0Dims[0] = 256;
    tilingDatafromBin->input1Dims[0] = 256;
    tilingDatafromBin->outputDims[0] = 256;
    tilingDatafromBin->input0Strides[0] = 1;
    tilingDatafromBin->input1Strides[0] = 1;
    tilingDatafromBin->outputStrides[0] = 1;

    ICPU_SET_TILING_KEY(SILU_GRAD_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(silu_grad, blockDim, dy, x, dx, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
