/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_deformable_conv2d.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "deformable_conv2d_tiling_def.h"

extern "C" __global__ __aicore__ void deformable_conv2d(GM_ADDR x, GM_ADDR weight, GM_ADDR offset, GM_ADDR bias,
                                                        GM_ADDR out, GM_ADDR deform_out, GM_ADDR workspace,
                                                        GM_ADDR tiling);

class deformable_conv2d_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "deformable_conv2d_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "deformable_conv2d_test TearDown\n" << std::endl;
    }
};

TEST_F(deformable_conv2d_test, test_case_bfloat16_1)
{
    // system(
    //     "cp -rf "
    //     "../../../../conv/deformable_conv2d/tests/ut/op_kernel/deformable_conv2d_data ./");
    // system("chmod -R 755 ./deformable_conv2d_data/");
    // system("cd ./deformable_conv2d_data/ && python3 gen_data.py '(1, 4, 4, 1)' '(1, 1, 3, 3)' '(1, 4, 4, 27)' '(1)' 'bfloat16'");

    size_t xByteSize = 4 * 4 * sizeof(float_t);
    size_t weightByteSize = 3 * 3 * sizeof(float_t);
    size_t offsetByteSize = 4 * 4 * 27 * sizeof(float_t);
    size_t biasByteSize = 1 * sizeof(float_t);
    size_t outputByteSize = 4 * 4 * sizeof(float_t);
    size_t yByteSize = 12 * 12 * sizeof(float_t);
    size_t tiling_data_size = sizeof(DeformableConv2dTilingData);
    size_t workspaceSize = 16 * 1024 * 1024;
    uint32_t blockDim = 4;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    // std::string fileName = "./deformable_conv2d_data/bfloat16_x_deformable_conv2d.bin";
    // ReadFile(fileName, xByteSize, x, xByteSize);
    // fileName = "./deformable_conv2d_data/bfloat16_weight_deformable_conv2d.bin";
    // ReadFile(fileName, weightByteSize, weight, weightByteSize);
    // fileName = "./deformable_conv2d_data/bfloat16_offset_deformable_conv2d.bin";
    // ReadFile(fileName, offsetByteSize, offset, offsetByteSize);
    // fileName = "./deformable_conv2d_data/bfloat16_bias_deformable_conv2d.bin";
    // ReadFile(fileName, biasByteSize, bias, biasByteSize);

    DeformableConv2dTilingData* tilingDatafromBin = reinterpret_cast<DeformableConv2dTilingData*>(tiling);

    tilingDatafromBin->n = 1;
    tilingDatafromBin->inC = 1;
    tilingDatafromBin->inH = 4;
    tilingDatafromBin->inW = 4;
    tilingDatafromBin->outC = 1;
    tilingDatafromBin->outH = 4;
    tilingDatafromBin->outW = 4;
    tilingDatafromBin->kH = 3;
    tilingDatafromBin->kW = 3;
    tilingDatafromBin->padTop = 1;
    tilingDatafromBin->padLeft = 1;
    tilingDatafromBin->strideH = 1;
    tilingDatafromBin->strideW = 1;
    tilingDatafromBin->dilationH = 1;
    tilingDatafromBin->dilationW = 1;
    tilingDatafromBin->deformableGroups = 1;
    tilingDatafromBin->groups = 1;
    tilingDatafromBin->hasBias = true;
    
    tilingDatafromBin->slideSizeW = 128;
    tilingDatafromBin->groupLen = 1;
    tilingDatafromBin->singleVecNum = 0;
    tilingDatafromBin->tailVecNum = 4;

    tilingDatafromBin->mmTilingData.usedCoreNum = 1;
    tilingDatafromBin->mmTilingData.M = 1;
    tilingDatafromBin->mmTilingData.N = 4;
    tilingDatafromBin->mmTilingData.Ka = 9;
    tilingDatafromBin->mmTilingData.Kb = 9;
    tilingDatafromBin->mmTilingData.singleCoreM = 1;
    tilingDatafromBin->mmTilingData.singleCoreN = 4;
    tilingDatafromBin->mmTilingData.singleCoreK = 9;
    tilingDatafromBin->mmTilingData.baseM = 16;
    tilingDatafromBin->mmTilingData.baseN = 16;
    tilingDatafromBin->mmTilingData.baseK = 16;
    tilingDatafromBin->mmTilingData.depthA1 = 1;
    tilingDatafromBin->mmTilingData.depthB1 = 1;
    tilingDatafromBin->mmTilingData.stepM = 1;
    tilingDatafromBin->mmTilingData.stepN = 1;
    tilingDatafromBin->mmTilingData.stepKa = 1;
    tilingDatafromBin->mmTilingData.stepKb = 1;
    tilingDatafromBin->mmTilingData.transLength = 0;
    tilingDatafromBin->mmTilingData.iterateOrder = 0;
    tilingDatafromBin->mmTilingData.shareMode = 0;
    tilingDatafromBin->mmTilingData.shareL1Size = 2048;
    tilingDatafromBin->mmTilingData.shareL0CSize = 1024;
    tilingDatafromBin->mmTilingData.shareUbSize = 0;
    tilingDatafromBin->mmTilingData.batchM = 1;
    tilingDatafromBin->mmTilingData.batchN = 1;
    tilingDatafromBin->mmTilingData.singleBatchM = 1;
    tilingDatafromBin->mmTilingData.singleBatchN = 1;

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(deformable_conv2d, blockDim, x, weight, offset, bias, output, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(weight));
    AscendC::GmFree((void*)(offset));
    AscendC::GmFree((void*)(bias));
    AscendC::GmFree((void*)(output));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(deformable_conv2d_test, test_case_bfloat16_2)
{
    // system(
    //     "cp -rf "
    //     "../../../../conv/deformable_conv2d/tests/ut/op_kernel/deformable_conv2d_data ./");
    // system("chmod -R 755 ./deformable_conv2d_data/");
    // system("cd ./deformable_conv2d_data/ && python3 gen_data.py '(1, 4, 4, 16)' '(1, 16, 3, 3)' '(1, 4, 4, 27)' '(1)' 'bfloat16'");

    size_t xByteSize = 4 * 4 * 16 * sizeof(float_t);
    size_t weightByteSize = 16 * 3 * 3 * sizeof(float_t);
    size_t offsetByteSize = 4 * 4 * 27 * sizeof(float_t);
    size_t biasByteSize = 1 * sizeof(float_t);
    size_t outputByteSize = 4 * 4 * sizeof(float_t);
    size_t yByteSize = 16 * 12 * 12 * sizeof(float_t);
    size_t tiling_data_size = sizeof(DeformableConv2dTilingData);
    size_t workspaceSize = 16 * 1024 * 1024;
    uint32_t blockDim = 4;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    // std::string fileName = "./deformable_conv2d_data/bfloat16_x_deformable_conv2d.bin";
    // ReadFile(fileName, xByteSize, x, xByteSize);
    // fileName = "./deformable_conv2d_data/bfloat16_weight_deformable_conv2d.bin";
    // ReadFile(fileName, weightByteSize, weight, weightByteSize);
    // fileName = "./deformable_conv2d_data/bfloat16_offset_deformable_conv2d.bin";
    // ReadFile(fileName, offsetByteSize, offset, offsetByteSize);
    // fileName = "./deformable_conv2d_data/bfloat16_bias_deformable_conv2d.bin";
    // ReadFile(fileName, biasByteSize, bias, biasByteSize);

    DeformableConv2dTilingData* tilingDatafromBin = reinterpret_cast<DeformableConv2dTilingData*>(tiling);

    tilingDatafromBin->n = 1;
    tilingDatafromBin->inC = 16;
    tilingDatafromBin->inH = 4;
    tilingDatafromBin->inW = 4;
    tilingDatafromBin->outC = 1;
    tilingDatafromBin->outH = 4;
    tilingDatafromBin->outW = 4;
    tilingDatafromBin->kH = 3;
    tilingDatafromBin->kW = 3;
    tilingDatafromBin->padTop = 1;
    tilingDatafromBin->padLeft = 1;
    tilingDatafromBin->strideH = 1;
    tilingDatafromBin->strideW = 1;
    tilingDatafromBin->dilationH = 1;
    tilingDatafromBin->dilationW = 1;
    tilingDatafromBin->deformableGroups = 1;
    tilingDatafromBin->groups = 1;
    tilingDatafromBin->hasBias = true;
    
    tilingDatafromBin->slideSizeW = 128;
    tilingDatafromBin->groupLen = 1;
    tilingDatafromBin->singleVecNum = 0;
    tilingDatafromBin->tailVecNum = 4;

    tilingDatafromBin->mmTilingData.usedCoreNum = 1;
    tilingDatafromBin->mmTilingData.M = 1;
    tilingDatafromBin->mmTilingData.N = 4;
    tilingDatafromBin->mmTilingData.Ka = 144;
    tilingDatafromBin->mmTilingData.Kb = 144;
    tilingDatafromBin->mmTilingData.singleCoreM = 1;
    tilingDatafromBin->mmTilingData.singleCoreN = 4;
    tilingDatafromBin->mmTilingData.singleCoreK = 144;
    tilingDatafromBin->mmTilingData.baseM = 16;
    tilingDatafromBin->mmTilingData.baseN = 16;
    tilingDatafromBin->mmTilingData.baseK = 144;
    tilingDatafromBin->mmTilingData.depthA1 = 1;
    tilingDatafromBin->mmTilingData.depthB1 = 1;
    tilingDatafromBin->mmTilingData.stepM = 1;
    tilingDatafromBin->mmTilingData.stepN = 1;
    tilingDatafromBin->mmTilingData.stepKa = 1;
    tilingDatafromBin->mmTilingData.stepKb = 1;
    tilingDatafromBin->mmTilingData.transLength = 0;
    tilingDatafromBin->mmTilingData.iterateOrder = 0;
    tilingDatafromBin->mmTilingData.shareMode = 0;
    tilingDatafromBin->mmTilingData.shareL1Size = 2048;
    tilingDatafromBin->mmTilingData.shareL0CSize = 1024;
    tilingDatafromBin->mmTilingData.shareUbSize = 0;
    tilingDatafromBin->mmTilingData.batchM = 1;
    tilingDatafromBin->mmTilingData.batchN = 1;
    tilingDatafromBin->mmTilingData.singleBatchM = 1;
    tilingDatafromBin->mmTilingData.singleBatchN = 1;

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(deformable_conv2d, blockDim, x, weight, offset, bias, output, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(weight));
    AscendC::GmFree((void*)(offset));
    AscendC::GmFree((void*)(bias));
    AscendC::GmFree((void*)(output));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}