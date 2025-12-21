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
 * \file test_ada_layer_norm_v2.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "ada_layer_norm_v2_tiling_def.h"
#include "data_utils.h"

extern "C" __global__ __aicore__ void ada_layer_norm_v2(
    GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR weight, GM_ADDR bias, GM_ADDR out, GM_ADDR mean, 
    GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling);

class ada_layer_norm_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ada_layer_norm_v2_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ada_layer_norm_v2_test TearDown\n" << std::endl;
    }
};

TEST_F(ada_layer_norm_v2_test, test_case_bfloat16_1)
{
    size_t xByteSize = 4 * 16 * 128 * sizeof(bfloat16_t);
    size_t scaleByteSize = 4 * 128 * sizeof(bfloat16_t);
    size_t weightByteSize = 128 * sizeof(bfloat16_t);
    size_t meanByteSize = 4 * 16 * 1 * sizeof(bfloat16_t);
    uint32_t blockDim = 20;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* shift = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AdaLayerNormTilingData));

    AdaLayerNormTilingData* tilingDatafromBin = reinterpret_cast<AdaLayerNormTilingData*>(tiling);

    tilingDatafromBin->batchSize = 4;
    tilingDatafromBin->seqLen = 16;
    tilingDatafromBin->hiddenDim = 128;
    tilingDatafromBin->epsilon = 0.00001f;
    tilingDatafromBin->aveFactor = static_cast<float>(1.0/128.0);
    tilingDatafromBin->hasWeight = 1;
    tilingDatafromBin->hasBias = 1;
    tilingDatafromBin->hasSmooth = 0;
    tilingDatafromBin->singleCoreNum = 1;
    tilingDatafromBin->tailNum = 24;
    tilingDatafromBin->sliceSize = 128;
    tilingDatafromBin->rowNum = 32;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(ada_layer_norm_v2, blockDim, x, scale, shift, weight, bias, out, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)shift);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)out);
    AscendC::GmFree((void*)mean);
    AscendC::GmFree((void*)rstd);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(ada_layer_norm_v2_test, test_case_bfloat16_2)
{
    size_t xByteSize = 1 * 1 * 10000 * sizeof(bfloat16_t);
    size_t scaleByteSize = 1 * 10000 * sizeof(bfloat16_t);
    size_t weightByteSize = 10000 * sizeof(bfloat16_t);
    size_t meanByteSize = 1 * 1 * 1 * sizeof(bfloat16_t);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* shift = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AdaLayerNormTilingData));

    AdaLayerNormTilingData* tilingDatafromBin = reinterpret_cast<AdaLayerNormTilingData*>(tiling);

    tilingDatafromBin->batchSize = 1;
    tilingDatafromBin->seqLen = 1;
    tilingDatafromBin->hiddenDim = 10000;
    tilingDatafromBin->epsilon = 0.00001f;
    tilingDatafromBin->aveFactor = static_cast<float>(1.0/10000.0);
    tilingDatafromBin->hasWeight = 1;
    tilingDatafromBin->hasBias = 1;
    tilingDatafromBin->hasSmooth = 0;
    tilingDatafromBin->singleCoreNum = 0;
    tilingDatafromBin->tailNum = 1;
    tilingDatafromBin->sliceSize = 3334;
    tilingDatafromBin->rowNum = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(ada_layer_norm_v2, blockDim, x, scale, shift, weight, bias, out, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)shift);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)out);
    AscendC::GmFree((void*)mean);
    AscendC::GmFree((void*)rstd);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}