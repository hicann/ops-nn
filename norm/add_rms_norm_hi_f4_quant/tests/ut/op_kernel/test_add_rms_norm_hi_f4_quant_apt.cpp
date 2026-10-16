/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_add_rms_norm_hi_f4_quant_apt.cpp
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

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void add_rms_norm_hi_f4_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
                                                               GM_ADDR y, GM_ADDR x, GM_ADDR scale, GM_ADDR rstd,
                                                               GM_ADDR workspace, GM_ADDR tiling);

class add_rms_norm_hi_f4_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << " add_rms_norm_hi_f4_quant_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << " add_rms_norm_hi_f4_quant_test TearDown\n" << endl; }
};

TEST_F(add_rms_norm_hi_f4_quant_test, test_case_r_full_load_hif4_tilingkey101_case1)
{
    int64_t numA = 1;
    int64_t numR = 64;
    size_t xByteSize = numA * numR * sizeof(DTYPE_X1);
    size_t gammaByteSize = numR * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numA * numR / 2;
    size_t scaleByteSize = numA * 1 * sizeof(float);
    size_t rstdByteSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(AddRmsNormHiF4QuantTilingData);
    uint32_t blockDim = 1;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AddRmsNormHiF4QuantTilingData* tilingDatafromBin = reinterpret_cast<AddRmsNormHiF4QuantTilingData*>(tiling);

    tilingDatafromBin->numRow = 1;
    tilingDatafromBin->numCol = 64;
    tilingDatafromBin->numColAlign = 64;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->rowFactor = 1;
    tilingDatafromBin->binAddQuotient = 32;
    tilingDatafromBin->epsilon = 0.0001;
    tilingDatafromBin->avgFactor = 1.0 / numR;
    tilingDatafromBin->mxBlockSize = 64;
    tilingDatafromBin->blockNumInColAxis = 1;
    tilingDatafromBin->mxScaleSize = 1;
    tilingDatafromBin->betaFlag = 1;
    tilingDatafromBin->rstdFlag = 1;
    tilingDatafromBin->scaleAlg = 0;
    tilingDatafromBin->innerLoopEle = 1;
    tilingDatafromBin->max8LoopEle = 1;
    tilingDatafromBin->scaleLoopEle = 1;
    tilingDatafromBin->outLoopEle = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(add_rms_norm_hi_f4_quant, blockDim, x1, x2, gamma, beta, y, x, scale, rstd, workspace,
                (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(add_rms_norm_hi_f4_quant_test, test_case_r_full_load_hif4_tilingkey101_case2)
{
    int64_t numA = 8;
    int64_t numR = 64;
    size_t xByteSize = numA * numR * sizeof(DTYPE_X1);
    size_t gammaByteSize = numR * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numA * numR / 2;
    size_t scaleByteSize = numA * 1 * sizeof(float);
    size_t rstdByteSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(AddRmsNormHiF4QuantTilingData);
    uint32_t blockDim = 8;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AddRmsNormHiF4QuantTilingData* tilingDatafromBin = reinterpret_cast<AddRmsNormHiF4QuantTilingData*>(tiling);

    tilingDatafromBin->numRow = 8;
    tilingDatafromBin->numCol = 64;
    tilingDatafromBin->numColAlign = 64;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->rowFactor = 1;
    tilingDatafromBin->binAddQuotient = 32;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->avgFactor = 1.0 / numR;
    tilingDatafromBin->mxBlockSize = 64;
    tilingDatafromBin->blockNumInColAxis = 1;
    tilingDatafromBin->mxScaleSize = 1;
    tilingDatafromBin->betaFlag = 0;
    tilingDatafromBin->rstdFlag = 0;
    tilingDatafromBin->innerLoopEle = 1;
    tilingDatafromBin->max8LoopEle = 1;
    tilingDatafromBin->scaleLoopEle = 1;
    tilingDatafromBin->outLoopEle = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(add_rms_norm_hi_f4_quant, blockDim, x1, x2, gamma, beta, y, x, scale, rstd, workspace,
                (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(add_rms_norm_hi_f4_quant_test, test_case_r_full_load_hif4_tilingkey101_case3)
{
    int64_t numA = 1;
    int64_t numR = 128;
    size_t xByteSize = numA * numR * sizeof(DTYPE_X1);
    size_t gammaByteSize = numR * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numA * numR / 2;
    size_t scaleByteSize = numA * 2 * sizeof(float);
    size_t rstdByteSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(AddRmsNormHiF4QuantTilingData);
    uint32_t blockDim = 1;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AddRmsNormHiF4QuantTilingData* tilingDatafromBin = reinterpret_cast<AddRmsNormHiF4QuantTilingData*>(tiling);

    tilingDatafromBin->numRow = 1;
    tilingDatafromBin->numCol = 128;
    tilingDatafromBin->numColAlign = 128;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->rowFactor = 1;
    tilingDatafromBin->binAddQuotient = 64;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->avgFactor = 1.0 / numR;
    tilingDatafromBin->mxBlockSize = 64;
    tilingDatafromBin->blockNumInColAxis = 2;
    tilingDatafromBin->mxScaleSize = 2;
    tilingDatafromBin->betaFlag = 1;
    tilingDatafromBin->rstdFlag = 1;
    tilingDatafromBin->scaleAlg = 0;
    tilingDatafromBin->innerLoopEle = 2;
    tilingDatafromBin->max8LoopEle = 2;
    tilingDatafromBin->scaleLoopEle = 1;
    tilingDatafromBin->outLoopEle = 4;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(add_rms_norm_hi_f4_quant, blockDim, x1, x2, gamma, beta, y, x, scale, rstd, workspace,
                (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
