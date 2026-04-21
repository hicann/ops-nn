/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_add_rms_norm_dynamic_mx_quant_apt.cpp
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
#include "add_rms_norm_dynamic_mx_quant_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "add_rms_norm_dynamic_mx_quant_apt.cpp"
#endif

using namespace std;

#define COMPUTE_MODE_FULL_LOAD 0
#define COMPUTE_MODE_SPLIT_R 1
#define COMPUTE_MODE_REDUCE_EMPTY 2

#define Y_DATA_TYPE_FP8 0
#define Y_DATA_TYPE_FP4 1

class add_rms_norm_dynamic_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << " add_rms_norm_dynamic_mx_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << " add_rms_norm_dynamic_mx_quant_test TearDown\n" << endl;
    }
};

TEST_F(add_rms_norm_dynamic_mx_quant_test, test_case_r_full_load_fp8_case1)
{
    int64_t numA = 1;
    int64_t numR = 64;
    size_t xByteSize = numA * numR * sizeof(DTYPE_X1);
    size_t gammaByteSize = numR * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numA * numR * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numA * 2 * sizeof(DTYPE_Y);
    size_t rstdByteSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(AddRmsNormDynamicMxQuantTilingData);
    uint32_t blockDim = 1;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AddRmsNormDynamicMxQuantTilingData* tilingDatafromBin =
        reinterpret_cast<AddRmsNormDynamicMxQuantTilingData*>(tiling);

    tilingDatafromBin->numRow = 1;
    tilingDatafromBin->numCol = 64;
    tilingDatafromBin->numColAlign = 64;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->rowFactor = 1;
    tilingDatafromBin->binAddQuotient = 32;
    tilingDatafromBin->epsilon = 0.0001;
    tilingDatafromBin->avgFactor = 1.0 / numR;
    tilingDatafromBin->roundMode = 0;
    tilingDatafromBin->mxBlockSize = 32;
    tilingDatafromBin->scaleAlg = 0;
    tilingDatafromBin->blockNumInColAxis = 2;
    tilingDatafromBin->dstStrideUbBlocks = 0;
    tilingDatafromBin->mxScaleSize = 2;
    tilingDatafromBin->betaFlag = 1;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto add_rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR x,
                                                GM_ADDR mxscale, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::add_rms_norm_dynamic_mx_quant<COMPUTE_MODE_FULL_LOAD, Y_DATA_TYPE_FP8>(
            x1, x2, gamma, beta, y, x, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        add_rms_norm_dynamic_mx_quant_wrapper, blockDim, x1, x2, gamma, beta, y, x, mxscale, rstd, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(add_rms_norm_dynamic_mx_quant_test, test_case_m_not0_n_0_outputRstd_true)
{
    int64_t numA = 176;
    int64_t numR = 0;
    int64_t nAxblockNumAlignedTwo = 0;
    size_t xByteSize = numA * numR * sizeof(DTYPE_X1);
    size_t gammaByteSize = numR * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numA * numR * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numA * nAxblockNumAlignedTwo * sizeof(DTYPE_Y);
    size_t rstdByteSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(AddRmsNormDynamicMxQuantReduceEmptyTilingData);
    uint32_t blockDim = 1;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AddRmsNormDynamicMxQuantReduceEmptyTilingData* tilingDatafromBin =
        reinterpret_cast<AddRmsNormDynamicMxQuantReduceEmptyTilingData*>(tiling);

    tilingDatafromBin->perCoreElements = 176;
    tilingDatafromBin->lastCoreElements = 176;
    tilingDatafromBin->perCoreLoops = 1;
    tilingDatafromBin->perCorePerLoopElements = 176;
    tilingDatafromBin->perCoreLastLoopElements = 176;
    tilingDatafromBin->lastCoreLoops = 1;
    tilingDatafromBin->lastCorePerLoopElements = 176;
    tilingDatafromBin->lastCoreLastLoopElements = 176;
    tilingDatafromBin->rstdFlag = 1;
    tilingDatafromBin->numRow = 176;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto add_rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR x,
                                                GM_ADDR mxscale, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::add_rms_norm_dynamic_mx_quant<COMPUTE_MODE_REDUCE_EMPTY, Y_DATA_TYPE_FP8>(
            x1, x2, gamma, beta, y, x, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        add_rms_norm_dynamic_mx_quant_wrapper, blockDim, x1, x2, gamma, beta, y, x, mxscale, rstd, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// empty_tensor, tilingKey(3000), numM=0, numN=100, putput_rstd=true
TEST_F(add_rms_norm_dynamic_mx_quant_test, test_case_m_0_n_not_0_outputRstd_true)
{
    int64_t numA = 0;
    int64_t numR = 100;
    int64_t nAxblockNumAlignedTwo = 4;
    size_t xByteSize = numA * numR * sizeof(DTYPE_X1);
    size_t gammaByteSize = numR * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numA * numR * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numA * nAxblockNumAlignedTwo * sizeof(DTYPE_Y);
    size_t rstdByteSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(AddRmsNormDynamicMxQuantReduceEmptyTilingData);
    uint32_t blockDim = 1;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AddRmsNormDynamicMxQuantReduceEmptyTilingData* tilingDatafromBin =
        reinterpret_cast<AddRmsNormDynamicMxQuantReduceEmptyTilingData*>(tiling);

    tilingDatafromBin->perCoreElements = 0;
    tilingDatafromBin->lastCoreElements = 0;
    tilingDatafromBin->perCoreLoops = 0;
    tilingDatafromBin->perCorePerLoopElements = 0;
    tilingDatafromBin->perCoreLastLoopElements = 0;
    tilingDatafromBin->lastCoreLoops = 0;
    tilingDatafromBin->lastCorePerLoopElements = 0;
    tilingDatafromBin->lastCoreLastLoopElements = 0;
    tilingDatafromBin->rstdFlag = 1;
    tilingDatafromBin->numRow = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto add_rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR x,
                                                GM_ADDR mxscale, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::add_rms_norm_dynamic_mx_quant<COMPUTE_MODE_REDUCE_EMPTY, Y_DATA_TYPE_FP8>(
            x1, x2, gamma, beta, y, x, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        add_rms_norm_dynamic_mx_quant_wrapper, blockDim, x1, x2, gamma, beta, y, x, mxscale, rstd, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// empty_tensor, tilingKey(3000), numM=0, numN=0, putput_rstd=true
TEST_F(add_rms_norm_dynamic_mx_quant_test, test_case_m_0_n_0_outputRstd_true)
{
    int64_t numA = 0;
    int64_t numR = 0;
    int64_t nAxblockNumAlignedTwo = 0;
    size_t xByteSize = numA * numR * sizeof(DTYPE_X1);
    size_t gammaByteSize = numR * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numA * numR * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numA * nAxblockNumAlignedTwo * sizeof(DTYPE_Y);
    size_t rstdByteSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(AddRmsNormDynamicMxQuantReduceEmptyTilingData);
    uint32_t blockDim = 1;

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AddRmsNormDynamicMxQuantReduceEmptyTilingData* tilingDatafromBin =
        reinterpret_cast<AddRmsNormDynamicMxQuantReduceEmptyTilingData*>(tiling);

    tilingDatafromBin->perCoreElements = 0;
    tilingDatafromBin->lastCoreElements = 0;
    tilingDatafromBin->perCoreLoops = 0;
    tilingDatafromBin->perCorePerLoopElements = 0;
    tilingDatafromBin->perCoreLastLoopElements = 0;
    tilingDatafromBin->lastCoreLoops = 0;
    tilingDatafromBin->lastCorePerLoopElements = 0;
    tilingDatafromBin->lastCoreLastLoopElements = 0;
    tilingDatafromBin->rstdFlag = 1;
    tilingDatafromBin->numRow = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto add_rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR x,
                                                GM_ADDR mxscale, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::add_rms_norm_dynamic_mx_quant<COMPUTE_MODE_REDUCE_EMPTY, Y_DATA_TYPE_FP8>(
            x1, x2, gamma, beta, y, x, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        add_rms_norm_dynamic_mx_quant_wrapper, blockDim, x1, x2, gamma, beta, y, x, mxscale, rstd, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}