/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_rms_norm_quant.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "rms_norm_quant_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void pre_rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR bias, GM_ADDR res_in, GM_ADDR y, GM_ADDR res_out, GM_ADDR workspace,
    GM_ADDR tiling);
extern "C" __global__ __aicore__ void rms_norm_quant(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling);
extern "C" __global__ __aicore__ void pre_rms_norm_quant(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR res_in, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR res_out,
    GM_ADDR workspace, GM_ADDR tiling);

class rms_norm_atb_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rms_norm_atb_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rms_norm_atb_test TearDown\n" << endl;
    }
};

// rms_norm_quant
TEST_F(rms_norm_atb_test, test_rms_norm_quant_fp16_tilingKey_0b0100000001)
{
    // half & FastCompute
    uint32_t x_shape = 32 * 32 * 512;
    uint32_t gamma_shape = 1 * 512;
    uint32_t beta_shape = 1 * 512;
    uint32_t scale_and_offset_shape = 1;

    uint32_t result_shape = 32 * 32 * 512;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t gamma_size = gamma_shape * sizeof(uint16_t);
    size_t beta_size = beta_shape * sizeof(uint16_t);
    size_t scale_size = scale_and_offset_shape * sizeof(uint16_t);
    size_t offset_size = scale_and_offset_shape * sizeof(int8_t);

    size_t result_size = result_shape * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(NormCommonTilingData1);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(beta_size);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scale_size);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offset_size);

    uint8_t* result = (uint8_t*)AscendC::GmAlloc(result_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4); // 待定，如何计算
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    NormCommonTilingData1* tilingData = reinterpret_cast<NormCommonTilingData1*>(tiling);

    tilingData->numCore = 47;
    tilingData->numCol = 512;
    tilingData->numRow = 1024;
    tilingData->avgFactor = static_cast<float>(1.0 / 512);
    tilingData->epsilon = 0.01;

    tilingData->sliceSize = 8192;
    tilingData->sliceNum = 1; // MaxSliceSize = 8192
    tilingData->tailSliceSize = 0;

    tilingData->quantMin = -128;
    tilingData->scale = 1.0;
    tilingData->offset = 0;
    tilingData->highPrecisionMode = 0;
    tilingData->gemmaMode = 0;

    ICPU_SET_TILING_KEY(0b0100000001);
    ICPU_RUN_KF(rms_norm_quant, blockDim, x, gamma, beta, scale, offset, result, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(result);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// rms_norm_quant
TEST_F(rms_norm_atb_test, test_rms_norm_quant_fp16_tilingKey_0b0001000001)
{
    // half & FastCompute
    uint32_t x_shape = 32 * 32 * 16384;
    uint32_t gamma_shape = 1 * 16384;
    uint32_t scale_and_offset_shape = 1;

    uint32_t result_shape = 32 * 32 * 16384;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t gamma_size = gamma_shape * sizeof(uint16_t);
    size_t scale_size = scale_and_offset_shape * sizeof(uint16_t);
    size_t offset_size = scale_and_offset_shape * sizeof(int8_t);

    size_t result_size = result_shape * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(NormCommonTilingData1);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gamma_size);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scale_size);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offset_size);

    uint8_t* result = (uint8_t*)AscendC::GmAlloc(result_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4); // 待定，如何计算
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    NormCommonTilingData1* tilingData = reinterpret_cast<NormCommonTilingData1*>(tiling);

    tilingData->numCore = 47;
    tilingData->numCol = 16384;
    tilingData->numRow = 1024;
    tilingData->avgFactor = static_cast<float>(1.0 / 16384);
    tilingData->epsilon = 0.01;

    tilingData->sliceSize = 8192;
    tilingData->sliceNum = 2; // MaxSliceSize = 8192
    tilingData->tailSliceSize = 0;

    tilingData->quantMin = -128;
    tilingData->scale = 1.0;
    tilingData->offset = 0;
    tilingData->highPrecisionMode = 0;
    tilingData->gemmaMode = 0;

    ICPU_SET_TILING_KEY(0b0001000001);
    ICPU_RUN_KF(rms_norm_quant, blockDim, x, gamma, nullptr, scale, offset, result, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(result);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
