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
 * \file test_quantized_batch_norm.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "quantized_batch_norm_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void quantized_batch_norm(
    GM_ADDR x, GM_ADDR mean, GM_ADDR variance, GM_ADDR input_scale, GM_ADDR input_zero_point, GM_ADDR output_scale,
    GM_ADDR output_zero_point, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class quantized_batch_norm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << " quantized_batch_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << " quantized_batch_norm_test TearDown\n" << endl;
    }
};

TEST_F(quantized_batch_norm_test, test_case_1000)
{
    size_t xByteSize = 1 * 1 * 1 * 18 * sizeof(int32_t);
    size_t weightByteSize = 1 * sizeof(float);
    size_t scaleByteSize = 1 * sizeof(float);
    size_t zeroPointByteSize = 1 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(QuantizedBatchNormWelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* input_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* input_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* output_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* output_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizedBatchNormWelfordTilingData* tilingDatafromBin =
        reinterpret_cast<QuantizedBatchNormWelfordTilingData*>(tiling);

    tilingDatafromBin->patternR1 = 1;
    tilingDatafromBin->patternR0 = 18;
    tilingDatafromBin->patternA = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->aUbFactor = 8;
    tilingDatafromBin->aUbLoop = 1;
    tilingDatafromBin->aUbTail = 1;
    tilingDatafromBin->tailCoreAUbLoop = 1;
    tilingDatafromBin->tailCoreAUbTail = 1;
    tilingDatafromBin->r0UbFactor = 256;
    tilingDatafromBin->r0UbLoop = 2;
    tilingDatafromBin->r0UbTail = 2;
    tilingDatafromBin->procNR0 = 1;
    tilingDatafromBin->nR0Loop = 1;
    tilingDatafromBin->lastLoopNR0 = 1;
    tilingDatafromBin->patternR0Align = 24;
    tilingDatafromBin->epsilon = 1e-5;

    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(
        quantized_batch_norm, blockDim, x, mean, variance, input_scale, input_zero_point, output_scale,
        output_zero_point, weight, bias, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(input_scale);
    AscendC::GmFree(input_zero_point);
    AscendC::GmFree(output_scale);
    AscendC::GmFree(output_zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantized_batch_norm_test, test_case_1001)
{
    size_t xByteSize = 1 * 1 * 1 * 32 * sizeof(int32_t);
    size_t weightByteSize = 1 * sizeof(float);
    size_t scaleByteSize = 1 * sizeof(float);
    size_t zeroPointByteSize = 1 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(QuantizedBatchNormWelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* input_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* input_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* output_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* output_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizedBatchNormWelfordTilingData* tilingDatafromBin =
        reinterpret_cast<QuantizedBatchNormWelfordTilingData*>(tiling);

    tilingDatafromBin->patternR1 = 1;
    tilingDatafromBin->patternR0 = 32;
    tilingDatafromBin->patternA = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->aUbFactor = 16;
    tilingDatafromBin->aUbLoop = 1;
    tilingDatafromBin->aUbTail = 1;
    tilingDatafromBin->tailCoreAUbLoop = 1;
    tilingDatafromBin->tailCoreAUbTail = 1;
    tilingDatafromBin->r0UbFactor = 256;
    tilingDatafromBin->r0UbLoop = 2;
    tilingDatafromBin->r0UbTail = 16;
    tilingDatafromBin->procNR0 = 1;
    tilingDatafromBin->nR0Loop = 1;
    tilingDatafromBin->lastLoopNR0 = 1;
    tilingDatafromBin->patternR0Align = 32;
    tilingDatafromBin->epsilon = 1e-5;

    ICPU_SET_TILING_KEY(1001);
    ICPU_RUN_KF(
        quantized_batch_norm, blockDim, x, mean, variance, input_scale, input_zero_point, output_scale,
        output_zero_point, weight, bias, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(input_scale);
    AscendC::GmFree(input_zero_point);
    AscendC::GmFree(output_scale);
    AscendC::GmFree(output_zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantized_batch_norm_test, test_case_1002)
{
    size_t xByteSize = 5 * 1 * 1 * 5 * sizeof(int32_t);
    size_t weightByteSize = 1 * sizeof(float);
    size_t scaleByteSize = 1 * sizeof(float);
    size_t zeroPointByteSize = 1 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(QuantizedBatchNormWelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* input_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* input_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* output_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* output_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizedBatchNormWelfordTilingData* tilingDatafromBin =
        reinterpret_cast<QuantizedBatchNormWelfordTilingData*>(tiling);

    tilingDatafromBin->patternR1 = 5;
    tilingDatafromBin->patternR0 = 5;
    tilingDatafromBin->patternA = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->aUbFactor = 16;
    tilingDatafromBin->aUbLoop = 1;
    tilingDatafromBin->aUbTail = 1;
    tilingDatafromBin->tailCoreAUbLoop = 1;
    tilingDatafromBin->tailCoreAUbTail = 1;
    tilingDatafromBin->r0UbFactor = 256;
    tilingDatafromBin->r0UbLoop = 1;
    tilingDatafromBin->r0UbTail = 5;
    tilingDatafromBin->procNR0 = 4;
    tilingDatafromBin->nR0Loop = 2;
    tilingDatafromBin->lastLoopNR0 = 1;
    tilingDatafromBin->patternR0Align = 8;
    tilingDatafromBin->epsilon = 1e-5;

    ICPU_SET_TILING_KEY(1002);
    ICPU_RUN_KF(
        quantized_batch_norm, blockDim, x, mean, variance, input_scale, input_zero_point, output_scale,
        output_zero_point, weight, bias, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(input_scale);
    AscendC::GmFree(input_zero_point);
    AscendC::GmFree(output_scale);
    AscendC::GmFree(output_zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantized_batch_norm_test, test_case_1012)
{
    size_t xByteSize = 5 * 1 * 1 * 8 * sizeof(int32_t);
    size_t weightByteSize = 1 * sizeof(float);
    size_t scaleByteSize = 1 * sizeof(float);
    size_t zeroPointByteSize = 1 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(QuantizedBatchNormWelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* input_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* input_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* output_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* output_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizedBatchNormWelfordTilingData* tilingDatafromBin =
        reinterpret_cast<QuantizedBatchNormWelfordTilingData*>(tiling);

    tilingDatafromBin->patternR1 = 5;
    tilingDatafromBin->patternR0 = 8;
    tilingDatafromBin->patternA = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->aUbFactor = 16;
    tilingDatafromBin->aUbLoop = 1;
    tilingDatafromBin->aUbTail = 1;
    tilingDatafromBin->tailCoreAUbLoop = 1;
    tilingDatafromBin->tailCoreAUbTail = 1;
    tilingDatafromBin->r0UbFactor = 256;
    tilingDatafromBin->r0UbLoop = 1;
    tilingDatafromBin->r0UbTail = 5;
    tilingDatafromBin->procNR0 = 4;
    tilingDatafromBin->nR0Loop = 2;
    tilingDatafromBin->lastLoopNR0 = 1;
    tilingDatafromBin->patternR0Align = 8;
    tilingDatafromBin->epsilon = 1e-5;

    ICPU_SET_TILING_KEY(1012);
    ICPU_RUN_KF(
        quantized_batch_norm, blockDim, x, mean, variance, input_scale, input_zero_point, output_scale,
        output_zero_point, weight, bias, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(input_scale);
    AscendC::GmFree(input_zero_point);
    AscendC::GmFree(output_scale);
    AscendC::GmFree(output_zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantized_batch_norm_test, test_case_1003)
{
    size_t xByteSize = 8 * 1 * 1 * 5 * sizeof(int32_t);
    size_t weightByteSize = 1 * sizeof(float);
    size_t scaleByteSize = 1 * sizeof(float);
    size_t zeroPointByteSize = 1 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(QuantizedBatchNormWelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* input_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* input_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* output_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* output_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizedBatchNormWelfordTilingData* tilingDatafromBin =
        reinterpret_cast<QuantizedBatchNormWelfordTilingData*>(tiling);

    tilingDatafromBin->patternR1 = 8;
    tilingDatafromBin->patternR0 = 5;
    tilingDatafromBin->patternA = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->aUbFactor = 8;
    tilingDatafromBin->aUbLoop = 1;
    tilingDatafromBin->aUbTail = 1;
    tilingDatafromBin->tailCoreAUbLoop = 1;
    tilingDatafromBin->tailCoreAUbTail = 1;
    tilingDatafromBin->r0UbFactor = 2560;
    tilingDatafromBin->r0UbLoop = 1;
    tilingDatafromBin->r0UbTail = 5;
    tilingDatafromBin->procNR0 = 4;
    tilingDatafromBin->nR0Loop = 2;
    tilingDatafromBin->lastLoopNR0 = 4;
    tilingDatafromBin->patternR0Align = 8;
    tilingDatafromBin->epsilon = 1e-5;

    ICPU_SET_TILING_KEY(1003);
    ICPU_RUN_KF(
        quantized_batch_norm, blockDim, x, mean, variance, input_scale, input_zero_point, output_scale,
        output_zero_point, weight, bias, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(input_scale);
    AscendC::GmFree(input_zero_point);
    AscendC::GmFree(output_scale);
    AscendC::GmFree(output_zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantized_batch_norm_test, test_case_1013)
{
    size_t xByteSize = 8 * 1 * 1 * 8 * sizeof(int32_t);
    size_t weightByteSize = 1 * sizeof(float);
    size_t scaleByteSize = 1 * sizeof(float);
    size_t zeroPointByteSize = 1 * sizeof(int32_t);

    size_t tiling_data_size = sizeof(QuantizedBatchNormWelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* input_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* input_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* output_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* output_zero_point = (uint8_t*)AscendC::GmAlloc(zeroPointByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizedBatchNormWelfordTilingData* tilingDatafromBin =
        reinterpret_cast<QuantizedBatchNormWelfordTilingData*>(tiling);

    tilingDatafromBin->patternR1 = 8;
    tilingDatafromBin->patternR0 = 8;
    tilingDatafromBin->patternA = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->aUbFactor = 16;
    tilingDatafromBin->aUbLoop = 1;
    tilingDatafromBin->aUbTail = 1;
    tilingDatafromBin->tailCoreAUbLoop = 1;
    tilingDatafromBin->tailCoreAUbTail = 1;
    tilingDatafromBin->r0UbFactor = 256;
    tilingDatafromBin->r0UbLoop = 1;
    tilingDatafromBin->r0UbTail = 5;
    tilingDatafromBin->procNR0 = 4;
    tilingDatafromBin->nR0Loop = 2;
    tilingDatafromBin->lastLoopNR0 = 4;
    tilingDatafromBin->patternR0Align = 8;
    tilingDatafromBin->epsilon = 1e-5;

    ICPU_SET_TILING_KEY(1013);
    ICPU_RUN_KF(
        quantized_batch_norm, blockDim, x, mean, variance, input_scale, input_zero_point, output_scale,
        output_zero_point, weight, bias, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(input_scale);
    AscendC::GmFree(input_zero_point);
    AscendC::GmFree(output_scale);
    AscendC::GmFree(output_zero_point);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}