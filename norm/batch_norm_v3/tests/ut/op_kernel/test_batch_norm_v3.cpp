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
 * \file test_batch_norm_v3.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "batch_norm_v3_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void batch_norm_v3(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean, GM_ADDR variance, GM_ADDR y, GM_ADDR mean_out,
    GM_ADDR variance_out, GM_ADDR save_mean, GM_ADDR save_var, GM_ADDR workspace, GM_ADDR tiling);

class batch_norm_v3_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << " batch_norm_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << " batch_norm_v3_test TearDown\n" << endl;
    }
};

TEST_F(batch_norm_v3_test, test_case_1000)
{
    size_t xByteSize = 1 * 1 * 1 * 18 * sizeof(float);
    size_t weightByteSize = 1 * sizeof(float);

    size_t tiling_data_size = sizeof(BatchNormV3WelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_var = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3WelfordTilingData* tilingDatafromBin = reinterpret_cast<BatchNormV3WelfordTilingData*>(tiling);

    tilingDatafromBin->patternR1 = 1;
    tilingDatafromBin->patternR0 = 18;
    tilingDatafromBin->patternA = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->aUbFactor = 16;
    tilingDatafromBin->aUbLoop = 1;
    tilingDatafromBin->aUbTail = 1;
    tilingDatafromBin->tailCoreAUbLoop = 1;
    tilingDatafromBin->tailCoreAUbTail = 1;
    tilingDatafromBin->r0UbFactor = 16;
    tilingDatafromBin->r0UbLoop = 2;
    tilingDatafromBin->r0UbTail = 2;
    tilingDatafromBin->procNR0 = 1;
    tilingDatafromBin->nR0Loop = 1;
    tilingDatafromBin->lastLoopNR0 = 1;
    tilingDatafromBin->patternR0Align = 24;
    tilingDatafromBin->dichotomizeAddDiffSize = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->momentumReverse = 0.9;
    tilingDatafromBin->batchVarScale = 1;

    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(save_mean);
    AscendC::GmFree(save_var);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(batch_norm_v3_test, test_case_1001)
{
    size_t xByteSize = 1 * 1 * 1 * 32 * sizeof(float);
    size_t weightByteSize = 1 * sizeof(float);

    size_t tiling_data_size = sizeof(BatchNormV3WelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_var = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3WelfordTilingData* tilingDatafromBin = reinterpret_cast<BatchNormV3WelfordTilingData*>(tiling);

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
    tilingDatafromBin->r0UbFactor = 16;
    tilingDatafromBin->r0UbLoop = 2;
    tilingDatafromBin->r0UbTail = 16;
    tilingDatafromBin->procNR0 = 1;
    tilingDatafromBin->nR0Loop = 1;
    tilingDatafromBin->lastLoopNR0 = 1;
    tilingDatafromBin->patternR0Align = 32;
    tilingDatafromBin->dichotomizeAddDiffSize = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->momentumReverse = 0.9;
    tilingDatafromBin->batchVarScale = 1;

    ICPU_SET_TILING_KEY(1001);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(save_mean);
    AscendC::GmFree(save_var);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(batch_norm_v3_test, test_case_1002)
{
    size_t xByteSize = 5 * 1 * 1 * 5 * sizeof(float);
    size_t weightByteSize = 1 * sizeof(float);

    size_t tiling_data_size = sizeof(BatchNormV3WelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_var = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3WelfordTilingData* tilingDatafromBin = reinterpret_cast<BatchNormV3WelfordTilingData*>(tiling);

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
    tilingDatafromBin->r0UbFactor = 32;
    tilingDatafromBin->r0UbLoop = 1;
    tilingDatafromBin->r0UbTail = 5;
    tilingDatafromBin->procNR0 = 4;
    tilingDatafromBin->nR0Loop = 2;
    tilingDatafromBin->lastLoopNR0 = 1;
    tilingDatafromBin->patternR0Align = 8;
    tilingDatafromBin->dichotomizeAddDiffSize = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->momentumReverse = 0.9;
    tilingDatafromBin->batchVarScale = 1;

    ICPU_SET_TILING_KEY(1002);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(save_mean);
    AscendC::GmFree(save_var);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(batch_norm_v3_test, test_case_1012)
{
    size_t xByteSize = 5 * 1 * 1 * 8 * sizeof(float);
    size_t weightByteSize = 1 * sizeof(float);

    size_t tiling_data_size = sizeof(BatchNormV3WelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_var = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3WelfordTilingData* tilingDatafromBin = reinterpret_cast<BatchNormV3WelfordTilingData*>(tiling);

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
    tilingDatafromBin->r0UbFactor = 32;
    tilingDatafromBin->r0UbLoop = 1;
    tilingDatafromBin->r0UbTail = 5;
    tilingDatafromBin->procNR0 = 4;
    tilingDatafromBin->nR0Loop = 2;
    tilingDatafromBin->lastLoopNR0 = 1;
    tilingDatafromBin->patternR0Align = 8;
    tilingDatafromBin->dichotomizeAddDiffSize = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->momentumReverse = 0.9;
    tilingDatafromBin->batchVarScale = 1;

    ICPU_SET_TILING_KEY(1012);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(save_mean);
    AscendC::GmFree(save_var);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(batch_norm_v3_test, test_case_1003)
{
    size_t xByteSize = 8 * 1 * 1 * 5 * sizeof(float);
    size_t weightByteSize = 1 * sizeof(float);

    size_t tiling_data_size = sizeof(BatchNormV3WelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_var = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3WelfordTilingData* tilingDatafromBin = reinterpret_cast<BatchNormV3WelfordTilingData*>(tiling);

    tilingDatafromBin->patternR1 = 8;
    tilingDatafromBin->patternR0 = 5;
    tilingDatafromBin->patternA = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->aUbFactor = 16;
    tilingDatafromBin->aUbLoop = 1;
    tilingDatafromBin->aUbTail = 1;
    tilingDatafromBin->tailCoreAUbLoop = 1;
    tilingDatafromBin->tailCoreAUbTail = 1;
    tilingDatafromBin->r0UbFactor = 32;
    tilingDatafromBin->r0UbLoop = 1;
    tilingDatafromBin->r0UbTail = 5;
    tilingDatafromBin->procNR0 = 4;
    tilingDatafromBin->nR0Loop = 2;
    tilingDatafromBin->lastLoopNR0 = 4;
    tilingDatafromBin->patternR0Align = 8;
    tilingDatafromBin->dichotomizeAddDiffSize = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->momentumReverse = 0.9;
    tilingDatafromBin->batchVarScale = 1;

    ICPU_SET_TILING_KEY(1003);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(save_mean);
    AscendC::GmFree(save_var);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(batch_norm_v3_test, test_case_1013)
{
    size_t xByteSize = 8 * 1 * 1 * 8 * sizeof(float);
    size_t weightByteSize = 1 * sizeof(float);

    size_t tiling_data_size = sizeof(BatchNormV3WelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_var = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3WelfordTilingData* tilingDatafromBin = reinterpret_cast<BatchNormV3WelfordTilingData*>(tiling);

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
    tilingDatafromBin->r0UbFactor = 32;
    tilingDatafromBin->r0UbLoop = 1;
    tilingDatafromBin->r0UbTail = 5;
    tilingDatafromBin->procNR0 = 4;
    tilingDatafromBin->nR0Loop = 2;
    tilingDatafromBin->lastLoopNR0 = 4;
    tilingDatafromBin->patternR0Align = 8;
    tilingDatafromBin->dichotomizeAddDiffSize = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->momentumReverse = 0.9;
    tilingDatafromBin->batchVarScale = 1;

    ICPU_SET_TILING_KEY(1013);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(save_mean);
    AscendC::GmFree(save_var);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(batch_norm_v3_test, test_case_2000)
{
    size_t xByteSize = 8 * 1 * 1 * 8 * sizeof(float);
    size_t weightByteSize = 1 * sizeof(float);

    size_t tiling_data_size = sizeof(BatchNormV3FullReduceTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_var = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3FullReduceTilingData* tilingDatafromBin = reinterpret_cast<BatchNormV3FullReduceTilingData*>(tiling);

    tilingDatafromBin->patternR1 = 8;
    tilingDatafromBin->patternR0 = 8;
    tilingDatafromBin->patternA = 1;
    tilingDatafromBin->patternR0Align = 8;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->aUbFactor = 1;
    tilingDatafromBin->aUbLoop = 1;
    tilingDatafromBin->aUbTail = 1;
    tilingDatafromBin->tailCoreAUbLoop = 1;
    tilingDatafromBin->tailCoreAUbTail = 1;
    tilingDatafromBin->aUbSize = 16;
    tilingDatafromBin->rUbSize = 64;

    tilingDatafromBin->dichotomizeAddDiffSize = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->coefficient0 = 0.1;
    tilingDatafromBin->coefficient1 = 0.1;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->momentumReverse = 0.9;
    tilingDatafromBin->batchVarScale = 1;

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(save_mean);
    AscendC::GmFree(save_var);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(batch_norm_v3_test, test_case_2001)
{
    size_t xByteSize = 8 * 1 * 1 * 1 * sizeof(float);
    size_t weightByteSize = 1 * sizeof(float);

    size_t tiling_data_size = sizeof(BatchNormV3FullReduceTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* save_var = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    BatchNormV3FullReduceTilingData* tilingDatafromBin = reinterpret_cast<BatchNormV3FullReduceTilingData*>(tiling);

    tilingDatafromBin->patternR1 = 8;
    tilingDatafromBin->patternR0 = 1;
    tilingDatafromBin->patternA = 1;
    tilingDatafromBin->patternR0Align = 8;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->aUbFactor = 1;
    tilingDatafromBin->aUbLoop = 1;
    tilingDatafromBin->aUbTail = 1;
    tilingDatafromBin->tailCoreAUbLoop = 1;
    tilingDatafromBin->tailCoreAUbTail = 1;
    tilingDatafromBin->aUbSize = 16;
    tilingDatafromBin->rUbSize = 64;

    tilingDatafromBin->dichotomizeAddDiffSize = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->coefficient0 = 0.1;
    tilingDatafromBin->coefficient1 = 0.1;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->momentumReverse = 0.9;
    tilingDatafromBin->batchVarScale = 1;

    ICPU_SET_TILING_KEY(2001);
    ICPU_RUN_KF(
        batch_norm_v3, blockDim, x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(save_mean);
    AscendC::GmFree(save_var);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}