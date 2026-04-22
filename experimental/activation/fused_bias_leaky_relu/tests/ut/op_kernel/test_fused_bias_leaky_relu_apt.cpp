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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_kernel/arch35/fused_bias_leaky_relu_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void fused_bias_leaky_relu(GM_ADDR x, GM_ADDR bias, GM_ADDR y, 
                                                            GM_ADDR workspace, GM_ADDR tiling);

class FusedBiasLeakyReluKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "FusedBiasLeakyReluKernelTest SetUp" << endl;
    }
    static void TearDownTestCase() {
        cout << "FusedBiasLeakyReluKernelTest TearDown" << endl;
    }
};

static void RunKernelTest(size_t totalNum, ge::DataType dtype, float negativeSlope, float scale, uint64_t tilingKey)
{
    size_t typeSize = (dtype == ge::DT_FLOAT16) ? 2 : 4;
    size_t inputSize = totalNum * typeSize;
    size_t tilingDataSize = sizeof(FusedBiasLeakyReluTilingData);
    size_t workspaceSize = 32;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    FusedBiasLeakyReluTilingData* tilingData = reinterpret_cast<FusedBiasLeakyReluTilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(totalNum);
    tilingData->blockFactor = static_cast<int64_t>(totalNum);
    tilingData->ubFactor = static_cast<int64_t>(totalNum > 4096 ? 4096 : totalNum);
    tilingData->negativeSlope = negativeSlope;
    tilingData->scale = scale;

    uint32_t blockDim = 1;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fused_bias_leaky_relu, blockDim, x, bias, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(FusedBiasLeakyReluKernelTest, test_fp32_small)
{
    size_t totalNum = 8;
    float negativeSlope = 0.2f;
    float scale = 1.414213562373f;
    uint64_t tilingKey = 0;  // float32, single buffer
    RunKernelTest(totalNum, ge::DT_FLOAT, negativeSlope, scale, tilingKey);
}

TEST_F(FusedBiasLeakyReluKernelTest, test_fp32_medium)
{
    size_t totalNum = 1024;
    float negativeSlope = 0.2f;
    float scale = 1.414213562373f;
    uint64_t tilingKey = 0;
    RunKernelTest(totalNum, ge::DT_FLOAT, negativeSlope, scale, tilingKey);
}

TEST_F(FusedBiasLeakyReluKernelTest, test_fp32_large)
{
    size_t totalNum = 8192;
    float negativeSlope = 0.2f;
    float scale = 1.414213562373f;
    uint64_t tilingKey = 1;  // float32, double buffer
    RunKernelTest(totalNum, ge::DT_FLOAT, negativeSlope, scale, tilingKey);
}

TEST_F(FusedBiasLeakyReluKernelTest, test_fp16_small)
{
    size_t totalNum = 8;
    float negativeSlope = 0.2f;
    float scale = 1.414213562373f;
    uint64_t tilingKey = 1000;  // float16, single buffer (key base offset)
    RunKernelTest(totalNum, ge::DT_FLOAT16, negativeSlope, scale, tilingKey);
}

TEST_F(FusedBiasLeakyReluKernelTest, test_fp16_medium)
{
    size_t totalNum = 1024;
    float negativeSlope = 0.2f;
    float scale = 1.414213562373f;
    uint64_t tilingKey = 1000;
    RunKernelTest(totalNum, ge::DT_FLOAT16, negativeSlope, scale, tilingKey);
}

TEST_F(FusedBiasLeakyReluKernelTest, test_fp16_large)
{
    size_t totalNum = 8192;
    float negativeSlope = 0.2f;
    float scale = 1.414213562373f;
    uint64_t tilingKey = 1001;  // float16, double buffer
    RunKernelTest(totalNum, ge::DT_FLOAT16, negativeSlope, scale, tilingKey);
}

TEST_F(FusedBiasLeakyReluKernelTest, test_negative_slope_zero)
{
    size_t totalNum = 1024;
    float negativeSlope = 0.0f;  // standard ReLU
    float scale = 1.0f;
    uint64_t tilingKey = 0;
    RunKernelTest(totalNum, ge::DT_FLOAT, negativeSlope, scale, tilingKey);
}

TEST_F(FusedBiasLeakyReluKernelTest, test_scale_one)
{
    size_t totalNum = 1024;
    float negativeSlope = 0.2f;
    float scale = 1.0f;  // no scaling
    uint64_t tilingKey = 0;
    RunKernelTest(totalNum, ge::DT_FLOAT, negativeSlope, scale, tilingKey);
}

TEST_F(FusedBiasLeakyReluKernelTest, test_large_negative_slope)
{
    size_t totalNum = 1024;
    float negativeSlope = 0.5f;
    float scale = 1.414213562373f;
    uint64_t tilingKey = 0;
    RunKernelTest(totalNum, ge::DT_FLOAT, negativeSlope, scale, tilingKey);
}

TEST_F(FusedBiasLeakyReluKernelTest, test_very_large_shape)
{
    size_t totalNum = 65536;
    float negativeSlope = 0.2f;
    float scale = 1.414213562373f;
    uint64_t tilingKey = 1;  // double buffer for large data
    RunKernelTest(totalNum, ge::DT_FLOAT, negativeSlope, scale, tilingKey);
}