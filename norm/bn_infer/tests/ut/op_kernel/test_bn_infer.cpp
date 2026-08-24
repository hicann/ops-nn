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
 * \file test_bn_infer.cpp
 * \brief BNInfer arch35 kernel UT.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void bn_infer(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean, GM_ADDR variance,
                                               GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

using namespace std;

namespace {
constexpr float EPSILON = 1e-5f;
constexpr size_t WORKSPACE_BYTES = 16 * 1024 * 1024;

float Golden(float x, float scale, float offset, float mean, float variance)
{
    return scale * (x - mean) / std::sqrt(variance + EPSILON) + offset;
}

void FillParams(float* scale, float* offset, float* mean, float* variance, int64_t channel)
{
    for (int64_t c = 0; c < channel; ++c) {
        scale[c] = 0.5f + 0.25f * c;
        offset[c] = -0.2f + 0.1f * c;
        mean[c] = 0.1f * c;
        variance[c] = 0.5f + 0.2f * c;
    }
}

void CheckClose(const float* actual, const vector<float>& expected)
{
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "mismatch at index " << i;
    }
}
} // namespace

class BNInferKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "BNInferKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "BNInferKernelTest TearDown" << endl; }
};

TEST_F(BNInferKernelTest, fp32TilingKey910000ND)
{
    constexpr int64_t n = 1;
    constexpr int64_t c = 2;
    constexpr int64_t hw = 8;
    constexpr int64_t total = n * c * hw;

    auto* x = reinterpret_cast<float*>(AscendC::GmAlloc(total * sizeof(float)));
    auto* scale = reinterpret_cast<float*>(AscendC::GmAlloc(c * sizeof(float)));
    auto* offset = reinterpret_cast<float*>(AscendC::GmAlloc(c * sizeof(float)));
    auto* mean = reinterpret_cast<float*>(AscendC::GmAlloc(c * sizeof(float)));
    auto* variance = reinterpret_cast<float*>(AscendC::GmAlloc(c * sizeof(float)));
    auto* y = reinterpret_cast<float*>(AscendC::GmAlloc(total * sizeof(float)));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(WORKSPACE_BYTES));
    auto* tiling = reinterpret_cast<BNInferTilingData*>(AscendC::GmAlloc(sizeof(BNInferTilingData)));

    for (int64_t i = 0; i < total; ++i) {
        x[i] = -1.0f + 0.2f * i;
        y[i] = 0.0f;
    }
    FillParams(scale, offset, mean, variance, c);
    *tiling = {1, 1, c, hw, 1, 1, 1, n, n, c, c, hw, hw, 0, EPSILON};

    ICPU_SET_TILING_KEY(910000);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(bn_infer, 1, reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(scale),
                reinterpret_cast<GM_ADDR>(offset), reinterpret_cast<GM_ADDR>(mean), reinterpret_cast<GM_ADDR>(variance),
                reinterpret_cast<GM_ADDR>(y), reinterpret_cast<GM_ADDR>(workspace), reinterpret_cast<GM_ADDR>(tiling));

    vector<float> expected(total);
    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t ci = 0; ci < c; ++ci) {
            for (int64_t hi = 0; hi < hw; ++hi) {
                int64_t idx = ni * c * hw + ci * hw + hi;
                expected[idx] = Golden(x[idx], scale[ci], offset[ci], mean[ci], variance[ci]);
            }
        }
    }
    CheckClose(y, expected);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(BNInferKernelTest, fp32TilingKey900000NHWC)
{
    constexpr int64_t b = 2;
    constexpr int64_t c = 16;
    constexpr int64_t total = b * c;

    auto* x = reinterpret_cast<float*>(AscendC::GmAlloc(total * sizeof(float)));
    auto* scale = reinterpret_cast<float*>(AscendC::GmAlloc(c * sizeof(float)));
    auto* offset = reinterpret_cast<float*>(AscendC::GmAlloc(c * sizeof(float)));
    auto* mean = reinterpret_cast<float*>(AscendC::GmAlloc(c * sizeof(float)));
    auto* variance = reinterpret_cast<float*>(AscendC::GmAlloc(c * sizeof(float)));
    auto* y = reinterpret_cast<float*>(AscendC::GmAlloc(total * sizeof(float)));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(WORKSPACE_BYTES));
    auto* tiling = reinterpret_cast<BNInferLastChannelTilingData*>(
        AscendC::GmAlloc(sizeof(BNInferLastChannelTilingData)));

    for (int64_t i = 0; i < total; ++i) {
        x[i] = 0.1f * (i + 1);
        y[i] = 0.0f;
    }
    FillParams(scale, offset, mean, variance, c);
    *tiling = {2, 1, c, 1, 2, c, c, 0, 1, 1, EPSILON};

    ICPU_SET_TILING_KEY(900000);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(bn_infer, 2, reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(scale),
                reinterpret_cast<GM_ADDR>(offset), reinterpret_cast<GM_ADDR>(mean), reinterpret_cast<GM_ADDR>(variance),
                reinterpret_cast<GM_ADDR>(y), reinterpret_cast<GM_ADDR>(workspace), reinterpret_cast<GM_ADDR>(tiling));

    vector<float> expected(total);
    for (int64_t bi = 0; bi < b; ++bi) {
        for (int64_t ci = 0; ci < c; ++ci) {
            int64_t idx = bi * c + ci;
            expected[idx] = Golden(x[idx], scale[ci], offset[ci], mean[ci], variance[ci]);
        }
    }
    CheckClose(y, expected);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
