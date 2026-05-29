/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fake_quant_with_min_max_args.cpp
 * \brief CPU-side mock invocation of the FakeQuantWithMinMaxArgs regbase kernel.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/arch35/fake_quant_with_min_max_args.cpp"

using namespace std;

extern "C" __global__ __aicore__ void fake_quant_with_min_max_args(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class FakeQuantWithMinMaxArgsTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "FakeQuantWithMinMaxArgsTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "FakeQuantWithMinMaxArgsTest TearDown\n" << endl;
    }
};

 static void FillTiling(FakeQuantWithMinMaxArgsTilingData* tilingDatafromBin, int64_t totalLen, int64_t numCore)
{
    tilingDatafromBin->totalLen = totalLen;
    tilingDatafromBin->numCore = numCore;
    tilingDatafromBin->blockFactor = totalLen / numCore;
    tilingDatafromBin->blockTailFactor = totalLen - tilingDatafromBin->blockFactor * (numCore - 1);
    tilingDatafromBin->baseLen = 8192;
    tilingDatafromBin->nudgedMin = -6.0f;
    tilingDatafromBin->nudgedMax = 6.0f;
    tilingDatafromBin->scale = 12.0f / 255.0f;
    tilingDatafromBin->scaleInv = 255.0f / 12.0f;
    tilingDatafromBin->quantZero = 128.0f;
}

// CPU golden matching kernel formula exactly:
//   if (isnan(x)) y = x;
//   else: c = clamp(x, nMin, nMax)
//         q = floor((c - nMin) * scaleInv + 0.5 - quantZero)
//         y = q * scale
static float ForwardGolden(float x, const FakeQuantWithMinMaxArgsTilingData& t)
{
    if (std::isnan(x)) {
        return x;
    }
    float c = x;
    if (c > t.nudgedMax) c = t.nudgedMax;
    if (c < t.nudgedMin) c = t.nudgedMin;
    float scaled = (c - t.nudgedMin) * t.scaleInv + (0.5f - t.quantZero);
    float q = std::floor(scaled);
    return q * t.scale;
}

 TEST_F(FakeQuantWithMinMaxArgsTest, test_case_fp32_default)
{
    int64_t totalLen = 1024 * 256;
    uint32_t blockDim = 2;
    size_t inputXSize = totalLen * sizeof(float);
    size_t outputYSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsTilingData* tilingDatafromBin = reinterpret_cast<FakeQuantWithMinMaxArgsTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);
    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args<0>(x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, x, y, workSpace, tiling);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}

TEST_F(FakeQuantWithMinMaxArgsTest, test_case_fp32_narrow_range)
{
    int64_t totalLen = 1024 * 256;
    uint32_t blockDim = 2;
    size_t inputXSize = totalLen * sizeof(float);
    size_t outputYSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsTilingData* tilingDatafromBin = reinterpret_cast<FakeQuantWithMinMaxArgsTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);
    tilingDatafromBin->nudgedMin = -6.0f + 12.0f / 255.0f;
    tilingDatafromBin->nudgedMax = 6.0f - 12.0f / 255.0f;
    tilingDatafromBin->scale = 12.0f / 254.0f;
    tilingDatafromBin->scaleInv = 254.0f / 12.0f;
    tilingDatafromBin->quantZero = 1.0f;
    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args<0>(x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, x, y, workSpace, tiling);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}

TEST_F(FakeQuantWithMinMaxArgsTest, test_case_fp32_4bits)
{
    int64_t totalLen = 1024 * 256;
    uint32_t blockDim = 2;
    size_t inputXSize = totalLen * sizeof(float);
    size_t outputYSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsTilingData* tilingDatafromBin = reinterpret_cast<FakeQuantWithMinMaxArgsTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);
    tilingDatafromBin->nudgedMin = -1.0f;
    tilingDatafromBin->nudgedMax = 1.0f;
    tilingDatafromBin->scale = 2.0f / 15.0f;
    tilingDatafromBin->scaleInv = 15.0f / 2.0f;
    tilingDatafromBin->quantZero = 7.5f;
    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args<0>(x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, x, y, workSpace, tiling);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}

TEST_F(FakeQuantWithMinMaxArgsTest, test_case_fp32_small_shape)
{
    int64_t totalLen = 128;
    uint32_t blockDim = 1;
    size_t inputXSize = totalLen * sizeof(float);
    size_t outputYSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsTilingData* tilingDatafromBin = reinterpret_cast<FakeQuantWithMinMaxArgsTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);
    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args<0>(x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, x, y, workSpace, tiling);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}

// -----------------------------------------------------------------------------
// Real-input + output-assertion cases (correctness verification).
//
// NOTE on the CPU simulator (tikicpulib + ICPU_RUN_KF):
//   This kernel is an arch35 / DAV_3510 RegBase implementation. Its hot loop
//   uses AscendC::Reg::* vector intrinsics inside a __VEC_SCOPE__ block
//   (Reg::DataCopy / Compare / Select / Cast / Adds / Muls / etc.). The
//   tikicpulib CPU simulator shipped with CANN 9.0.0-beta.2 does NOT
//   currently lower these RegBase intrinsics -- at runtime it logs
//   "scalar_pv.cc:204 exec unknown instr->name()" for every vector op and
//   then exits "successfully" without producing any output, so y stays at
//   whatever GmAlloc returned. As a result, ICPU_RUN_KF cannot validate
//   compute correctness for RegBase kernels today.
//
//   The cases below are written with real inputs and a CPU golden that
//   matches the kernel formula exactly, so they will produce meaningful
//   assertions the moment they are run on real hardware (or on a future
//   simulator that supports RegBase). On the current CPU-only path they
//   are marked GTEST_SKIP so that the kernel-UT suite stays green.
// -----------------------------------------------------------------------------

// Case A: default 8-bit tiling, blockDim=1, small aligned shape.
// Coverage: in-range positive / in-range negative / zero / x<nudgedMin /
// x>nudgedMax / NaN passthrough.
TEST_F(FakeQuantWithMinMaxArgsTest, test_case_fp32_compute_default_with_assert)
{
    GTEST_SKIP() << "tikicpulib CPU simulator does not execute RegBase vector "
                    "intrinsics (Reg::* / __VEC_SCOPE__); assertions can only "
                    "run on real NPU hardware. Test body kept for hardware runs.";

    int64_t totalLen = 32;          // 8-aligned, small
    uint32_t blockDim = 1;          // single block simplifies golden mapping
    size_t inputXSize = totalLen * sizeof(float);
    size_t outputYSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsTilingData);

    uint8_t* xBuf = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* yBuf = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsTilingData* tilingDatafromBin =
        reinterpret_cast<FakeQuantWithMinMaxArgsTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);

    // Fill x with known values; remaining slots filled with arbitrary in-range
    // values that exercise the rounding path.
    float* xFloat = reinterpret_cast<float*>(xBuf);
    std::vector<float> seeds = {
        0.0f, 1.0f, -1.0f, 2.5f, -3.7f, 5.999f, -5.999f,         // in-range
        7.5f, 12.0f, 1.0e30f,                                    // > nudgedMax
        -7.5f, -12.0f, -1.0e30f,                                 // < nudgedMin
        std::nanf(""), -std::nanf(""),                           // NaN
        6.0f, -6.0f,                                             // boundary
    };
    for (int64_t i = 0; i < totalLen; ++i) {
        xFloat[i] = (i < (int64_t)seeds.size()) ? seeds[i] : (float)(i % 13) * 0.37f - 2.0f;
    }

    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args<0>(x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, xBuf, yBuf, workSpace, tiling);

    float* yFloat = reinterpret_cast<float*>(yBuf);
    for (int64_t i = 0; i < totalLen; ++i) {
        float xi = xFloat[i];
        float gold = ForwardGolden(xi, *tilingDatafromBin);
        float got = yFloat[i];
        if (std::isnan(xi)) {
            EXPECT_TRUE(std::isnan(got)) << "idx=" << i << " expected NaN passthrough";
        } else {
            EXPECT_NEAR(got, gold, 1e-5f)
                << "idx=" << i << " x=" << xi << " gold=" << gold << " got=" << got;
        }
    }

    AscendC::GmFree(xBuf);
    AscendC::GmFree(yBuf);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}

// Case B: narrow-range tiling, custom nudged scalars and quantZero=1.
// Coverage: tiling-scalar parameterisation actually flows through to compute.
TEST_F(FakeQuantWithMinMaxArgsTest, test_case_fp32_compute_narrow_range_with_assert)
{
    GTEST_SKIP() << "tikicpulib CPU simulator does not execute RegBase vector "
                    "intrinsics (Reg::* / __VEC_SCOPE__); assertions can only "
                    "run on real NPU hardware. Test body kept for hardware runs.";

    int64_t totalLen = 64;          // 8-aligned, small
    uint32_t blockDim = 1;
    size_t inputXSize = totalLen * sizeof(float);
    size_t outputYSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsTilingData);

    uint8_t* xBuf = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* yBuf = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsTilingData* tilingDatafromBin =
        reinterpret_cast<FakeQuantWithMinMaxArgsTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);
    // narrow-range overrides
    tilingDatafromBin->nudgedMin = -6.0f + 12.0f / 255.0f;
    tilingDatafromBin->nudgedMax = 6.0f - 12.0f / 255.0f;
    tilingDatafromBin->scale = 12.0f / 254.0f;
    tilingDatafromBin->scaleInv = 254.0f / 12.0f;
    tilingDatafromBin->quantZero = 1.0f;

    float* xFloat = reinterpret_cast<float*>(xBuf);
    for (int64_t i = 0; i < totalLen; ++i) {
        // Spread values across the range, plus push some outside.
        float v;
        if (i == 0) v = std::nanf("");
        else if (i == 1) v = 1.0e20f;
        else if (i == 2) v = -1.0e20f;
        else if (i == 3) v = 0.0f;
        else v = -7.0f + 0.21875f * (float)i;
        xFloat[i] = v;
    }

    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args<0>(x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, xBuf, yBuf, workSpace, tiling);

    float* yFloat = reinterpret_cast<float*>(yBuf);
    for (int64_t i = 0; i < totalLen; ++i) {
        float xi = xFloat[i];
        float gold = ForwardGolden(xi, *tilingDatafromBin);
        float got = yFloat[i];
        if (std::isnan(xi)) {
            EXPECT_TRUE(std::isnan(got)) << "idx=" << i << " expected NaN passthrough";
        } else {
            EXPECT_NEAR(got, gold, 1e-5f)
                << "idx=" << i << " x=" << xi << " gold=" << gold << " got=" << got;
        }
    }

    AscendC::GmFree(xBuf);
    AscendC::GmFree(yBuf);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}
