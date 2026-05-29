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
 * \file test_fake_quant_with_min_max_args_gradient.cpp
 * \brief CPU-side mock invocation of the FakeQuantWithMinMaxArgsGradient regbase kernel.
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
#include "../../../op_kernel/arch35/fake_quant_with_min_max_args_gradient.cpp"

using namespace std;

extern "C" __global__ __aicore__ void fake_quant_with_min_max_args_gradient(
    GM_ADDR gradients, GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class FakeQuantWithMinMaxArgsGradientTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "FakeQuantWithMinMaxArgsGradientTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "FakeQuantWithMinMaxArgsGradientTest TearDown\n" << endl;
    }
};

static void FillTiling(FakeQuantWithMinMaxArgsGradientTilingData* tilingDatafromBin, int64_t totalLen, int64_t numCore)
{
    tilingDatafromBin->totalLen = totalLen;
    tilingDatafromBin->numCore = numCore;
    tilingDatafromBin->blockFactor = totalLen / numCore;
    tilingDatafromBin->blockTailFactor = totalLen - tilingDatafromBin->blockFactor * (numCore - 1);
    tilingDatafromBin->baseLen = 8192;
    tilingDatafromBin->nudgedMin = -6.0f;
    tilingDatafromBin->nudgedMax = 6.0f;
}

// CPU golden matching kernel formula exactly:
//   mask = (x >= nMin && x <= nMax) ? 1 : 0   (NaN x -> mask=0)
//   y    = grad * mask                         (with ±0 sign re-inject from grad)
// NaN grad on in-range x -> NaN * 1 = NaN (propagates).
// NaN grad on out-of-range x -> NaN * 0 = NaN per IEEE754 (kernel uses Mul, so
// this also propagates as NaN). We mark that case explicitly.
static float GradientGolden(float grad, float xi,
                            const FakeQuantWithMinMaxArgsGradientTilingData& t,
                            bool& expectNan)
{
    expectNan = false;
    bool inRange = !std::isnan(xi) && (xi >= t.nudgedMin) && (xi <= t.nudgedMax);
    float mask = inRange ? 1.0f : 0.0f;
    if (std::isnan(grad)) {
        // grad NaN * 1 -> NaN; grad NaN * 0 -> NaN as well per IEEE754.
        expectNan = true;
        return std::nanf("");
    }
    float prod = grad * mask;
    // Sign re-injection: if grad has sign bit set, OR it back into prod (preserves -0).
    uint32_t gradBits, prodBits;
    std::memcpy(&gradBits, &grad, sizeof(uint32_t));
    std::memcpy(&prodBits, &prod, sizeof(uint32_t));
    uint32_t outBits = prodBits | (gradBits & 0x80000000u);
    float out;
    std::memcpy(&out, &outBits, sizeof(uint32_t));
    return out;
}

TEST_F(FakeQuantWithMinMaxArgsGradientTest, test_case_fp32_default)
{
    int64_t totalLen = 1024 * 256;
    uint32_t blockDim = 2;

    size_t inputGradSize = totalLen * sizeof(float);
    size_t inputXSize = totalLen * sizeof(float);
    size_t outputYSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsGradientTilingData);

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(inputGradSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsGradientTilingData* tilingDatafromBin = reinterpret_cast<FakeQuantWithMinMaxArgsGradientTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);

    ICPU_SET_TILING_KEY(0);

    auto kernel_lambda = [](GM_ADDR gradients, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args_gradient<0>(gradients, x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, gradients, x, y, workSpace, tiling);

    AscendC::GmFree(gradients);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}

TEST_F(FakeQuantWithMinMaxArgsGradientTest, test_case_fp32_narrow_range)
{
    int64_t totalLen = 1024 * 256;
    uint32_t blockDim = 2;

    size_t inputGradSize = totalLen * sizeof(float);
    size_t inputXSize = totalLen * sizeof(float);
    size_t outputYSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsGradientTilingData);

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(inputGradSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsGradientTilingData* tilingDatafromBin = reinterpret_cast<FakeQuantWithMinMaxArgsGradientTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);
    tilingDatafromBin->nudgedMin = -6.0f + 12.0f / 255.0f;
    tilingDatafromBin->nudgedMax = 6.0f - 12.0f / 255.0f;

    ICPU_SET_TILING_KEY(0);

    auto kernel_lambda = [](GM_ADDR gradients, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args_gradient<0>(gradients, x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, gradients, x, y, workSpace, tiling);

    AscendC::GmFree(gradients);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}

TEST_F(FakeQuantWithMinMaxArgsGradientTest, test_case_fp32_4bits)
{
    int64_t totalLen = 1024 * 256;
    uint32_t blockDim = 2;

    size_t inputGradSize = totalLen * sizeof(float);
    size_t inputXSize = totalLen * sizeof(float);
    size_t outputYSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsGradientTilingData);

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(inputGradSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsGradientTilingData* tilingDatafromBin = reinterpret_cast<FakeQuantWithMinMaxArgsGradientTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);
    tilingDatafromBin->nudgedMin = -1.0f;
    tilingDatafromBin->nudgedMax = 1.0f;

    ICPU_SET_TILING_KEY(0);

    auto kernel_lambda = [](GM_ADDR gradients, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args_gradient<0>(gradients, x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, gradients, x, y, workSpace, tiling);

    AscendC::GmFree(gradients);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}

TEST_F(FakeQuantWithMinMaxArgsGradientTest, test_case_fp32_small_shape)
{
    int64_t totalLen = 128;
    uint32_t blockDim = 1;

    size_t inputGradSize = totalLen * sizeof(float);
    size_t inputXSize = totalLen * sizeof(float);
    size_t outputYSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsGradientTilingData);

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(inputGradSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsGradientTilingData* tilingDatafromBin = reinterpret_cast<FakeQuantWithMinMaxArgsGradientTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);

    ICPU_SET_TILING_KEY(0);

    auto kernel_lambda = [](GM_ADDR gradients, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args_gradient<0>(gradients, x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, gradients, x, y, workSpace, tiling);

    AscendC::GmFree(gradients);
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
//   uses AscendC::Reg::* vector intrinsics inside __VEC_SCOPE__ (Reg::DataCopy
//   / CompareScalar / Select / Mul / And / Or / etc.). The tikicpulib CPU
//   simulator shipped with CANN 9.0.0-beta.2 does NOT currently lower these
//   RegBase intrinsics -- it logs "scalar_pv.cc:204 exec unknown instr->name()"
//   and exits without producing output, so y stays at whatever GmAlloc gave
//   back. ICPU_RUN_KF therefore cannot validate compute correctness for
//   RegBase kernels today.
//
//   The cases below are written with real grad/x inputs and a CPU golden
//   matching the kernel formula exactly. They will produce meaningful
//   assertions the moment they are executed on real NPU hardware (or on a
//   future simulator that supports RegBase). On the current CPU-only path
//   they are marked GTEST_SKIP so the kernel-UT suite stays green.
// -----------------------------------------------------------------------------

// Case A: default tiling, blockDim=1, small aligned shape.
// Coverage: in-range pass-through / x>nMax → 0 / x<nMin → 0 / x=NaN → 0 (mask=0)
//           + grad=NaN propagation / grad=±0 sign preservation.
TEST_F(FakeQuantWithMinMaxArgsGradientTest, test_case_fp32_compute_default_with_assert)
{
    GTEST_SKIP() << "tikicpulib CPU simulator does not execute RegBase vector "
                    "intrinsics (Reg::* / __VEC_SCOPE__); assertions can only "
                    "run on real NPU hardware. Test body kept for hardware runs.";

    int64_t totalLen = 32;
    uint32_t blockDim = 1;

    size_t bufSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsGradientTilingData);

    uint8_t* gradBuf = (uint8_t*)AscendC::GmAlloc(bufSize);
    uint8_t* xBuf = (uint8_t*)AscendC::GmAlloc(bufSize);
    uint8_t* yBuf = (uint8_t*)AscendC::GmAlloc(bufSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsGradientTilingData* tilingDatafromBin =
        reinterpret_cast<FakeQuantWithMinMaxArgsGradientTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);

    float* gradFloat = reinterpret_cast<float*>(gradBuf);
    float* xFloat = reinterpret_cast<float*>(xBuf);

    // Designed coverage points:
    //   idx 0:  x=0,    grad=2.0     in-range pass-through  -> 2.0
    //   idx 1:  x=3.0,  grad=-1.5    in-range pass-through  -> -1.5
    //   idx 2:  x=-6.0, grad=4.0     boundary in-range      -> 4.0
    //   idx 3:  x=6.0,  grad=-2.5    boundary in-range      -> -2.5
    //   idx 4:  x=7.0,  grad=9.0     x>nMax                  -> 0
    //   idx 5:  x=100,  grad=-3.0    x>nMax                  -> 0 (with -0 sign)
    //   idx 6:  x=-7.0, grad=1.0     x<nMin                  -> 0
    //   idx 7:  x=-100, grad=-7.0    x<nMin                  -> 0 (with -0 sign)
    //   idx 8:  x=NaN,  grad=5.0     NaN x                   -> 0
    //   idx 9:  x=NaN,  grad=-5.0    NaN x                   -> 0 (with -0 sign)
    //   idx 10: x=0.5,  grad=NaN     NaN grad propagate      -> NaN
    //   idx 11: x=8.0,  grad=NaN     NaN grad out-of-range   -> NaN (Mul: NaN*0=NaN)
    //   remaining idx: in-range varied values
    for (int64_t i = 0; i < totalLen; ++i) {
        xFloat[i] = 0.1f * (float)i - 1.0f;     // default in-range
        gradFloat[i] = 0.25f * (float)i + 0.5f; // default finite
    }
    xFloat[0] = 0.0f;    gradFloat[0] = 2.0f;
    xFloat[1] = 3.0f;    gradFloat[1] = -1.5f;
    xFloat[2] = -6.0f;   gradFloat[2] = 4.0f;
    xFloat[3] = 6.0f;    gradFloat[3] = -2.5f;
    xFloat[4] = 7.0f;    gradFloat[4] = 9.0f;
    xFloat[5] = 100.0f;  gradFloat[5] = -3.0f;
    xFloat[6] = -7.0f;   gradFloat[6] = 1.0f;
    xFloat[7] = -100.0f; gradFloat[7] = -7.0f;
    xFloat[8] = std::nanf("");  gradFloat[8] = 5.0f;
    xFloat[9] = std::nanf("");  gradFloat[9] = -5.0f;
    xFloat[10] = 0.5f;   gradFloat[10] = std::nanf("");
    xFloat[11] = 8.0f;   gradFloat[11] = std::nanf("");

    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR gradients, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args_gradient<0>(gradients, x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, gradBuf, xBuf, yBuf, workSpace, tiling);

    float* yFloat = reinterpret_cast<float*>(yBuf);
    for (int64_t i = 0; i < totalLen; ++i) {
        bool expectNan = false;
        float gold = GradientGolden(gradFloat[i], xFloat[i], *tilingDatafromBin, expectNan);
        float got = yFloat[i];
        if (expectNan) {
            EXPECT_TRUE(std::isnan(got))
                << "idx=" << i << " grad=" << gradFloat[i] << " x=" << xFloat[i]
                << " expected NaN, got=" << got;
        } else {
            EXPECT_NEAR(got, gold, 1e-5f)
                << "idx=" << i << " grad=" << gradFloat[i] << " x=" << xFloat[i]
                << " gold=" << gold << " got=" << got;
        }
    }

    AscendC::GmFree(gradBuf);
    AscendC::GmFree(xBuf);
    AscendC::GmFree(yBuf);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}

// Case B: narrow-range tiling (different nudgedMin/Max) exercises tiling-scalar wiring.
TEST_F(FakeQuantWithMinMaxArgsGradientTest, test_case_fp32_compute_narrow_range_with_assert)
{
    GTEST_SKIP() << "tikicpulib CPU simulator does not execute RegBase vector "
                    "intrinsics (Reg::* / __VEC_SCOPE__); assertions can only "
                    "run on real NPU hardware. Test body kept for hardware runs.";

    int64_t totalLen = 64;
    uint32_t blockDim = 1;

    size_t bufSize = totalLen * sizeof(float);
    size_t tiling_data_size = sizeof(FakeQuantWithMinMaxArgsGradientTilingData);

    uint8_t* gradBuf = (uint8_t*)AscendC::GmAlloc(bufSize);
    uint8_t* xBuf = (uint8_t*)AscendC::GmAlloc(bufSize);
    uint8_t* yBuf = (uint8_t*)AscendC::GmAlloc(bufSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FakeQuantWithMinMaxArgsGradientTilingData* tilingDatafromBin =
        reinterpret_cast<FakeQuantWithMinMaxArgsGradientTilingData*>(tiling);
    FillTiling(tilingDatafromBin, totalLen, blockDim);
    tilingDatafromBin->nudgedMin = -1.0f;
    tilingDatafromBin->nudgedMax = 1.0f;

    float* gradFloat = reinterpret_cast<float*>(gradBuf);
    float* xFloat = reinterpret_cast<float*>(xBuf);
    for (int64_t i = 0; i < totalLen; ++i) {
        // Sweep x across [-2, 2] so half are inside [-1,1], half outside.
        xFloat[i] = -2.0f + 4.0f * (float)i / (float)(totalLen - 1);
        gradFloat[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }
    // Inject special points
    xFloat[0] = std::nanf("");   gradFloat[0] = 3.14f;     // NaN x
    xFloat[1] = 0.0f;            gradFloat[1] = std::nanf(""); // NaN grad in-range
    xFloat[2] = -1.0f;           gradFloat[2] = 7.0f;      // boundary
    xFloat[3] = 1.0f;            gradFloat[3] = -8.0f;     // boundary

    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR gradients, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::fake_quant_with_min_max_args_gradient<0>(gradients, x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, gradBuf, xBuf, yBuf, workSpace, tiling);

    float* yFloat = reinterpret_cast<float*>(yBuf);
    for (int64_t i = 0; i < totalLen; ++i) {
        bool expectNan = false;
        float gold = GradientGolden(gradFloat[i], xFloat[i], *tilingDatafromBin, expectNan);
        float got = yFloat[i];
        if (expectNan) {
            EXPECT_TRUE(std::isnan(got))
                << "idx=" << i << " grad=" << gradFloat[i] << " x=" << xFloat[i]
                << " expected NaN, got=" << got;
        } else {
            EXPECT_NEAR(got, gold, 1e-5f)
                << "idx=" << i << " grad=" << gradFloat[i] << " x=" << xFloat[i]
                << " gold=" << gold << " got=" << got;
        }
    }

    AscendC::GmFree(gradBuf);
    AscendC::GmFree(xBuf);
    AscendC::GmFree(yBuf);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}
