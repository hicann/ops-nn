/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void swiglu_group_quant(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR scale,
                                                         GM_ADDR y, GM_ADDR yScale, GM_ADDR yOrigin, GM_ADDR workspace,
                                                         GM_ADDR tiling);

namespace {
class SwigluGroupQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluGroupQuantKernelTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SwigluGroupQuantKernelTest TearDown" << std::endl; }
};

void RunKernelWithTilingKey(uint64_t tilingKey, bool outputOrigin, bool hasWeight = false)
{
    constexpr int64_t bs = 2;
    constexpr int64_t d = 256;
    constexpr int64_t splitD = d / 2;
    constexpr int64_t scaleCol = 1;
    constexpr uint32_t blockDim = 2;

    const size_t inputSize = bs * d * sizeof(half);
    const size_t outputYSize = bs * splitD * sizeof(uint8_t);
    const size_t outputScaleSize = bs * scaleCol * sizeof(float);
    const size_t yOriginSize = bs * splitD * sizeof(half);
    const size_t weightSize = bs * sizeof(float);
    const size_t tilingDataSize = sizeof(SwigluGroupQuantTilingData);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputYSize));
    uint8_t* yScale = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputScaleSize));
    uint8_t* yOrigin = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yOriginSize));
    uint8_t* weight = hasWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(weightSize)) : nullptr;
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SwigluGroupQuantTilingData*>(tiling);
    tilingData->bs = bs;
    tilingData->d = d;
    tilingData->splitD = splitD;
    tilingData->scaleCol = scaleCol;
    tilingData->rowOfFormerBlock = 1;
    tilingData->rowOfTailBlock = 1;
    tilingData->rowLoopOfFormerBlock = 1;
    tilingData->rowLoopOfTailBlock = 1;
    tilingData->rowFactor = 1;
    tilingData->tailRowFactorOfFormerBlock = 1;
    tilingData->tailRowFactorOfTailBlock = 1;
    tilingData->dLoop = 1;
    tilingData->dFactor = splitD;
    tilingData->tailDFactor = splitD;
    tilingData->roundScale = 0;
    tilingData->outputOrigin = outputOrigin ? 1 : 0;
    tilingData->clampLimit = 0.0f;
    tilingData->hasClampLimit = 0;
    tilingData->g = 0;
    tilingData->ubSize = 253952;
    tilingData->gLoop = 0;
    tilingData->gFactor = 0;
    tilingData->tailGFactor = 0;
    tilingData->coreNum = blockDim;

    ICPU_SET_TILING_KEY(tilingKey);
    auto swigluGroupQuantKernel = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR scale, GM_ADDR y,
                                     GM_ADDR yScale, GM_ADDR yOrigin, GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_group_quant(x, weight, groupIndex, scale, y, yScale, yOrigin, workspace, tiling);
    };
    ICPU_RUN_KF(swigluGroupQuantKernel, blockDim, x, weight, nullptr, nullptr, y, yScale, yOrigin, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(yScale);
    AscendC::GmFree(yOrigin);
    if (hasWeight) {
        AscendC::GmFree(weight);
    }
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluGroupQuantKernelTest, block_fp8) { RunKernelWithTilingKey(1000, false); }

TEST_F(SwigluGroupQuantKernelTest, block_fp8_y_origin) { RunKernelWithTilingKey(1100, true); }

TEST_F(SwigluGroupQuantKernelTest, block_fp8_y_origin_weight) { RunKernelWithTilingKey(1100, true, true); }

void RunMxKernelWithTilingKey(uint64_t tilingKey, bool outputOrigin, bool hasWeight = false)
{
    constexpr int64_t bs = 2;
    constexpr int64_t d = 256;
    constexpr int64_t splitD = d / 2;
    constexpr int64_t scaleCol = 4; // ceil((D/2)/32)
    constexpr uint32_t blockDim = 2;

    const size_t inputSize = bs * d * sizeof(half);
    const size_t outputYSize = bs * splitD * sizeof(uint8_t);
    // UT kernel is compiled with DTYPE_Y_SCALE=float, so the scale GM buffer is sized in float.
    const size_t outputScaleSize = bs * scaleCol * sizeof(float);
    const size_t yOriginSize = bs * splitD * sizeof(half);
    const size_t weightSize = bs * sizeof(float);
    const size_t tilingDataSize = sizeof(SwigluGroupQuantTilingData);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputYSize));
    uint8_t* yScale = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputScaleSize));
    uint8_t* yOrigin = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yOriginSize));
    uint8_t* weight = hasWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(weightSize)) : nullptr;
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SwigluGroupQuantTilingData*>(tiling);
    tilingData->bs = bs;
    tilingData->d = d;
    tilingData->splitD = splitD;
    tilingData->scaleCol = scaleCol;
    tilingData->rowOfFormerBlock = 1;
    tilingData->rowOfTailBlock = 1;
    tilingData->rowLoopOfFormerBlock = 1;
    tilingData->rowLoopOfTailBlock = 1;
    tilingData->rowFactor = 1;
    tilingData->tailRowFactorOfFormerBlock = 1;
    tilingData->tailRowFactorOfTailBlock = 1;
    tilingData->dLoop = 1;
    tilingData->dFactor = splitD;
    tilingData->tailDFactor = splitD;
    tilingData->roundScale = 1;
    tilingData->outputOrigin = outputOrigin ? 1 : 0;
    tilingData->clampLimit = 0.0f;
    tilingData->hasClampLimit = 0;
    tilingData->g = 0;
    tilingData->ubSize = 253952;
    tilingData->gLoop = 0;
    tilingData->gFactor = 0;
    tilingData->tailGFactor = 0;
    tilingData->coreNum = blockDim;

    ICPU_SET_TILING_KEY(tilingKey);
    auto swigluGroupQuantKernel = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR scale, GM_ADDR y,
                                     GM_ADDR yScale, GM_ADDR yOrigin, GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_group_quant(x, weight, groupIndex, scale, y, yScale, yOrigin, workspace, tiling);
    };
    ICPU_RUN_KF(swigluGroupQuantKernel, blockDim, x, weight, nullptr, nullptr, y, yScale, yOrigin, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(yScale);
    AscendC::GmFree(yOrigin);
    if (hasWeight) {
        AscendC::GmFree(weight);
    }
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluGroupQuantKernelTest, mx_fp8) { RunMxKernelWithTilingKey(2000, false); }

TEST_F(SwigluGroupQuantKernelTest, mx_fp8_y_origin) { RunMxKernelWithTilingKey(2100, true); }

TEST_F(SwigluGroupQuantKernelTest, mx_fp8_y_origin_weight) { RunMxKernelWithTilingKey(2100, true, true); }

void RunMxFp4KernelWithTilingKey(uint64_t tilingKey, bool outputOrigin, bool hasWeight = false)
{
    constexpr int64_t bs = 2;
    constexpr int64_t d = 256;
    constexpr int64_t splitD = d / 2;
    constexpr int64_t scaleCol = 4; // ceil((D/2)/32)
    constexpr uint32_t blockDim = 2;

    const size_t inputSize = bs * d * sizeof(half);
    // fp4 y is packed 2 elements per byte: D/2 elements occupy D/4 bytes.
    const size_t outputYSize = bs * splitD / 2;
    // UT kernel is compiled with DTYPE_Y_SCALE=float, so the scale GM buffer is sized in float.
    const size_t outputScaleSize = bs * scaleCol * sizeof(float);
    const size_t yOriginSize = bs * splitD * sizeof(half);
    const size_t weightSize = bs * sizeof(float);
    const size_t tilingDataSize = sizeof(SwigluGroupQuantTilingData);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputYSize));
    uint8_t* yScale = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputScaleSize));
    uint8_t* yOrigin = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yOriginSize));
    uint8_t* weight = hasWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(weightSize)) : nullptr;
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SwigluGroupQuantTilingData*>(tiling);
    tilingData->bs = bs;
    tilingData->d = d;
    tilingData->splitD = splitD;
    tilingData->scaleCol = scaleCol;
    tilingData->rowOfFormerBlock = 1;
    tilingData->rowOfTailBlock = 1;
    tilingData->rowLoopOfFormerBlock = 1;
    tilingData->rowLoopOfTailBlock = 1;
    tilingData->rowFactor = 1;
    tilingData->tailRowFactorOfFormerBlock = 1;
    tilingData->tailRowFactorOfTailBlock = 1;
    tilingData->dLoop = 1;
    tilingData->dFactor = splitD;
    tilingData->tailDFactor = splitD;
    tilingData->roundScale = 1;
    tilingData->outputOrigin = outputOrigin ? 1 : 0;
    tilingData->clampLimit = 0.0f;
    tilingData->hasClampLimit = 0;
    tilingData->g = 0;
    tilingData->ubSize = 253952;
    tilingData->gLoop = 0;
    tilingData->gFactor = 0;
    tilingData->tailGFactor = 0;
    tilingData->coreNum = blockDim;

    ICPU_SET_TILING_KEY(tilingKey);
    auto swigluGroupQuantKernel = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR scale, GM_ADDR y,
                                     GM_ADDR yScale, GM_ADDR yOrigin, GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_group_quant(x, weight, groupIndex, scale, y, yScale, yOrigin, workspace, tiling);
    };
    ICPU_RUN_KF(swigluGroupQuantKernel, blockDim, x, weight, nullptr, nullptr, y, yScale, yOrigin, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(yScale);
    AscendC::GmFree(yOrigin);
    if (hasWeight) {
        AscendC::GmFree(weight);
    }
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluGroupQuantKernelTest, mx_fp4) { RunMxFp4KernelWithTilingKey(3000, false); }

TEST_F(SwigluGroupQuantKernelTest, mx_fp4_y_origin) { RunMxFp4KernelWithTilingKey(3100, true); }

TEST_F(SwigluGroupQuantKernelTest, mx_fp4_y_origin_weight) { RunMxFp4KernelWithTilingKey(3100, true, true); }

void RunHifp8KernelWithTilingKey(uint64_t tilingKey, bool hasScale, bool outputOrigin)
{
    constexpr int64_t totalTokens = 4;
    constexpr int64_t dimH = 128;
    constexpr int64_t dim2H = 2 * dimH;
    constexpr int64_t groupNum = 1;
    constexpr int64_t usedCoreNum = 2;
    constexpr int64_t tokensPerCore = totalTokens / usedCoreNum;
    constexpr int64_t tileLength = tokensPerCore * dimH;
    constexpr uint32_t blockDim = 2;

    const size_t inputSize = totalTokens * dim2H * sizeof(half);
    const size_t outputYSize = totalTokens * dimH * sizeof(uint8_t);
    const size_t outputScaleSize = hasScale ? groupNum * sizeof(float) : 32;
    const size_t yOriginSize = outputOrigin ? totalTokens * dimH * sizeof(half) : 32;
    const size_t tilingDataSize = sizeof(SwigluGroupQuantHifp8TilingData);

    const size_t weightSize = totalTokens * sizeof(float);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputSize));
    uint8_t* weight = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(weightSize));
    uint8_t* scale = hasScale ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputScaleSize)) : nullptr;
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputYSize));
    uint8_t* yScale = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputScaleSize));
    uint8_t* yOrigin = outputOrigin ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yOriginSize)) : nullptr;
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SwigluGroupQuantHifp8TilingData*>(tiling);
    tilingData->totalTokens = totalTokens;
    tilingData->dim2H = dim2H;
    tilingData->dimH = dimH;
    tilingData->isGroup = 0;
    tilingData->hasWeight = 1;
    tilingData->hasClamp = 0;
    tilingData->outputOrigin = outputOrigin ? 1 : 0;
    tilingData->clampLimit = 0.0f;
    tilingData->dstTypeMax = 15.0f;
    tilingData->tileTokens = tokensPerCore;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->tokensPerCore = tokensPerCore;
    tilingData->groupNum = groupNum;
    tilingData->tileLength = tileLength;

    ICPU_SET_TILING_KEY(tilingKey);
    auto swigluGroupQuantKernel = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR scale, GM_ADDR y,
                                     GM_ADDR yScale, GM_ADDR yOrigin, GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_group_quant(x, weight, groupIndex, scale, y, yScale, yOrigin, workspace, tiling);
    };
    ICPU_RUN_KF(swigluGroupQuantKernel, blockDim, x, weight, nullptr, scale, y, yScale, yOrigin, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    if (hasScale)
        AscendC::GmFree(scale);
    AscendC::GmFree(y);
    AscendC::GmFree(yScale);
    if (outputOrigin)
        AscendC::GmFree(yOrigin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluGroupQuantKernelTest, hifp8_static) { RunHifp8KernelWithTilingKey(4100, true, false); }

TEST_F(SwigluGroupQuantKernelTest, hifp8_static_output_origin) { RunHifp8KernelWithTilingKey(4100, true, true); }

TEST_F(SwigluGroupQuantKernelTest, hifp8_dynamic) { RunHifp8KernelWithTilingKey(4000, false, false); }

TEST_F(SwigluGroupQuantKernelTest, hifp8_dynamic_output_origin) { RunHifp8KernelWithTilingKey(4000, false, true); }

float HostSilu(float v) { return v / (1.0f + std::exp(-v)); }

void RunHifp8YOriginWeightVerify(uint64_t tilingKey, bool isDynamic)
{
    constexpr int64_t totalTokens = 4;
    constexpr int64_t dimH = 128;
    constexpr int64_t dim2H = 2 * dimH;
    constexpr int64_t groupNum = 1;
    constexpr int64_t usedCoreNum = 2;
    constexpr int64_t tokensPerCore = totalTokens / usedCoreNum;
    constexpr int64_t tileLength = tokensPerCore * dimH;
    constexpr uint32_t blockDim = 2;
    constexpr float dstTypeMax = 15.0f;

    const float gateVals[totalTokens] = {2.0f, 1.0f, 1.5f, 0.5f};
    const float upVals[totalTokens] = {3.0f, 1.0f, 2.0f, 1.0f};
    const float weightVals[totalTokens] = {2.0f, 0.5f, 3.0f, 1.5f};

    const size_t inputSize = totalTokens * dim2H * sizeof(half);
    const size_t outputYSize = totalTokens * dimH * sizeof(uint8_t);
    const size_t outputScaleSize = groupNum * sizeof(float);
    const size_t yOriginSize = totalTokens * dimH * sizeof(half);
    const size_t weightSize = totalTokens * sizeof(float);
    const size_t tilingDataSize = sizeof(SwigluGroupQuantHifp8TilingData);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputSize));
    uint8_t* weight = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(weightSize));
    uint8_t* scale = isDynamic ? nullptr : reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputScaleSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputYSize));
    uint8_t* yScale = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputScaleSize));
    uint8_t* yOrigin = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yOriginSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    auto* xHalf = reinterpret_cast<half*>(x);
    for (int64_t t = 0; t < totalTokens; t++) {
        for (int64_t j = 0; j < dimH; j++) {
            xHalf[t * dim2H + j] = static_cast<half>(gateVals[t]);
            xHalf[t * dim2H + dimH + j] = static_cast<half>(upVals[t]);
        }
    }
    auto* weightFp32 = reinterpret_cast<float*>(weight);
    for (int64_t t = 0; t < totalTokens; t++) {
        weightFp32[t] = weightVals[t];
    }
    if (!isDynamic) {
        reinterpret_cast<float*>(scale)[0] = 1.0f;
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SwigluGroupQuantHifp8TilingData*>(tiling);
    tilingData->totalTokens = totalTokens;
    tilingData->dim2H = dim2H;
    tilingData->dimH = dimH;
    tilingData->isGroup = 0;
    tilingData->hasWeight = 1;
    tilingData->hasClamp = 0;
    tilingData->outputOrigin = 1;
    tilingData->clampLimit = 0.0f;
    tilingData->dstTypeMax = dstTypeMax;
    tilingData->tileTokens = tokensPerCore;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->tokensPerCore = tokensPerCore;
    tilingData->groupNum = groupNum;
    tilingData->tileLength = tileLength;

    ICPU_SET_TILING_KEY(tilingKey);
    auto swigluGroupQuantKernel = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR scale, GM_ADDR y,
                                     GM_ADDR yScale, GM_ADDR yOrigin, GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_group_quant(x, weight, groupIndex, scale, y, yScale, yOrigin, workspace, tiling);
    };
    ICPU_RUN_KF(swigluGroupQuantKernel, blockDim, x, weight, nullptr, scale, y, yScale, yOrigin, workspace, tiling);

    // yOrigin excludes weight: yOrigin = silu(gate) * up
    auto* yOriginHalf = reinterpret_cast<half*>(yOrigin);
    for (int64_t t = 0; t < totalTokens; t++) {
        float expected = HostSilu(gateVals[t]) * upVals[t];
        for (int64_t j = 0; j < dimH; j++) {
            EXPECT_NEAR(static_cast<float>(yOriginHalf[t * dimH + j]), expected, 1e-2f);
        }
    }

    // dynamic scale is derived from amax of weight-multiplied values: yScale = amax(w * silu(gate) * up) / dstTypeMax
    if (isDynamic) {
        float amax = 0.0f;
        for (int64_t t = 0; t < totalTokens; t++) {
            amax = std::max(amax, std::fabs(HostSilu(gateVals[t]) * upVals[t] * weightVals[t]));
        }
        EXPECT_NEAR(reinterpret_cast<float*>(yScale)[0], amax / dstTypeMax, 2e-3f);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    if (!isDynamic) {
        AscendC::GmFree(scale);
    }
    AscendC::GmFree(y);
    AscendC::GmFree(yScale);
    AscendC::GmFree(yOrigin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluGroupQuantKernelTest, hifp8_static_y_origin_excludes_weight) { RunHifp8YOriginWeightVerify(4100, false); }

TEST_F(SwigluGroupQuantKernelTest, hifp8_dynamic_y_origin_excludes_weight) { RunHifp8YOriginWeightVerify(4000, true); }
} // namespace
