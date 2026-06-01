/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "conv2d_v2_tiling_def.h"

constexpr uint8_t FP16_SIZE = 2;
constexpr uint8_t N0 = 16;
constexpr uint32_t DIM2 = 2;
constexpr uint32_t DIM3 = 3;
constexpr uint32_t SIZE_1K = 1024;
constexpr uint32_t NUM_16 = 16;

extern "C" __global__ __aicore__ void conv2dv2(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w,
                                               GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class Conv2DV2KernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "Conv2DV2KernelTest SetUp." << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "Conv2DV2KernelTest TearDown." << std::endl;
    }
};
namespace {
struct ConvShape {
    uint64_t shape;
    uint64_t pad1;
    uint64_t pad2;
    uint64_t dilation;
    uint64_t stride;
    uint64_t kernelSize;
};

struct TilingInput {
    std::vector<uint64_t> inputShape;
    std::vector<uint64_t> weightShape;
    std::vector<uint64_t> outputShape;
    std::vector<uint64_t> pads;
    std::vector<uint64_t> strides;
    std::vector<uint64_t> dilations;
};

uint64_t CalcOutputShape(const ConvShape& convShape)
{
    return ((convShape.shape + convShape.pad1 + convShape.pad2 - convShape.dilation * (convShape.kernelSize - 1) - 1) /
        convShape.stride + 1);
}

size_t VectorReduceMul(const std::vector<uint64_t>& vec)
{
    return std::accumulate(std::begin(vec), std::end(vec), 1, std::multiplies<size_t>());
}

void SetConv2dRunInfo(Conv2DTilingData* tiling, const TilingInput& tilingInput)
{
    std::vector<uint64_t> inputShape = tilingInput.inputShape;
    std::vector<uint64_t> weightShape = tilingInput.weightShape;
    std::vector<uint64_t> outputShape = tilingInput.outputShape;
    std::vector<uint64_t> pads = tilingInput.pads;
    std::vector<uint64_t> strides = tilingInput.strides;
    std::vector<uint64_t> dilations = tilingInput.dilations;
    tiling->hin = inputShape[DIM2];
    tiling->win = inputShape[DIM3];
    tiling->hout = outputShape[DIM2];
    tiling->wout = outputShape[DIM3];
    tiling->batch = inputShape[0];
    tiling->cin = inputShape[1];
    tiling->cout = weightShape[0];
    tiling->kh = weightShape[DIM2];
    tiling->kw = weightShape[DIM3];
    tiling->batchDim = 1;
    tiling->groupDim = 1;
    tiling->nDim = 1;
    tiling->hoDim = 1;
    tiling->woDim = 1;
    tiling->strideH = strides[0];
    tiling->strideW = strides[1];
    tiling->dilationH = dilations[0];
    tiling->dilationW = dilations[1];
    tiling->padTop = pads[0];
    tiling->padLeft = pads[DIM2];
    tiling->groups = 1;
    tiling->enlarge = 0;
    tiling->cinOpt = 0;
    tiling->coutOpt = 0;
    tiling->groupOpt = 0;
    tiling->hasBias = 0;
}

void SetConv2dApiPartOne(Conv2DTilingData* tiling, const TilingInput& tilingInput)
{
    std::vector<uint64_t> inputShape = tilingInput.inputShape;
    std::vector<uint64_t> weightShape = tilingInput.weightShape;
    std::vector<uint64_t> outputShape = tilingInput.outputShape;
    tiling->orgHi = inputShape[DIM2];
    tiling->orgWi = inputShape[DIM3];
    tiling->orgHo = outputShape[DIM2];
    tiling->orgWo = outputShape[DIM3];
    tiling->singleCoreBatch = inputShape[0];
    tiling->singleCoreHo = outputShape[DIM2];
    tiling->singleCoreWo = outputShape[DIM3];
    tiling->orgCi = inputShape[1];
    tiling->orgCo = weightShape[0];
    tiling->singleCoreCi = inputShape[1];
    tiling->singleCoreCo = weightShape[0];
    tiling->hoL1 = NUM_16;
    tiling->woL1 = 0;
    tiling->kAL1 = NUM_16;
    tiling->kBL1 = NUM_16;
    tiling->nBL1 = NUM_16;
    tiling->hoL0 = NUM_16;
    tiling->woL0 = 0;
    tiling->kL0 = NUM_16;
    tiling->nL0 = NUM_16;
    tiling->pBufferFlag = 0;
    tiling->groups = 1;
    tiling->enlarge = 0;
    tiling->singleCoreGroups = 0;
    tiling->singleCoreGroupOpt = 0;
    tiling->bUbNStep = 0;
    tiling->bUbKStep = 0;
    tiling->orgHixWi = inputShape[DIM2] * inputShape[DIM3];
    tiling->kernelHxkernelW = weightShape[DIM2] * weightShape[DIM3];
    tiling->kernelHxkernelWxkernelD = weightShape[DIM2] * weightShape[DIM3];
    tiling->aL1SpaceSize = SIZE_1K;
    tiling->multiNBL1 = 1;
    tiling->cinAInCore = tiling->kAL1 / tiling->kernelHxkernelW;
    tiling->cinATailInCore = tiling->cinAInCore;
    tiling->cinBInCore = tiling->kBL1 / tiling->kernelHxkernelW;
    tiling->cinBTailInCore = tiling->cinBInCore;
}

void SetConv2dApiPartTwo(Conv2DTilingData* tiling, const TilingInput& tilingInput)
{
    std::vector<uint64_t> weightShape = tilingInput.weightShape;
    std::vector<uint64_t> pads = tilingInput.pads;
    std::vector<uint64_t> strides = tilingInput.strides;
    std::vector<uint64_t> dilations = tilingInput.dilations;
    tiling->mStep = NUM_16;
    tiling->kStep = 1;
    tiling->nStep = 1;
    tiling->fmapKStride = 1;
    tiling->weightKStride = 1;
    tiling->cinOffsetBlockInGM = tiling->kAL1 /
        tiling->kernelHxkernelW * tiling->orgHixWi;
    tiling->coutOffsetBlock = (tiling->orgCi /
        tiling->groups) * tiling->kernelHxkernelW;
    tiling->nL1DivBlockSize = tiling->nBL1 / N0;
    tiling->kernelH = weightShape[DIM2];
    tiling->kernelW = weightShape[DIM3];
    tiling->strideH = strides[0];
    tiling->strideW = strides[1];
    tiling->dilationH = dilations[0];
    tiling->dilationW = dilations[1];
    tiling->padTop = pads[0];
    tiling->padBottom = pads[1];
    tiling->padLeft = pads[DIM2];
    tiling->padRight = pads[DIM3];
    tiling->iterateMNOrder = 0;
    tiling->biasFullLoadFlag = 1;
    tiling->fixpParamsFullLoadFlag = 1;
    tiling->hf32Enable = 0;
    tiling->hf32TransMode = 0;
    tiling->hasBias = 0;
    tiling->hasScale = 0;
    tiling->dualOutput = 0;
    tiling->quantMode0 = 0;
    tiling->reluMode0 = 0;
    tiling->clipMode0 = 0;
    tiling->quantMode1 = 0;
    tiling->reluMode1 = 0;
    tiling->clipMode1 = 0;
    tiling->offsetx = 0;
    tiling->roundMode = 0;
}

void SetTilingData(Conv2DTilingData* tiling, const TilingInput& tilingInput)
{
    SetConv2dRunInfo(tiling, tilingInput);
    SetConv2dApiPartOne(tiling, tilingInput);
    SetConv2dApiPartTwo(tiling, tilingInput);
}

void TestSimpleKernel(const std::vector<uint64_t>& inputShape, const std::vector<uint64_t>& weightShape)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const uint64_t numBlocks = 1;
    const uint64_t groups = 1;
    std::vector<uint64_t> pads = {0, 0, 0, 0};
    std::vector<uint64_t> strides = {1, 1};
    std::vector<uint64_t> dilations = {1, 1};
    ConvShape convShapeH =
        {inputShape[DIM2], pads[0], pads[1], dilations[0], strides[0], weightShape[DIM2]};
    ConvShape convShapeW =
        {inputShape[DIM3], pads[DIM2], pads[DIM3], dilations[1], strides[1], weightShape[DIM3]};
    uint64_t ho = CalcOutputShape(convShapeH);
    uint64_t wo = CalcOutputShape(convShapeW);
    std::vector<uint64_t> outputShape = {inputShape[0], weightShape[0], ho, wo};

    size_t inputBtyes = VectorReduceMul(inputShape) * FP16_SIZE;
    size_t weightBytes = VectorReduceMul(weightShape) * FP16_SIZE;
    size_t outputBytes = VectorReduceMul(outputShape) * FP16_SIZE;
    size_t workspaceSize = SIZE_1K * SIZE_1K * NUM_16;
    size_t tilingDataSize = sizeof(Conv2DTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputBtyes);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightBytes);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(SIZE_1K * SIZE_1K * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(Conv2DTilingData));

    memset(workspace, 0, workspaceSize);

    Conv2DTilingData* tilingData = reinterpret_cast<Conv2DTilingData*>(tiling);
    TilingInput tilingInput = {inputShape, weightShape, outputShape, pads, strides, dilations};
    SetTilingData(tilingData, tilingInput);

    // auto conv2dv2_func = [](GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w,
    //     GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    //         conv2dv2<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(x, filter, bias, offset_w, y, workspace, tiling);
    // };
    // ICPU_RUN_KF(conv2dv2_func, numBlocks, input, weight, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace
TEST_F(Conv2DV2KernelTest, conv2dv2_kernel_test_base)
{
    std::vector<uint64_t> inputShape = {1, 1, 1, 1};
    std::vector<uint64_t> weightShape = {1, 1, 1, 1};

    //TestSimpleKernel(inputShape, weightShape);
}