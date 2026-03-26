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
    tiling->conv2dRunInfo.hin = inputShape[DIM2];
    tiling->conv2dRunInfo.win = inputShape[DIM3];
    tiling->conv2dRunInfo.hout = outputShape[DIM2];
    tiling->conv2dRunInfo.wout = outputShape[DIM3];
    tiling->conv2dRunInfo.batch = inputShape[0];
    tiling->conv2dRunInfo.cin = inputShape[1];
    tiling->conv2dRunInfo.cout = weightShape[0];
    tiling->conv2dRunInfo.kh = weightShape[DIM2];
    tiling->conv2dRunInfo.kw = weightShape[DIM3];
    tiling->conv2dRunInfo.batchDim = 1;
    tiling->conv2dRunInfo.groupDim = 1;
    tiling->conv2dRunInfo.nDim = 1;
    tiling->conv2dRunInfo.hoDim = 1;
    tiling->conv2dRunInfo.woDim = 1;
    tiling->conv2dRunInfo.strideH = strides[0];
    tiling->conv2dRunInfo.strideW = strides[1];
    tiling->conv2dRunInfo.dilationH = dilations[0];
    tiling->conv2dRunInfo.dilationW = dilations[1];
    tiling->conv2dRunInfo.padTop = pads[0];
    tiling->conv2dRunInfo.padLeft = pads[DIM2];
    tiling->conv2dRunInfo.groups = 1;
    tiling->conv2dRunInfo.enlarge = 0;
    tiling->conv2dRunInfo.cinOpt = 0;
    tiling->conv2dRunInfo.coutOpt = 0;
    tiling->conv2dRunInfo.groupOpt = 0;
    tiling->conv2dRunInfo.hasBias = 0;
}

void SetConv2dApiPartOne(Conv2DTilingData* tiling, const TilingInput& tilingInput)
{
    std::vector<uint64_t> inputShape = tilingInput.inputShape;
    std::vector<uint64_t> weightShape = tilingInput.weightShape;
    std::vector<uint64_t> outputShape = tilingInput.outputShape;
    tiling->conv2dApiTiling.orgHi = inputShape[DIM2];
    tiling->conv2dApiTiling.orgWi = inputShape[DIM3];
    tiling->conv2dApiTiling.orgHo = outputShape[DIM2];
    tiling->conv2dApiTiling.orgWo = outputShape[DIM3];
    tiling->conv2dApiTiling.singleCoreBatch = inputShape[0];
    tiling->conv2dApiTiling.singleCoreHo = outputShape[DIM2];
    tiling->conv2dApiTiling.singleCoreWo = outputShape[DIM3];
    tiling->conv2dApiTiling.orgCi = inputShape[1];
    tiling->conv2dApiTiling.orgCo = weightShape[0];
    tiling->conv2dApiTiling.singleCoreCi = inputShape[1];
    tiling->conv2dApiTiling.singleCoreCo = weightShape[0];
    tiling->conv2dApiTiling.hoL1 = NUM_16;
    tiling->conv2dApiTiling.woL1 = 0;
    tiling->conv2dApiTiling.kAL1 = NUM_16;
    tiling->conv2dApiTiling.kBL1 = NUM_16;
    tiling->conv2dApiTiling.nBL1 = NUM_16;
    tiling->conv2dApiTiling.hoL0 = NUM_16;
    tiling->conv2dApiTiling.woL0 = 0;
    tiling->conv2dApiTiling.kL0 = NUM_16;
    tiling->conv2dApiTiling.nL0 = NUM_16;
    tiling->conv2dApiTiling.pBufferFlag = 0;
    tiling->conv2dApiTiling.groups = 1;
    tiling->conv2dApiTiling.enlarge = 0;
    tiling->conv2dApiTiling.singleCoreGroups = 0;
    tiling->conv2dApiTiling.singleCoreGroupOpt = 0;
    tiling->conv2dApiTiling.bUbNStep = 0;
    tiling->conv2dApiTiling.bUbKStep = 0;
    tiling->conv2dApiTiling.orgHixWi = inputShape[DIM2] * inputShape[DIM3];
    tiling->conv2dApiTiling.kernelHxkernelW = weightShape[DIM2] * weightShape[DIM3];
    tiling->conv2dApiTiling.kernelHxkernelWxkernelD = weightShape[DIM2] * weightShape[DIM3];
    tiling->conv2dApiTiling.aL1SpaceSize = SIZE_1K;
    tiling->conv2dApiTiling.multiNBL1 = 1;
    tiling->conv2dApiTiling.cinAInCore = tiling->conv2dApiTiling.kAL1 / tiling->conv2dApiTiling.kernelHxkernelW;
    tiling->conv2dApiTiling.cinATailInCore = tiling->conv2dApiTiling.cinAInCore;
    tiling->conv2dApiTiling.cinBInCore = tiling->conv2dApiTiling.kBL1 / tiling->conv2dApiTiling.kernelHxkernelW;
    tiling->conv2dApiTiling.cinBTailInCore = tiling->conv2dApiTiling.cinBInCore;
}

void SetConv2dApiPartTwo(Conv2DTilingData* tiling, const TilingInput& tilingInput)
{
    std::vector<uint64_t> weightShape = tilingInput.weightShape;
    std::vector<uint64_t> pads = tilingInput.pads;
    std::vector<uint64_t> strides = tilingInput.strides;
    std::vector<uint64_t> dilations = tilingInput.dilations;
    tiling->conv2dApiTiling.mStep = NUM_16;
    tiling->conv2dApiTiling.kStep = 1;
    tiling->conv2dApiTiling.nStep = 1;
    tiling->conv2dApiTiling.fmapKStride = 1;
    tiling->conv2dApiTiling.weightKStride = 1;
    tiling->conv2dApiTiling.cinOffsetBlockInGM = tiling->conv2dApiTiling.kAL1 /
        tiling->conv2dApiTiling.kernelHxkernelW * tiling->conv2dApiTiling.orgHixWi;
    tiling->conv2dApiTiling.coutOffsetBlock = (tiling->conv2dApiTiling.orgCi /
        tiling->conv2dApiTiling.groups) * tiling->conv2dApiTiling.kernelHxkernelW;
    tiling->conv2dApiTiling.nL1DivBlockSize = tiling->conv2dApiTiling.nBL1 / N0;
    tiling->conv2dApiTiling.kernelH = weightShape[DIM2];
    tiling->conv2dApiTiling.kernelW = weightShape[DIM3];
    tiling->conv2dApiTiling.strideH = strides[0];
    tiling->conv2dApiTiling.strideW = strides[1];
    tiling->conv2dApiTiling.dilationH = dilations[0];
    tiling->conv2dApiTiling.dilationW = dilations[1];
    tiling->conv2dApiTiling.padTop = pads[0];
    tiling->conv2dApiTiling.padBottom = pads[1];
    tiling->conv2dApiTiling.padLeft = pads[DIM2];
    tiling->conv2dApiTiling.padRight = pads[DIM3];
    tiling->conv2dApiTiling.iterateMNOrder = 0;
    tiling->conv2dApiTiling.biasFullLoadFlag = 1;
    tiling->conv2dApiTiling.fixpParamsFullLoadFlag = 1;
    tiling->conv2dApiTiling.hf32Enable = 0;
    tiling->conv2dApiTiling.hf32TransMode = 0;
    tiling->conv2dApiTiling.hasBias = 0;
    tiling->conv2dApiTiling.hasScale = 0;
    tiling->conv2dApiTiling.dualOutput = 0;
    tiling->conv2dApiTiling.quantMode0 = 0;
    tiling->conv2dApiTiling.reluMode0 = 0;
    tiling->conv2dApiTiling.clipMode0 = 0;
    tiling->conv2dApiTiling.quantMode1 = 0;
    tiling->conv2dApiTiling.reluMode1 = 0;
    tiling->conv2dApiTiling.clipMode1 = 0;
    tiling->conv2dApiTiling.offsetx = 0;
    tiling->conv2dApiTiling.roundMode = 0;
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

    auto conv2dv2_func = [](GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w,
        GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
            conv2dv2<0, 0, 0, 0, 0, 0, 0, 0, 0>(x, filter, bias, offset_w, y, workspace, tiling);
    };
    ICPU_RUN_KF(conv2dv2_func, numBlocks, input, weight, nullptr, nullptr, output, workspace, tiling);

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

    TestSimpleKernel(inputShape, weightShape);
}