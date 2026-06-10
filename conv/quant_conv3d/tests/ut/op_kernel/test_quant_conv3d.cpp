/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <iostream>
#include <vector>
#include <numeric>
#include <functional>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "version/asc_devkit_version.h"
#include "quant_conv3d_tiling_def.h"

#ifndef CONV_KERNEL
#include "quant_conv3d/quant_conv3d.cpp"
#define CONV_KERNEL
#endif

constexpr uint8_t FP16_SIZE = 2;
constexpr uint32_t DIM1 = 1;
constexpr uint32_t DIM2 = 2;
constexpr uint32_t DIM3 = 3;
constexpr uint32_t DIM4 = 4;
constexpr uint32_t SIZE_1K = 1024;
constexpr uint32_t NUM_16 = 16;

using Ops::NN::Conv3dV2::Conv3DV2TilingDataV2;

class QuantConv3DKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "QuantConv3DKernelTest SetUp." << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "QuantConv3DKernelTest TearDown." << std::endl;
    }
};
namespace {
struct Conv3DShape {
    uint64_t shape;
    uint64_t pad1;
    uint64_t pad2;
    uint64_t dilation;
    uint64_t stride;
    uint64_t kernelSize;
};

uint64_t CalcOutputShape(const Conv3DShape& convShape)
{
    return ((convShape.shape + convShape.pad1 + convShape.pad2 - convShape.dilation * (convShape.kernelSize - 1) - 1) /
        convShape.stride + 1);
}

size_t VectorReduceMul(const std::vector<uint64_t>& vec)
{
    return std::accumulate(std::begin(vec), std::end(vec), 1, std::multiplies<size_t>());
}

void SetQuantConv3dTiling(Conv3DV2TilingDataV2* tiling, const std::vector<uint64_t>& inputShape,
    const std::vector<uint64_t>& weightShape, const std::vector<uint64_t>& outputShape)
{
    tiling->din = inputShape[DIM2];
    tiling->hin = inputShape[DIM3];
    tiling->win = inputShape[DIM4];
    tiling->dout = outputShape[DIM2];
    tiling->hout = outputShape[DIM3];
    tiling->wout = outputShape[DIM4];
    tiling->batch = inputShape[0];
    tiling->cin = inputShape[1];
    tiling->cout = weightShape[0];
    tiling->kd = weightShape[DIM2];
    tiling->kh = weightShape[DIM3];
    tiling->kw = weightShape[DIM4];

    tiling->orgDi = inputShape[DIM2];
    tiling->orgHi = inputShape[DIM3];
    tiling->orgWi = inputShape[DIM4];
    tiling->orgDo = outputShape[DIM2];
    tiling->orgHo = outputShape[DIM3];
    tiling->orgWo = outputShape[DIM4];
    tiling->orgHixWi = inputShape[DIM3] * inputShape[DIM4];
    tiling->orgHoxWo = outputShape[DIM3] * outputShape[DIM4];

    tiling->singleCoreBatch = inputShape[0];
    tiling->singleCoreDo = outputShape[DIM2];
    tiling->singleCoreM = outputShape[DIM2] * outputShape[DIM3] * outputShape[DIM4];
    tiling->singleCoreHo = outputShape[DIM3];
    tiling->singleCoreWo = 0;

    tiling->orgCi = inputShape[1];
    tiling->orgCo = weightShape[0];
    tiling->singleCoreCi = inputShape[1];
    tiling->singleCoreCo = weightShape[0];
    tiling->kernelD = weightShape[DIM2];
    tiling->kernelH = weightShape[DIM3];
    tiling->kernelW = weightShape[DIM4];
    tiling->kernelHxkernelW = weightShape[DIM3] * weightShape[DIM4];
    tiling->kernelHxkernelWxkernelD = weightShape[DIM2] * weightShape[DIM3] * weightShape[DIM4];

    tiling->batchDim = 1;
    tiling->doDim = 1;
    tiling->mDim = 1;
    tiling->wDim = 1;
    tiling->nDim = 1;
    tiling->groupDim = 1;
    tiling->hoDim = 1;

    tiling->strideD = 1;
    tiling->strideH = 1;
    tiling->strideW = 1;
    tiling->dilationD = 1;
    tiling->dilationH = 1;
    tiling->dilationW = 1;
    tiling->padHead = 0;
    tiling->padTail = 0;
    tiling->padTop = 0;
    tiling->padBottom = 0;
    tiling->padLeft = 0;
    tiling->padRight = 0;

    tiling->groups = 1;
    tiling->enlarge = 0;
    tiling->singleCoreGroups = 0;
    tiling->singleCoreGroupOpt = 0;
    tiling->groupOpt = 0;
    tiling->cinOpt = 0;
    tiling->coutOpt = 0;

    uint32_t ci = inputShape[1];
    uint32_t kernelVol = weightShape[DIM2] * weightShape[DIM3] * weightShape[DIM4];
    uint32_t kL1 = ci * kernelVol;
    tiling->kAL1 = kL1;
    tiling->kBL1 = kL1;
    tiling->kAL1Tail = 0;
    tiling->kBL1Tail = 0;
    tiling->nBL1 = NUM_16;

    tiling->hoL1 = NUM_16;
    tiling->hoL0 = NUM_16;
    tiling->woL1 = 0;
    tiling->woL0 = 0;
    tiling->kL0 = NUM_16;
    tiling->nL0 = NUM_16;
    tiling->mL0 = NUM_16;

    tiling->mAL1 = NUM_16;
    tiling->pBufferFlag = 0;
    tiling->fmapKStride = 1;
    tiling->weightKStride = 1;
    tiling->cinOffsetBlockInGM = tiling->kAL1 / kernelVol * inputShape[DIM3] * inputShape[DIM4];
    tiling->coutOffsetBlock = ci * kernelVol;
    tiling->nL1DivBlockSize = tiling->nBL1 / NUM_16;
    tiling->mStep = NUM_16;
    tiling->kStep = 1;
    tiling->nStep = 1;

    tiling->cinAInCore = tiling->kAL1 / kernelVol;
    tiling->cinATailInCore = tiling->cinAInCore;
    tiling->cinBInCore = tiling->kBL1 / kernelVol;
    tiling->cinBTailInCore = tiling->cinBInCore;
    tiling->cin1InAL1 = tiling->kAL1 / kernelVol;
    tiling->cin1InAL1Tail = tiling->cin1InAL1;
    tiling->aL1SpaceSize = tiling->cinAInCore * tiling->orgHixWi * FP16_SIZE;
    tiling->multiNBL1 = 1;

    tiling->nL0xk0 = NUM_16;
    tiling->nBL1DivnL0 = 1;
    tiling->mAL1DivmL0 = 1;
    tiling->KBL1Divk0 = tiling->kBL1 / NUM_16;
    tiling->KBL1TailDivk0 = 0;

    tiling->oriWixk0 = inputShape[DIM4];
    tiling->oriHixOriWixk0 = inputShape[DIM3] * inputShape[DIM4];
    tiling->cin1xOriHixOriWixk0 = tiling->cin1InAL1 * inputShape[DIM3] * inputShape[DIM4];
    tiling->kL0xorgCoAlignN0 = NUM_16;

    tiling->biasFullLoadFlag = 1;
    tiling->fixpParamsFullLoadFlag = 1;
    tiling->hasBias = 0;
    tiling->hasScale = 0;
    tiling->iterateMNOrder = 0;
    tiling->hf32Enable = 0;
    tiling->hf32TransMode = 0;
    tiling->outputOrder = 1;
    tiling->offsetx = 0;
    tiling->roundMode = 0;

    tiling->mUB = 0;
    tiling->nUB = 0;
    tiling->scaleAndBiasLoadType = 0;
    tiling->workspaceSize = 0;
}

void SetUnionDataXt(Conv3DV2TilingDataV2* tiling)
{
    uint64_t xt = 0;
    xt |= static_cast<uint64_t>(tiling->strideW & 0x3f);
    xt |= (static_cast<uint64_t>(tiling->strideH & 0x3f) << 6);
    xt |= (static_cast<uint64_t>(tiling->kw & 0xff) << 12);
    xt |= (static_cast<uint64_t>(tiling->kh & 0xff) << 20);
    xt |= (static_cast<uint64_t>(tiling->dilationW & 0xff) << 28);
    xt |= (static_cast<uint64_t>(tiling->dilationH & 0xff) << 36);
    xt |= ((static_cast<uint64_t>(tiling->kw) & 0x100) >> 8) << 44;
    xt |= ((static_cast<uint64_t>(tiling->kh) & 0x100) >> 8) << 45;
    tiling->unionDataXt = xt;
}

void TestSimpleKernel(const std::vector<uint64_t>& inputShape, const std::vector<uint64_t>& weightShape)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const uint64_t numBlocks = 1;
    Conv3DShape convShapeD =
        {inputShape[DIM2], static_cast<uint64_t>(0), static_cast<uint64_t>(0), static_cast<uint64_t>(1),
         static_cast<uint64_t>(1), weightShape[DIM2]};
    Conv3DShape convShapeH =
        {inputShape[DIM3], static_cast<uint64_t>(0), static_cast<uint64_t>(0), static_cast<uint64_t>(1),
         static_cast<uint64_t>(1), weightShape[DIM3]};
    Conv3DShape convShapeW =
        {inputShape[DIM4], static_cast<uint64_t>(0), static_cast<uint64_t>(0), static_cast<uint64_t>(1),
         static_cast<uint64_t>(1), weightShape[DIM4]};
    uint64_t dout = CalcOutputShape(convShapeD);
    uint64_t hout = CalcOutputShape(convShapeH);
    uint64_t wout = CalcOutputShape(convShapeW);
    std::vector<uint64_t> outputShape = {inputShape[0], weightShape[0], dout, hout, wout};

    size_t inputBtyes = VectorReduceMul(inputShape) * FP16_SIZE;
    size_t weightBytes = VectorReduceMul(weightShape) * FP16_SIZE;
    size_t outputBytes = VectorReduceMul(outputShape) * FP16_SIZE;
    size_t workspaceSize = SIZE_1K * SIZE_1K * NUM_16;
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputBtyes);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightBytes);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(Conv3DV2TilingDataV2));

    memset(workspace, 0, workspaceSize);

    Conv3DV2TilingDataV2* tilingData = reinterpret_cast<Conv3DV2TilingDataV2*>(tiling);
    SetQuantConv3dTiling(tilingData, inputShape, weightShape, outputShape);
    SetUnionDataXt(tilingData);

    auto quant_conv3d_func = [](GM_ADDR x, GM_ADDR filter, GM_ADDR scale, GM_ADDR bias, GM_ADDR offset,
        GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::quant_conv3d<0, 0, 0, 0, 1, 0, 0>(
            x, filter, scale, bias, offset, y, workspace, tiling);
    };
    ICPU_RUN_KF(quant_conv3d_func, numBlocks, input, weight, nullptr, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace
TEST_F(QuantConv3DKernelTest, quant_conv3d_kernel_test_base)
{
    std::vector<uint64_t> inputShape = {1, 16, 4, 4, 4};
    std::vector<uint64_t> weightShape = {16, 16, 1, 1, 1};

    TestSimpleKernel(inputShape, weightShape);
}