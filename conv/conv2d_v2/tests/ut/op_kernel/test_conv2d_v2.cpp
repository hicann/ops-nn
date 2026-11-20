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
#include <vector>
#include <numeric>
#include <functional>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "version/asc_devkit_version.h"
#include "conv2d_v2_tiling_def.h"

#ifndef CONV_KERNEL
#include "conv2d_v2/conv2d_v2.cpp"
#define CONV_KERNEL
#endif

constexpr uint8_t FP16_SIZE = 2;
constexpr uint32_t DIM2 = 2;
constexpr uint32_t DIM3 = 3;
constexpr uint32_t SIZE_1K = 1024;
constexpr uint32_t NUM_16 = 16;

class Conv2DV2KernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
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
                convShape.stride +
            1);
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
    // M_MODE: singleCoreHo stores flattened M (Ho * Wo) per core
    tiling->singleCoreHo = outputShape[DIM2] * outputShape[DIM3];
    tiling->singleCoreWo = outputShape[DIM3];
    tiling->orgCi = inputShape[1];
    tiling->orgCo = weightShape[0];
    tiling->singleCoreCi = inputShape[1];
    tiling->singleCoreCo = weightShape[0];
    tiling->hoL1 = NUM_16;
    tiling->woL1 = 0;
    tiling->orgHixWi = inputShape[DIM2] * inputShape[DIM3];
    tiling->kernelHxkernelW = weightShape[DIM2] * weightShape[DIM3];
    tiling->kernelHxkernelWxkernelD = weightShape[DIM2] * weightShape[DIM3];
    uint64_t kL1 = tiling->singleCoreCi * tiling->kernelHxkernelW;
    tiling->kAL1 = kL1;
    tiling->kBL1 = kL1;
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
    tiling->multiNBL1 = 1;
    tiling->cinAInCore = tiling->kAL1 / tiling->kernelHxkernelW;
    tiling->cinATailInCore = tiling->cinAInCore;
    tiling->cinBInCore = tiling->kBL1 / tiling->kernelHxkernelW;
    tiling->cinBTailInCore = tiling->cinBInCore;
    tiling->aL1SpaceSize = tiling->cinAInCore * tiling->orgHixWi * FP16_SIZE;
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
    tiling->cinOffsetBlockInGM = tiling->kAL1 / tiling->kernelHxkernelW * tiling->orgHixWi;
    tiling->coutOffsetBlock = (tiling->orgCi / tiling->groups) * tiling->kernelHxkernelW;
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

void SetUnionDataXt(Conv2DTilingData* tiling)
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

void SetTilingData(Conv2DTilingData* tiling, const TilingInput& tilingInput)
{
    SetConv2dRunInfo(tiling, tilingInput);
    SetConv2dApiPartOne(tiling, tilingInput);
    SetConv2dApiPartTwo(tiling, tilingInput);
    SetUnionDataXt(tiling);
}

void TestSimpleKernel(const std::vector<uint64_t>& inputShape, const std::vector<uint64_t>& weightShape)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const uint64_t numBlocks = 1;
    std::vector<uint64_t> pads = {0, 0, 0, 0};
    std::vector<uint64_t> strides = {1, 1};
    std::vector<uint64_t> dilations = {1, 1};
    ConvShape convShapeH = {inputShape[DIM2], pads[0], pads[1], dilations[0], strides[0], weightShape[DIM2]};
    ConvShape convShapeW = {inputShape[DIM3], pads[DIM2], pads[DIM3], dilations[1], strides[1], weightShape[DIM3]};
    uint64_t ho = CalcOutputShape(convShapeH);
    uint64_t wo = CalcOutputShape(convShapeW);
    std::vector<uint64_t> outputShape = {inputShape[0], weightShape[0], ho, wo};

    size_t inputBtyes = VectorReduceMul(inputShape) * FP16_SIZE;
    size_t weightBytes = VectorReduceMul(weightShape) * FP16_SIZE;
    size_t outputBytes = VectorReduceMul(outputShape) * FP16_SIZE;
    size_t workspaceSize = SIZE_1K * SIZE_1K * NUM_16;
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputBtyes);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightBytes);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(Conv2DTilingData));

    memset(workspace, 0, workspaceSize);
    memset(tiling, 0, sizeof(Conv2DTilingData));

    Conv2DTilingData* tilingData = reinterpret_cast<Conv2DTilingData*>(tiling);
    TilingInput tilingInput = {inputShape, weightShape, outputShape, pads, strides, dilations};
    SetTilingData(tilingData, tilingInput);

    auto conv2dv2_func = [](GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace,
                            GM_ADDR tiling) {
        // OutputOrder=1 (M_MODE): tiling uses hoL1>0 && woL1==0
        ::conv2dv2<0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(x, filter, bias, offset_w, y, workspace, tiling);
    };
    ICPU_RUN_KF(conv2dv2_func, numBlocks, input, weight, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

void TestStrideWriteKernel(const std::vector<uint64_t>& inputShape, const std::vector<uint64_t>& weightShape,
                           const uint64_t totalShapeCout)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const uint64_t numBlocks = 1;
    std::vector<uint64_t> pads = {0, 0, 0, 0};
    std::vector<uint64_t> strides = {1, 1};
    std::vector<uint64_t> dilations = {1, 1};
    ConvShape convShapeH = {inputShape[DIM2], pads[0], pads[1], dilations[0], strides[0], weightShape[DIM2]};
    ConvShape convShapeW = {inputShape[DIM3], pads[DIM2], pads[DIM3], dilations[1], strides[1], weightShape[DIM3]};
    uint64_t ho = CalcOutputShape(convShapeH);
    uint64_t wo = CalcOutputShape(convShapeW);
    std::vector<uint64_t> outputShape = {inputShape[0], weightShape[0], ho, wo};
    std::vector<uint64_t> totalOutputShape = {inputShape[0], totalShapeCout, ho, wo};

    size_t inputBtyes = VectorReduceMul(inputShape) * FP16_SIZE;
    size_t weightBytes = VectorReduceMul(weightShape) * FP16_SIZE;
    size_t outputBytes = VectorReduceMul(totalOutputShape) * FP16_SIZE;
    size_t workspaceSize = SIZE_1K * SIZE_1K * NUM_16;
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputBtyes);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightBytes);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(Conv2DTilingData));

    memset(workspace, 0, workspaceSize);
    memset(tiling, 0, sizeof(Conv2DTilingData));
    memset(output, 0xAA, outputBytes);

    Conv2DTilingData* tilingData = reinterpret_cast<Conv2DTilingData*>(tiling);
    TilingInput tilingInput = {inputShape, weightShape, outputShape, pads, strides, dilations};
    SetTilingData(tilingData, tilingInput);
    tilingData->total_shape_cout = totalShapeCout;
    tilingData->strideWriteDstStride = ho * wo * totalShapeCout;

    auto conv2dv2_func = [](GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace,
                            GM_ADDR tiling) {
        ::conv2dv2<0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(x, filter, bias, offset_w, y, workspace, tiling);
    };
    ICPU_RUN_KF(conv2dv2_func, numBlocks, input, weight, nullptr, nullptr, output, workspace, tiling);

    const size_t planeBytes = ho * wo * FP16_SIZE;
    for (uint64_t b = 0; b < inputShape[0]; b++) {
        for (uint64_t c = 0; c < totalShapeCout; c++) {
            const uint8_t* plane = output + static_cast<size_t>(b * totalShapeCout + c) * planeBytes;
            bool anyWritten = false;
            for (size_t i = 0; i < planeBytes; i++) {
                if (plane[i] != 0xAA) {
                    anyWritten = true;
                    break;
                }
            }
            EXPECT_EQ(anyWritten, c < weightShape[0]);
        }
    }

    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace
TEST_F(Conv2DV2KernelTest, conv2dv2_kernel_test_base)
{
    std::vector<uint64_t> inputShape = {1, 16, 16, 16};
    std::vector<uint64_t> weightShape = {16, 16, 1, 1};

    TestSimpleKernel(inputShape, weightShape);
}

TEST_F(Conv2DV2KernelTest, conv2dv2_kernel_test_stridewrite_nchw)
{
    std::vector<uint64_t> inputShape = {2, 16, 2, 2};
    std::vector<uint64_t> weightShape = {16, 16, 1, 1};

    TestStrideWriteKernel(inputShape, weightShape, 32);
}

namespace {
void TestInnerBatchKernel(const std::vector<uint64_t>& inputShape, const std::vector<uint64_t>& weightShape,
                          uint64_t innerBatch, uint64_t hoL0)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const uint64_t numBlocks = 1;
    std::vector<uint64_t> pads = {0, 0, 0, 0};
    std::vector<uint64_t> strides = {1, 1};
    std::vector<uint64_t> dilations = {1, 1};
    ConvShape convShapeH = {inputShape[DIM2], pads[0], pads[1], dilations[0], strides[0], weightShape[DIM2]};
    ConvShape convShapeW = {inputShape[DIM3], pads[DIM2], pads[DIM3], dilations[1], strides[1], weightShape[DIM3]};
    uint64_t ho = CalcOutputShape(convShapeH);
    uint64_t wo = CalcOutputShape(convShapeW);
    std::vector<uint64_t> outputShape = {inputShape[0], weightShape[0], ho, wo};

    size_t inputBtyes = VectorReduceMul(inputShape) * FP16_SIZE;
    size_t weightBytes = VectorReduceMul(weightShape) * FP16_SIZE;
    size_t outputBytes = VectorReduceMul(outputShape) * FP16_SIZE;
    size_t workspaceSize = SIZE_1K * SIZE_1K * NUM_16;
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputBtyes);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightBytes);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(Conv2DTilingData));

    memset(workspace, 0, workspaceSize);
    memset(tiling, 0, sizeof(Conv2DTilingData));
    memset(output, 0xAA, outputBytes);

    Conv2DTilingData* tilingData = reinterpret_cast<Conv2DTilingData*>(tiling);
    TilingInput tilingInput = {inputShape, weightShape, outputShape, pads, strides, dilations};
    SetTilingData(tilingData, tilingInput);
    tilingData->innerBatch = innerBatch;
    tilingData->hoL0 = hoL0;

    auto conv2dv2_func = [](GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace,
                            GM_ADDR tiling) {
        // OutputOrder=1 (M_MODE), InnerBatch=1 (KERNEL_1X1_MULTI), SmallKernel=1
        ::conv2dv2<0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0>(x, filter, bias, offset_w, y, workspace, tiling);
    };
    ICPU_RUN_KF(conv2dv2_func, numBlocks, input, weight, nullptr, nullptr, output, workspace, tiling);

    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

TEST_F(Conv2DV2KernelTest, conv2dv2_kernel_test_innerbatch_1x1_multichunk)
{
    std::vector<uint64_t> inputShape = {32, 256, 7, 7};
    std::vector<uint64_t> weightShape = {128, 256, 1, 1};
    TestInnerBatchKernel(inputShape, weightShape, 16, 16);
}

TEST_F(Conv2DV2KernelTest, conv2dv2_kernel_test_innerbatch_1x1_singlechunk)
{
    std::vector<uint64_t> inputShape = {32, 256, 7, 7};
    std::vector<uint64_t> weightShape = {128, 256, 1, 1};
    TestInnerBatchKernel(inputShape, weightShape, 16, 49);
}

TEST_F(Conv2DV2KernelTest, conv2dv2_kernel_test_innerbatch_1x1_group_doublebuffer)
{
    // 2 groups x ib=2 (12.5KB per group): the two-group AL1 layout fits L1, so
    // the next group's fmap is prefetched into the alternate buffer while the
    // current group computes.
    std::vector<uint64_t> inputShape = {4, 64, 7, 7};
    std::vector<uint64_t> weightShape = {128, 64, 1, 1};
    TestInnerBatchKernel(inputShape, weightShape, 2, 49);
}

namespace {
// Board-captured tiling for the failing ST case (x 32,64,7,7 / w 128,64,1,1):
// innerBatch=8 singleCoreBatch=8 singleCoreHo=64 hoL0=64 kAL1=kBL1=64 kL0=32
// nL0=128 pBufferFlag=3 multiNBL1=1 mStep=64 nStep=8 kStep=2, with bias.
void TestInnerBatchBoardCase()
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const uint64_t numBlocks = 4; // batchDim: 32 maps / 8 per core
    std::vector<uint64_t> inputShape = {32, 64, 7, 7};
    std::vector<uint64_t> weightShape = {128, 64, 1, 1};
    std::vector<uint64_t> outputShape = {32, 128, 7, 7};
    size_t inputBtyes = VectorReduceMul(inputShape) * FP16_SIZE;
    size_t weightBytes = VectorReduceMul(weightShape) * FP16_SIZE;
    size_t biasBytes = weightShape[0] * FP16_SIZE;
    size_t outputBytes = VectorReduceMul(outputShape) * FP16_SIZE;
    size_t workspaceSize = SIZE_1K * SIZE_1K * NUM_16;
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputBtyes);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightBytes);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasBytes);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(Conv2DTilingData));

    memset(workspace, 0, workspaceSize);
    memset(tiling, 0, sizeof(Conv2DTilingData));
    memset(output, 0xAA, outputBytes);

    Conv2DTilingData* tilingData = reinterpret_cast<Conv2DTilingData*>(tiling);
    TilingInput tilingInput = {inputShape, weightShape, outputShape, {0, 0, 0, 0}, {1, 1}, {1, 1}};
    SetTilingData(tilingData, tilingInput);
    tilingData->innerBatch = 8;
    tilingData->singleCoreBatch = 8;
    tilingData->singleCoreHo = 64;
    tilingData->hoL1 = 64;
    tilingData->hoL0 = 64;
    tilingData->kAL1 = 64;
    tilingData->kBL1 = 64;
    tilingData->kL0 = 32;
    tilingData->nL0 = 128;
    tilingData->nBL1 = 128;
    tilingData->pBufferFlag = 3;
    tilingData->multiNBL1 = 1;
    tilingData->mStep = 64;
    tilingData->nStep = 8;
    tilingData->kStep = 2;
    tilingData->hasBias = 1;
    tilingData->cinAInCore = 64;
    tilingData->cinATailInCore = 64;
    tilingData->cinBInCore = 64;
    tilingData->cinBTailInCore = 64;
    tilingData->aL1SpaceSize = 64 * 49 * FP16_SIZE;

    auto conv2dv2_func = [](GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace,
                            GM_ADDR tiling) {
        ::conv2dv2<0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0>(x, filter, bias, offset_w, y, workspace, tiling);
    };
    ICPU_RUN_KF(conv2dv2_func, numBlocks, input, weight, bias, nullptr, output, workspace, tiling);

    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(bias);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

TEST_F(Conv2DV2KernelTest, conv2dv2_kernel_test_innerbatch_board_case) { TestInnerBatchBoardCase(); }

namespace {
// MULTI mode (SEL InnerBatch=2): compile+run cover for the per-map L0A load /
// per-map Mmad loop instantiation, which the 1x1 SEL never compiles.
void TestInnerBatchMultiKernel()
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const uint64_t numBlocks = 1;
    std::vector<uint64_t> inputShape = {8, 64, 7, 7};
    std::vector<uint64_t> weightShape = {64, 64, 3, 3};
    std::vector<uint64_t> pads = {1, 1, 1, 1};
    std::vector<uint64_t> strides = {1, 1};
    std::vector<uint64_t> dilations = {1, 1};
    ConvShape convShapeH = {inputShape[DIM2], pads[0], pads[1], dilations[0], strides[0], weightShape[DIM2]};
    ConvShape convShapeW = {inputShape[DIM3], pads[DIM2], pads[DIM3], dilations[1], strides[1], weightShape[DIM3]};
    uint64_t ho = CalcOutputShape(convShapeH);
    uint64_t wo = CalcOutputShape(convShapeW);
    std::vector<uint64_t> outputShape = {inputShape[0], weightShape[0], ho, wo};

    size_t inputBtyes = VectorReduceMul(inputShape) * FP16_SIZE;
    size_t weightBytes = VectorReduceMul(weightShape) * FP16_SIZE;
    size_t outputBytes = VectorReduceMul(outputShape) * FP16_SIZE;
    size_t workspaceSize = SIZE_1K * SIZE_1K * NUM_16;
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputBtyes);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightBytes);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(Conv2DTilingData));
    memset(workspace, 0, workspaceSize);
    memset(tiling, 0, sizeof(Conv2DTilingData));
    Conv2DTilingData* tilingData = reinterpret_cast<Conv2DTilingData*>(tiling);
    TilingInput tilingInput = {inputShape, weightShape, outputShape, pads, strides, dilations};
    SetTilingData(tilingData, tilingInput);
    tilingData->innerBatch = 8;
    tilingData->singleCoreBatch = 8;
    tilingData->singleCoreHo = 49;
    tilingData->hoL0 = 64;

    auto conv2dv2_func = [](GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace,
                            GM_ADDR tiling) {
        // OutputOrder=1 (M_MODE), InnerBatch=2 (MULTI), SmallKernel=1
        ::conv2dv2<0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0>(x, filter, bias, offset_w, y, workspace, tiling);
    };
    ICPU_RUN_KF(conv2dv2_func, numBlocks, input, weight, nullptr, nullptr, output, workspace, tiling);
    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

TEST_F(Conv2DV2KernelTest, conv2dv2_kernel_test_innerbatch_multi) { TestInnerBatchMultiKernel(); }
