/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv3d_v2_pointwise.cpp
 * \brief
 */

#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <numeric>
#include <functional>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "conv3d_v2_tiling_def.h"
#include "data_utils.h"

#ifndef CONV_KERNEL
#include "../../../op_kernel/conv3d_v2.cpp"
#define CONV_KERNEL
#endif

namespace {

constexpr uint8_t DIM3_NUM = 3;
constexpr uint8_t KERNEL_DIM_IDX = 2;
constexpr uint8_t PAD_DOUBLE_DIM = 2;
constexpr uint8_t C0_B16 = 16;
constexpr uint8_t BF16_SIZE = 2;

class KernelConv3DV2PointWise_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "KernelConv3DV2PointWise_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "KernelConv3DV2PointWise_test TearDown\n" << std::endl;
    }
};

uint32_t CountOutShape(
    const uint32_t& inShape, const uint32_t& padFirst, const uint32_t& padSecond, const uint32_t& dilation,
    const uint32_t& kernelSize, const uint32_t& stride)
{
    return float((inShape + padFirst + padSecond - dilation * (kernelSize - 1) - 1) / float(stride)) + 1;
}

size_t VectorReduceMul(const std::vector<uint32_t>& vec)
{
    return std::accumulate(std::begin(vec), std::end(vec), 1, std::multiplies<size_t>());
}

static uint64_t CeilDivUT(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

void CallSimpleKernel(
    const std::vector<uint32_t>& inputShape, const std::vector<uint32_t>& weightShape,
    const std::vector<uint32_t>& dimConfig)
{
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    const uint32_t numBlocks = 24;
    const uint32_t groups = 1;

    std::vector<uint32_t> pad = {0, 0, 0, 0, 0, 0};
    std::vector<uint32_t> stride = {1, 1, 1};
    std::vector<uint32_t> dilation = {1, 1, 1};

    std::vector<uint32_t> outputShape = {inputShape[0], weightShape[0]}; // NCDHW

    const uint32_t& batchDim = dimConfig[0];
    const uint32_t& doDim = dimConfig[1];
    const uint32_t& nDim = dimConfig[2];
    const uint32_t& mDim = dimConfig[3];

    for (uint8_t curDim = 0; curDim < DIM3_NUM; ++curDim) {
        outputShape.emplace_back(CountOutShape(
            inputShape[KERNEL_DIM_IDX + curDim], pad[PAD_DOUBLE_DIM * curDim], pad[PAD_DOUBLE_DIM * curDim + 1],
            dilation[curDim], weightShape[KERNEL_DIM_IDX + curDim], stride[curDim]));
    }

    std::vector<uint32_t> inputShapeNew(inputShape);
    std::vector<uint32_t> weightShapeNew(weightShape);
    std::vector<uint32_t> outputShapeNew(outputShape);

    size_t inputByteSize = VectorReduceMul(inputShapeNew) * BF16_SIZE;
    size_t weightByteSize = VectorReduceMul(weightShapeNew) * BF16_SIZE;
    size_t outputByteSize = VectorReduceMul(outputShapeNew) * BF16_SIZE;
    size_t tilingDataSize = sizeof(Ops::NN::Conv3dV2::Conv3DRunInfo);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(Ops::NN::Conv3dV2::Conv3DV2TilingData));

    memset(workspace, 0, 16 * 1024 * 1024);

    char* path_ = get_current_dir_name();
    std::string path(path_);

    Ops::NN::Conv3dV2::Conv3DV2TilingData* tilingDataFromBin = reinterpret_cast<Ops::NN::Conv3dV2::Conv3DV2TilingData*>(tiling);

    tilingDataFromBin->convRunInfo.batch = inputShapeNew[0];
    tilingDataFromBin->convRunInfo.din = inputShapeNew[2];
    tilingDataFromBin->convRunInfo.cin = inputShapeNew[1];
    tilingDataFromBin->convRunInfo.hin = inputShapeNew[3];
    tilingDataFromBin->convRunInfo.win = inputShapeNew[4];
    tilingDataFromBin->convRunInfo.cout = outputShapeNew[1];
    tilingDataFromBin->convRunInfo.kd = weightShape[2];
    tilingDataFromBin->convRunInfo.kh = weightShape[3];
    tilingDataFromBin->convRunInfo.kw = weightShape[4];
    tilingDataFromBin->convRunInfo.dout = outputShapeNew[2];
    tilingDataFromBin->convRunInfo.hout = outputShapeNew[3];
    tilingDataFromBin->convRunInfo.wout = outputShapeNew[4];
    tilingDataFromBin->convRunInfo.batchDim = batchDim;
    tilingDataFromBin->convRunInfo.doDim = doDim;
    tilingDataFromBin->convRunInfo.mDim = mDim;
    tilingDataFromBin->convRunInfo.nDim = nDim;
    tilingDataFromBin->convRunInfo.strideH = stride[1];
    tilingDataFromBin->convRunInfo.dilationH = dilation[1];
    tilingDataFromBin->convRunInfo.strideW = stride[2];
    tilingDataFromBin->convRunInfo.dilationW = dilation[2];
    tilingDataFromBin->convRunInfo.strideD = stride[0];
    tilingDataFromBin->convRunInfo.dilationD = dilation[0];
    tilingDataFromBin->convRunInfo.padHead = pad[0];
    tilingDataFromBin->convRunInfo.padTail = pad[1];
    tilingDataFromBin->convRunInfo.padTop = pad[2];
    tilingDataFromBin->convRunInfo.padBottom = pad[3];
    tilingDataFromBin->convRunInfo.padLeft = pad[4];
    tilingDataFromBin->convRunInfo.padRight = pad[5];
    tilingDataFromBin->convRunInfo.hasBias = false;

    tilingDataFromBin->convApiTiling.groups = 1;
    tilingDataFromBin->convApiTiling.orgDo = outputShapeNew[2];
    tilingDataFromBin->convApiTiling.orgCo = outputShapeNew[1];
    tilingDataFromBin->convApiTiling.orgHo = outputShapeNew[3];
    tilingDataFromBin->convApiTiling.orgWo = outputShapeNew[4];
    tilingDataFromBin->convApiTiling.orgCi = inputShapeNew[1];
    tilingDataFromBin->convApiTiling.orgDi = inputShapeNew[2];
    tilingDataFromBin->convApiTiling.orgHi = inputShapeNew[3];
    tilingDataFromBin->convApiTiling.orgWi = inputShapeNew[4];
    tilingDataFromBin->convApiTiling.kernelD = weightShape[2];
    tilingDataFromBin->convApiTiling.kernelH = weightShape[3];
    tilingDataFromBin->convApiTiling.kernelW = weightShape[4];
    tilingDataFromBin->convApiTiling.singleCoreCo = outputShapeNew[1];
    tilingDataFromBin->convApiTiling.singleCoreDo = outputShapeNew[2];
    tilingDataFromBin->convApiTiling.singleCoreM = outputShapeNew[3] * outputShapeNew[4];
    tilingDataFromBin->convApiTiling.strideH = stride[1];
    tilingDataFromBin->convApiTiling.strideW = stride[2];
    tilingDataFromBin->convApiTiling.strideD = stride[0];
    tilingDataFromBin->convApiTiling.dilationH = dilation[1];
    tilingDataFromBin->convApiTiling.dilationW = dilation[2];
    tilingDataFromBin->convApiTiling.dilationD = dilation[0];
    tilingDataFromBin->convApiTiling.padHead = pad[0];
    tilingDataFromBin->convApiTiling.padTail = pad[1];
    tilingDataFromBin->convApiTiling.padTop = pad[2];
    tilingDataFromBin->convApiTiling.padBottom = pad[3];
    tilingDataFromBin->convApiTiling.padLeft = pad[4];
    tilingDataFromBin->convApiTiling.padRight = pad[5];
    tilingDataFromBin->convApiTiling.mL0 = 16;
    tilingDataFromBin->convApiTiling.kL0 = 16;
    tilingDataFromBin->convApiTiling.nL0 = 16;
    tilingDataFromBin->convApiTiling.kAL1 = inputShape[2] * inputShape[3] * inputShape[4];
    tilingDataFromBin->convApiTiling.kBL1 = weightShape[1] * weightShape[2] * weightShape[3] * weightShape[4];
    tilingDataFromBin->convApiTiling.nBL1 = 16;
    tilingDataFromBin->convApiTiling.mAL1 = 16;
    tilingDataFromBin->convApiTiling.pBufferFlag = 0;
    tilingDataFromBin->convApiTiling.kernelHxkernelW = weightShape[3] * weightShape[4];
    tilingDataFromBin->convApiTiling.cin1xOriHixOriWixk0 =
        CeilDivUT(inputShapeNew[1], 16) * inputShapeNew[3] * inputShapeNew[4] * 16;
    tilingDataFromBin->convApiTiling.oriHixOriWixk0 = inputShapeNew[3] * inputShapeNew[4] * 16;
    tilingDataFromBin->convApiTiling.oriWixk0 = inputShapeNew[4] * 16;
    tilingDataFromBin->convApiTiling.orgHixWi = inputShapeNew[3] * inputShapeNew[4];
    tilingDataFromBin->convApiTiling.orgHoxWo = outputShapeNew[3] * outputShapeNew[4];
    tilingDataFromBin->convApiTiling.mAL1DivmL0 = 1;
    tilingDataFromBin->convApiTiling.nBL1DivnL0 = 1;
    tilingDataFromBin->convApiTiling.cin1InAL1 = 1;
    tilingDataFromBin->convApiTiling.kAL1Tail = inputShape[2] * inputShape[3] * inputShape[4];
    tilingDataFromBin->convApiTiling.cin1InAL1Tail = 1;
    tilingDataFromBin->convApiTiling.KBL1Divk0 =
        weightShape[1] * weightShape[2] * weightShape[3] * weightShape[4] / 16;
    tilingDataFromBin->convApiTiling.kBL1Tail = weightShape[1] * weightShape[2] * weightShape[3] * weightShape[4];
    tilingDataFromBin->convApiTiling.KBL1TailDivk0 =
        weightShape[1] * weightShape[2] * weightShape[3] * weightShape[4] / 16;
    tilingDataFromBin->convApiTiling.nL0xk0 = 16 * 16;
    tilingDataFromBin->convApiTiling.kL0xorgCoAlignN0 = 16 * outputShapeNew[1] * C0_B16;
    tilingDataFromBin->convApiTiling.offsetx = 0;
    tilingDataFromBin->convApiTiling.bl1FullLoad = 0;
    tilingDataFromBin->convApiTiling.al1FullLoad = 0;
    tilingDataFromBin->convApiTiling.bl1BypassFlag = 0;
    tilingDataFromBin->convApiTiling.iterateMNOrder = 0;
    tilingDataFromBin->convApiTiling.biasFullLoadFlag = 0;
    tilingDataFromBin->convApiTiling.fixpParamsFullLoadFlag = 0;
    tilingDataFromBin->convApiTiling.singleCoreGroupOpt = 1;
    tilingDataFromBin->convApiTiling.groupOpt = 1;
    tilingDataFromBin->convApiTiling.cinOpt = inputShapeNew[1];
    tilingDataFromBin->convApiTiling.coutOpt = outputShapeNew[1];

    auto Conv3dv2Kernel = [](GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset, GM_ADDR offset_w,
                              GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::conv3dv2<0, 0, 0, 0>(x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    };
    ICPU_RUN_KF(
        Conv3dv2Kernel, numBlocks, input, weight, nullptr, nullptr, nullptr, nullptr, output, workspace,
        (uint8_t*)(tilingDataFromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(KernelConv3DV2PointWise_test, test_conv3dv2_base)
{
    std::vector<uint32_t> inputShape = {1, 16, 1, 16, 16}; // NCDHW
    std::vector<uint32_t> weightShape = {16, 16, 1, 1, 1}; // NCDHW
    std::vector<uint32_t> dimConfig = {1, 1, 1, 1};        // batchDim, doDim, nDim, mDim

    CallSimpleKernel(inputShape, weightShape, dimConfig);
}

TEST_F(KernelConv3DV2PointWise_test, test_conv3dv2_b_can_div_dout_can_div_c1_cant_div_m_cant_div)
{
    std::vector<uint32_t> inputShape = {1, 3, 3, 16, 16}; // NCDHW
    std::vector<uint32_t> weightShape = {48, 3, 1, 1, 1}; // NCDHW
    std::vector<uint32_t> dimConfig = {1, 1, 1, 1};       // batchDim, doDim, nDim, mDim

    CallSimpleKernel(inputShape, weightShape, dimConfig);
}

TEST_F(KernelConv3DV2PointWise_test, test_conv3dv2_b_can_div_dout_cant_div_c1_can_div_m_cant_div)
{
    std::vector<uint32_t> inputShape = {1, 3, 3, 16, 16}; // NCDHW
    std::vector<uint32_t> weightShape = {5, 3, 1, 1, 1};  // NCDHW
    std::vector<uint32_t> dimConfig = {1, 1, 1, 1};       // batchDim, doDim, nDim, mDim

    CallSimpleKernel(inputShape, weightShape, dimConfig);
}

} // namespace