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
 * \file test_conv3dv2_pointwise.cpp
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
    const uint32_t blockDim = 24;
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

    tilingDataFromBin->conv3dRunInfo.batch = inputShapeNew[0];
    tilingDataFromBin->conv3dRunInfo.din = inputShapeNew[2];
    tilingDataFromBin->conv3dRunInfo.cin = inputShapeNew[1];
    tilingDataFromBin->conv3dRunInfo.hin = inputShapeNew[3];
    tilingDataFromBin->conv3dRunInfo.win = inputShapeNew[4];
    tilingDataFromBin->conv3dRunInfo.cout = outputShapeNew[1];
    tilingDataFromBin->conv3dRunInfo.kd = weightShape[2];
    tilingDataFromBin->conv3dRunInfo.kh = weightShape[3];
    tilingDataFromBin->conv3dRunInfo.kw = weightShape[4];
    tilingDataFromBin->conv3dRunInfo.dout = outputShapeNew[2];
    tilingDataFromBin->conv3dRunInfo.hout = outputShapeNew[3];
    tilingDataFromBin->conv3dRunInfo.wout = outputShapeNew[4];
    tilingDataFromBin->conv3dRunInfo.batchDim = batchDim;
    tilingDataFromBin->conv3dRunInfo.doDim = doDim;
    tilingDataFromBin->conv3dRunInfo.mDim = mDim;
    tilingDataFromBin->conv3dRunInfo.nDim = nDim;
    tilingDataFromBin->conv3dRunInfo.strideH = stride[1];
    tilingDataFromBin->conv3dRunInfo.dilationH = dilation[1];
    tilingDataFromBin->conv3dRunInfo.strideW = stride[2];
    tilingDataFromBin->conv3dRunInfo.dilationW = dilation[2];
    tilingDataFromBin->conv3dRunInfo.strideD = stride[0];
    tilingDataFromBin->conv3dRunInfo.dilationD = dilation[0];
    tilingDataFromBin->conv3dRunInfo.padHead = pad[0];
    tilingDataFromBin->conv3dRunInfo.padTail = pad[1];
    tilingDataFromBin->conv3dRunInfo.padTop = pad[2];
    tilingDataFromBin->conv3dRunInfo.padBottom = pad[3];
    tilingDataFromBin->conv3dRunInfo.padLeft = pad[4];
    tilingDataFromBin->conv3dRunInfo.padRight = pad[5];
    tilingDataFromBin->conv3dRunInfo.hasBias = false;

    tilingDataFromBin->conv3dApiTiling.groups = 1;
    tilingDataFromBin->conv3dApiTiling.orgDo = outputShapeNew[2];
    tilingDataFromBin->conv3dApiTiling.orgCo = outputShapeNew[1];
    tilingDataFromBin->conv3dApiTiling.orgHo = outputShapeNew[3];
    tilingDataFromBin->conv3dApiTiling.orgWo = outputShapeNew[4];
    tilingDataFromBin->conv3dApiTiling.orgCi = inputShapeNew[1];
    tilingDataFromBin->conv3dApiTiling.orgDi = inputShapeNew[2];
    tilingDataFromBin->conv3dApiTiling.orgHi = inputShapeNew[3];
    tilingDataFromBin->conv3dApiTiling.orgWi = inputShapeNew[4];
    tilingDataFromBin->conv3dApiTiling.kernelD = weightShape[2];
    tilingDataFromBin->conv3dApiTiling.kernelH = weightShape[3];
    tilingDataFromBin->conv3dApiTiling.kernelW = weightShape[4];
    tilingDataFromBin->conv3dApiTiling.singleCoreCo = outputShapeNew[1];
    tilingDataFromBin->conv3dApiTiling.singleCoreDo = outputShapeNew[2];
    tilingDataFromBin->conv3dApiTiling.singleCoreM = outputShapeNew[3] * outputShapeNew[4];
    tilingDataFromBin->conv3dApiTiling.strideH = stride[1];
    tilingDataFromBin->conv3dApiTiling.strideW = stride[2];
    tilingDataFromBin->conv3dApiTiling.strideD = stride[0];
    tilingDataFromBin->conv3dApiTiling.dilationH = dilation[1];
    tilingDataFromBin->conv3dApiTiling.dilationW = dilation[2];
    tilingDataFromBin->conv3dApiTiling.dilationD = dilation[0];
    tilingDataFromBin->conv3dApiTiling.padHead = pad[0];
    tilingDataFromBin->conv3dApiTiling.padTail = pad[1];
    tilingDataFromBin->conv3dApiTiling.padTop = pad[2];
    tilingDataFromBin->conv3dApiTiling.padBottom = pad[3];
    tilingDataFromBin->conv3dApiTiling.padLeft = pad[4];
    tilingDataFromBin->conv3dApiTiling.padRight = pad[5];
    tilingDataFromBin->conv3dApiTiling.mL0 = 16;
    tilingDataFromBin->conv3dApiTiling.kL0 = 16;
    tilingDataFromBin->conv3dApiTiling.nL0 = 16;
    tilingDataFromBin->conv3dApiTiling.kAL1 = inputShape[2] * inputShape[3] * inputShape[4];
    tilingDataFromBin->conv3dApiTiling.kBL1 = weightShape[1] * weightShape[2] * weightShape[3] * weightShape[4];
    tilingDataFromBin->conv3dApiTiling.nBL1 = 16;
    tilingDataFromBin->conv3dApiTiling.mAL1 = 16;
    tilingDataFromBin->conv3dApiTiling.pBufferFlag = 0;
    tilingDataFromBin->conv3dApiTiling.kernelHxkernelW = weightShape[3] * weightShape[4];
    tilingDataFromBin->conv3dApiTiling.cin1xOriHixOriWixk0 =
        CeilDivUT(inputShapeNew[1], 16) * inputShapeNew[3] * inputShapeNew[4] * 16;
    tilingDataFromBin->conv3dApiTiling.oriHixOriWixk0 = inputShapeNew[3] * inputShapeNew[4] * 16;
    tilingDataFromBin->conv3dApiTiling.oriWixk0 = inputShapeNew[4] * 16;
    tilingDataFromBin->conv3dApiTiling.orgHixWi = inputShapeNew[3] * inputShapeNew[4];
    tilingDataFromBin->conv3dApiTiling.orgHoxWo = outputShapeNew[3] * outputShapeNew[4];
    tilingDataFromBin->conv3dApiTiling.mAL1DivmL0 = 1;
    tilingDataFromBin->conv3dApiTiling.nBL1DivnL0 = 1;
    tilingDataFromBin->conv3dApiTiling.cin1InAL1 = 1;
    tilingDataFromBin->conv3dApiTiling.kAL1Tail = inputShape[2] * inputShape[3] * inputShape[4];
    tilingDataFromBin->conv3dApiTiling.cin1InAL1Tail = 1;
    tilingDataFromBin->conv3dApiTiling.KBL1Divk0 =
        weightShape[1] * weightShape[2] * weightShape[3] * weightShape[4] / 16;
    tilingDataFromBin->conv3dApiTiling.kBL1Tail = weightShape[1] * weightShape[2] * weightShape[3] * weightShape[4];
    tilingDataFromBin->conv3dApiTiling.KBL1TailDivk0 =
        weightShape[1] * weightShape[2] * weightShape[3] * weightShape[4] / 16;
    tilingDataFromBin->conv3dApiTiling.nL0xk0 = 16 * 16;
    tilingDataFromBin->conv3dApiTiling.kL0xorgCoAlignN0 = 16 * outputShapeNew[1] * C0_B16;
    tilingDataFromBin->conv3dApiTiling.offsetx = 0;
    tilingDataFromBin->conv3dApiTiling.bl1FullLoad = 0;
    tilingDataFromBin->conv3dApiTiling.al1FullLoad = 0;
    tilingDataFromBin->conv3dApiTiling.bl1BypassFlag = 0;
    tilingDataFromBin->conv3dApiTiling.iterateMNOrder = 0;
    tilingDataFromBin->conv3dApiTiling.biasFullLoadFlag = 0;
    tilingDataFromBin->conv3dApiTiling.fixpParamsFullLoadFlag = 0;
    tilingDataFromBin->conv3dApiTiling.singleCoreGroupOpt = 1;
    tilingDataFromBin->conv3dApiTiling.groupOpt = 1;
    tilingDataFromBin->conv3dApiTiling.cinOpt = inputShapeNew[1];
    tilingDataFromBin->conv3dApiTiling.coutOpt = outputShapeNew[1];

    auto Conv3dv2Kernel = [](GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset, GM_ADDR offset_w,
                              GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::conv3dv2<0, 0, 0, 0>(x, filter, bias, scale, offset, offset_w, y, workspace, tiling);
    };
    ICPU_RUN_KF(
        Conv3dv2Kernel, blockDim, input, weight, nullptr, nullptr, nullptr, nullptr, output, workspace,
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