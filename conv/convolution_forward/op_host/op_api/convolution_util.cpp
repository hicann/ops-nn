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
 * \file convolution_util.cpp
 * \brief
 */

#include <memory>
#include <vector>
#include <string>

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/reshape.h"
#include "convolution_util.h"

using namespace op;
using namespace ge;
using namespace l0op;

static uint64_t g_ubSize = 0;
static uint64_t g_l1Size = 0;

namespace SplitWInfo {
constexpr int STRIDEH_MAX = 63;
constexpr int DILATION_MAX = 255;
constexpr int PAD_MAX = 255;
constexpr int WEIGHT_MAX = 511;

constexpr int HI_INDEX = 2;
constexpr int WI_INDEX = 3;
constexpr int LEFT_INDEX_ATTR = 2;
constexpr int RIGHT_INDEX_ATTR = 3;
constexpr int W_INDEX_ATTR_CONV3D = 2;
constexpr int CONV3D_ATTR_NUM = 3;
constexpr size_t CONV_2D_DIM_SIZE = 4;
constexpr int EXTRA_NUM = 2;
constexpr int64_t BLK_M = 16;
constexpr int64_t BLK_N = 16;
constexpr int64_t POSK_LIMIT = 65535;
constexpr int64_t BLK_LEN = 32;
constexpr int64_t CONST_DOUBLE = 2;
std::map<op::DataType, uint32_t> gDataTypeSizeTab = {{op::DataType::DT_FLOAT16, 2}, {op::DataType::DT_FLOAT, 4},
                                                        {op::DataType::DT_BF16, 2}, {op::DataType::DT_INT8, 1},
                                                        {op::DataType::DT_UINT8, 1}, {op::DataType::DT_INT64, 8},
                                                        {op::DataType::DT_UINT64, 8}, {op::DataType::DT_INT32, 4}};
}  // namespace SplitWInfo

namespace ConvolutionUtil {

void Conv2DSplitWInfo::InitConv2DSplitWInfo(const aclTensor* input, const aclTensor* weight, const aclIntArray* stride,
                                            const aclIntArray* padding, const aclIntArray* dilation)
{
    hi = input->GetViewShape().GetDim(SplitWInfo::HI_INDEX);
    wi = input->GetViewShape().GetDim(SplitWInfo::WI_INDEX);
    kh = weight->GetViewShape().GetDim(SplitWInfo::HI_INDEX);
    kw = weight->GetViewShape().GetDim(SplitWInfo::WI_INDEX);
    strideH = (*stride)[0];
    strideW = (*stride)[1];
    dilationH = (*dilation)[0];
    dilationW = (*dilation)[1];
    padU = (*padding)[0];
    padD = (*padding)[1];
    padL = (*padding)[SplitWInfo::LEFT_INDEX_ATTR];
    padR = (*padding)[SplitWInfo::RIGHT_INDEX_ATTR];
}

bool Conv2DSplitWInfo::CanSwitchSplitW(const aclTensor* bias, aclTensor* output, int64_t groups,
                                        const ConvolutionOpInfo& opInfo)
{
    if (!CheckBasicInfoInSplitW(groups, opInfo)) {
        OP_LOGD("Conv2d splitw only support groups is 1, dtypes are [FP16/BF16/FP32] and soc is A2 or A3.\n");
        return false;
    }

    if (CheckConv2DPad()) {
        OP_LOGD("Conv2d splitw satisfying padU/D or padL/R not same.\n");
        return false;
    }

    if (CheckConv2DInput()) {
        OP_LOGD("Conv2d splitw satisfying strideh that is greater than or equal to 2*kernelh.\n");
        return false;
    }
    if (CheckConv2DTbeOptFlag(opInfo)) {
        OP_LOGD("Conv2d splitw satisfying TBE optimization.\n");
        return false;
    }
    if (bias) {
        biasTypeSize = SplitWInfo::gDataTypeSizeTab[opInfo.biasDtype];
    }
    k0 = SplitWInfo::BLK_LEN / SplitWInfo::gDataTypeSizeTab[opInfo.inputDtype];

    if (!CheckLoad3dIns()) {
        OP_LOGD("Conv2d splitw exceeding load3d ins posk limit %ld.\n", SplitWInfo::POSK_LIMIT);
        return false;
    }
    if (!CheckLoadL1InSplitW(bias, output)) {
        return false;
    }
    return true;
}

bool Conv2DSplitWInfo::CheckConv2DPad() const
{
    // Conv3dv2只支持padU == padB && padL==padR场景
    return (padU != padD) || (padL != padR);
}

bool Conv2DSplitWInfo::CheckConv2DInput() const
{
    // strideh >= 2*kernelh时走原始TBE
    return strideH >= SplitWInfo::CONST_DOUBLE * kh;
}

// padding = [pad_up, pad_down, pad_left, pad_right]
bool Conv2DSplitWInfo::CheckConv2DTbeOptFlag(const ConvolutionOpInfo& opInfo)
{
// 校验Load2D规格，当满足Load2D优化规格时，conv2d走原始的TBE
    bool load2dFeature = (kh == 1 && kw == 1) && (strideH == 1 && strideW == 1) && hi != 1 &&
                        (padU == 0 && padD == 0 && padL == 0 && padR == 0);
    bool supportDtype = (opInfo.inputDtype == op::DataType::DT_BF16 && opInfo.weightDtype == op::DataType::DT_BF16);
    bool canUseLoad2D = (load2dFeature && supportDtype);
    if (canUseLoad2D) {
        return true;
    }

    // 校验Conv1D
    bool canUseConv1D = (hi == 1 && kh == 1 && padU == 0 && padD == 0);
    if (canUseConv1D) {
        return true;
    }

    // 校验DMA,条件与非Load3D指令约束相同
    bool padDMAFlag = (padU > SplitWInfo::PAD_MAX || padD > SplitWInfo::PAD_MAX || padL > SplitWInfo::PAD_MAX ||
                        padR > SplitWInfo::PAD_MAX);
    bool strideDMAFlag = (strideH > SplitWInfo::STRIDEH_MAX || strideW > SplitWInfo::STRIDEH_MAX);
    bool dilationDMAFlag = (dilationH > SplitWInfo::DILATION_MAX || dilationW > SplitWInfo::DILATION_MAX);
    bool kernelDMAFlag = (kh > SplitWInfo::WEIGHT_MAX || kw > SplitWInfo::WEIGHT_MAX);
    if (padDMAFlag || strideDMAFlag || dilationDMAFlag || kernelDMAFlag) {
        return true;
    }
    return false;
}

bool Conv2DSplitWInfo::CheckBasicInfoInSplitW(int64_t groups, const ConvolutionOpInfo& opInfo)
{
    if (groups != 1) {
        return false;
    }

    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93 &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        return false;
    }

    const std::unordered_set<op::DataType> supportedDtypes{op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                           op::DataType::DT_BF16};
    bool supportDtypeFlag = (supportedDtypes.count(opInfo.inputDtype) > 0) &&
        (supportedDtypes.count(opInfo.weightDtype) > 0);
    if (!supportDtypeFlag) {
        return false;
    }
    return true;
}

bool Conv2DSplitWInfo::CheckLoad3dIns()
{
    // 在CheckConv2DTbeOptFlag中已经校验了load3d指令，当走到这里说明满足load3d指令约束
    bool posKFlag = (kh * kw * k0 <= SplitWInfo::POSK_LIMIT);
    return posKFlag;
}

bool Conv2DSplitWInfo::CheckLoadL1InSplitW(const aclTensor* bias, aclTensor* output)
{
    // 910B/910_93中tbe和ascendc可用的hardwareInfo.l1size 分别为524288和524032字节
    constexpr int64_t l1BufferSizeForTbe = 524288;
    constexpr int64_t l1BufferSizeForAsc = 524032;
    // load3d M模式最小切分
    int64_t outputW = output->GetViewShape().GetDim(SplitWInfo::WI_INDEX);
    if (outputW == 0) {
        return false;
    }
    int64_t hoAL1Min = SplitWInfo::BLK_M / outputW + SplitWInfo::EXTRA_NUM;
    int64_t hkDilation = (kh - 1) * dilationH + 1;
    int64_t hiAL1Min = std::min(((hoAL1Min - 1) * strideH + hkDilation), hi);
    int64_t minL1Size = hiAL1Min * wi * SplitWInfo::BLK_LEN;
    if (bias != nullptr) {
        minL1Size += SplitWInfo::BLK_N * biasTypeSize;
    }
    if (minL1Size <= l1BufferSizeForTbe) {
        OP_LOGD("Conv2d splitw minL1Size less than L1 Buffer in m mode.\n");
        return false;
    }

    // load3d HW模式最小切分, hoAL1Min = 1
    hiAL1Min = std::min(hkDilation, hi);
    int64_t woAL1Min = 16;
    int64_t wkDilation = (kw - 1) * dilationW + 1;
    int64_t wiAL1_min = std::min(((woAL1Min - 1) * strideW + wkDilation), wi);
    minL1Size = hiAL1Min * wiAL1_min * SplitWInfo::BLK_LEN;
    if (bias != nullptr) {
        minL1Size += SplitWInfo::BLK_N * biasTypeSize;
    }
    if (minL1Size > l1BufferSizeForAsc) {
        OP_LOGD("Conv2d splitw minL1Size greater than L1 Buffer in hw mode.\n");
        return false;
    }
    return true;
}

aclIntArray* View2dAs3dForAttr(const aclIntArray* intArray, int64_t expendValue, aclOpExecutor* executor, bool isPad)
{
    int64_t data[SplitWInfo::CONV3D_ATTR_NUM];
    uint64_t size = intArray->Size();
    // 对于非pad的attr需要校验size，pad在前面已经保证了为4维
    if (!isPad && (size != static_cast<uint64_t>(SplitWInfo::CONV3D_ATTR_NUM - 1))) {
        return nullptr;
    }
    data[0] = expendValue;
    data[1] = (*intArray)[0];
    if (isPad) {
        data[SplitWInfo::W_INDEX_ATTR_CONV3D] = (*intArray)[SplitWInfo::LEFT_INDEX_ATTR];
    } else {
        data[SplitWInfo::W_INDEX_ATTR_CONV3D] = (*intArray)[1];
    }
    aclIntArray* newArray = executor->AllocIntArray(data, SplitWInfo::CONV3D_ATTR_NUM);
    return newArray;
}

aclIntArray* View2DSwapHWForAttr(const aclIntArray* intArray, aclOpExecutor* executor)
{
    int64_t data[2];
    uint64_t size = intArray->Size();
    // 对于非pad的attr需要校验size，pad在前面已经保证了为4维
    if ((size != static_cast<uint64_t>(SplitWInfo::CONV3D_ATTR_NUM - 1))) {
        return nullptr;
    }
    data[0] = (*intArray)[1];
    data[1] = (*intArray)[0];
    aclIntArray* newArray = executor->AllocIntArray(data, 2);
    return newArray;
}

const aclTensor* View4DSwapHWForTensor(const aclTensor* input, aclOpExecutor* executor)
{
    auto dims = input->GetViewShape().GetDimNum();
    CHECK_RET(dims == SplitWInfo::CONV_2D_DIM_SIZE, nullptr);
    auto shape = op::ToShapeVector(input->GetViewShape());
    FVector<int64_t> newShape = {shape[0], shape[1], shape[3], shape[2]};
    aclIntArray* shapeArray = executor->AllocIntArray(newShape.data(), newShape.size());
    CHECK_RET(shapeArray != nullptr, nullptr);
    input = l0op::Reshape(input, shapeArray, executor);
    CHECK_RET(input != nullptr, nullptr);
    return input;
}

const aclTensor* View4dAs5dForInput(const aclTensor* input, aclOpExecutor* executor)
{
    // input NCHW->contigious->unsqueeze(2)->reformat NCDHW
    // 非连续转连续contigious
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);

    // unsqeeze(2)
    const int64_t appendDim[] = {SplitWInfo::HI_INDEX};
    aclIntArray* dim = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dim != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dim, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);

    // reformat
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

aclnnStatus ChangeConv2dAttrToConv3d(const aclIntArray* &stride, const aclIntArray* &padding,
                                    const aclIntArray* &dilation, aclOpExecutor* executor)
{
    stride = View2dAs3dForAttr(stride, 1, executor, false);
    CHECK_RET(stride != nullptr, ACLNN_ERR_INNER_NULLPTR);
    dilation = View2dAs3dForAttr(dilation, 1, executor, false);
    CHECK_RET(dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);
    padding = View2dAs3dForAttr(padding, 0, executor, true);
    CHECK_RET(padding != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus ChangeConv2dInputToConv3d(const aclTensor* &input, const aclTensor* &weight, aclOpExecutor* executor)
{
    input = View4dAs5dForInput(input, executor);
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    weight = View4dAs5dForInput(weight, executor);
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

const aclTensor* View5dAs4dForOutput(const aclTensor* input, aclOpExecutor* executor)
{
    // input NCDHW->contigious->squeeze(2)->reformat NCHW
    // 非连续转连续contigious
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);
    // sqeeze(2)
    const int64_t appendDim[] = {SplitWInfo::HI_INDEX};
    aclIntArray* dim = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dim != nullptr, nullptr);
    auto squeezedInput = l0op::SqueezeNd(contiguousInput, dim, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);

    // reformat
    auto reformatInput = l0op::ReFormat(squeezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

bool CheckDisContinuousStride(const aclTensor* input, const std::vector<int64_t>& newStrides, uint32_t dims)
{
    auto viewStrides = input->GetViewStrides();
    uint32_t totalDims = std::min(viewStrides.size(), newStrides.size());
    if (dims > totalDims) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Invalid dims");
    }
    for (size_t i = 0; i < dims; i++) {
        if (viewStrides[i] != newStrides[i]) {
            return false;
        }
    }
    return true;
}

void GetUbSize()
{
    if (g_ubSize != 0) {
        return;
    }
    uint64_t ubSize = 0;
    auto platformInfo = GetCurrentPlatformInfo().GetPlatformInfos();
    if (platformInfo == nullptr) {
        OP_LOGW("Platform info is null, ubSize fallback to 0.");
        return;
    }
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, ubSize);
    OP_LOGD("GetUbSize returned: %lu", ubSize);
    g_ubSize = ubSize - RESERVED_SIZE_8K;
}

void GetL1Size()
{
    if (g_l1Size != 0) {
        return;
    }
    auto platformInfo = GetCurrentPlatformInfo().GetPlatformInfos();
    if (platformInfo == nullptr) {
        OP_LOGW("Platform info is null, ubSize fallback to 0.");
        return;
    }
    platformInfo->GetLocalMemSize(fe::LocalMemType::L1, g_l1Size);
    OP_LOGD("GetL1Size returned: %lu", g_l1Size);
}

bool CheckDmaLimits(const struct ConvolutionOpInfo* opInfo, const aclTensor* input, const aclTensor* weight,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, const aclTensor* bias)
{
    int64_t orgKh = static_cast<int64_t>(weight->GetViewShape().GetDim(SplitWInfo::HI_INDEX));
    int64_t orgKw = static_cast<int64_t>(weight->GetViewShape().GetDim(SplitWInfo::WI_INDEX));
    int64_t hin = static_cast<int64_t>(input->GetViewShape().GetDim(SplitWInfo::HI_INDEX));
    int64_t win = static_cast<int64_t>(input->GetViewShape().GetDim(SplitWInfo::WI_INDEX));
    int64_t strideH = (*stride)[0];
    int64_t strideW = (*stride)[1];
    int64_t dilationH = (*dilation)[0];
    int64_t dilationW = (*dilation)[1];
    int64_t padLeft = (*padding)[SplitWInfo::LEFT_INDEX_ATTR];
    int64_t padRight = (*padding)[SplitWInfo::RIGHT_INDEX_ATTR];
    int64_t m0 = SplitWInfo::BLK_M;
    int64_t n0 = SplitWInfo::BLK_N;
    int64_t k0 = SplitWInfo::BLK_LEN / SplitWInfo::gDataTypeSizeTab[opInfo->weightDtype];
    uint32_t inputDtypeSize = SplitWInfo::gDataTypeSizeTab[opInfo->inputDtype];
    uint32_t weightDtypeSize = SplitWInfo::gDataTypeSizeTab[opInfo->weightDtype];
    uint32_t biasDtypeSize = SplitWInfo::gDataTypeSizeTab[opInfo->biasDtype];
    uint64_t nBL1min = n0;

    if (orgKh * orgKw * k0 > SplitWInfo::POSK_LIMIT) {
        return true;
    }

    uint64_t biasL1Size = 0;
    if (bias != nullptr) {
        biasL1Size = ConvAlignB(nBL1min * biasDtypeSize, SplitWInfo::BLK_LEN);
    }

    uint64_t kBL1min = k0 * orgKh * orgKw;
    uint64_t weightL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, SplitWInfo::BLK_LEN);
    uint64_t inputL1Size = 0;
    uint64_t orgWo = (win + padLeft + padRight - (dilationW * (orgKw - 1) + 1)) / strideW + 1;
    uint64_t hoAL1min = orgWo < m0 ? (m0 + orgWo - 1) / orgWo : 1;
    uint64_t khDilated = (orgKh - 1) * dilationH + 1; 
    uint64_t hiAL1min = Conv2DInferHiL1(hoAL1min, khDilated, hin, strideH);
    uint64_t kAL1min = k0;
    uint64_t woAL1min = m0;
    uint64_t kwDilated = (orgKw - 1) * dilationW + 1;
    uint64_t wiAL1min = Conv2DInferHiL1(woAL1min, kwDilated, win, strideW);
    inputL1Size = ConvAlignB(hiAL1min * wiAL1min * kAL1min * inputDtypeSize, SplitWInfo::BLK_LEN);

    GetL1Size();
    uint64_t minL1LoadSize = biasL1Size + inputL1Size + weightL1Size;
    if (minL1LoadSize > g_l1Size) {
        return true;
    }

    return !CheckL1SizeLimitsDma(inputDtypeSize, biasL1Size, weightDtypeSize, k0);
}

bool CheckL1SizeLimitsDma(uint32_t inputDtypeSize, uint64_t biasL1Size, uint32_t weightDtypeSize, int64_t k0)
{
    uint64_t hoAL1Min = 1;
    GetUbSize();
    uint64_t inputUbSizeMin = ConvAlignB(hoAL1Min * SplitWInfo::BLK_M * k0 * inputDtypeSize, SplitWInfo::BLK_LEN);
    if (inputUbSizeMin > g_ubSize) {
        OP_LOGD("DMA min ub size not enough: ubSize=%lu, inputUbSizeMin=%lu.", g_ubSize, inputUbSizeMin);
        return false;
    }

    uint64_t nBL1min = SplitWInfo::BLK_M;
    uint64_t kBL1min = k0;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, SplitWInfo::BLK_LEN);
    uint64_t kAL1min = k0;
    uint64_t woAL1min = SplitWInfo::BLK_M;
    uint64_t inputUsedL1Size = ConvAlignB(hoAL1Min * woAL1min * kAL1min * inputDtypeSize, SplitWInfo::BLK_LEN);
    uint64_t minL1LoadSize = biasL1Size + inputUsedL1Size + weightUsedL1Size;
    GetL1Size();
    if (minL1LoadSize > g_l1Size) {
        return false;
    }
    return true;
}

uint64_t Conv2DInferHiL1(uint64_t inputHoL1, uint64_t khDilated, uint64_t hi, uint64_t strideH)
{
    uint64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }
    return tmpHiL1;
}

uint64_t ConvAlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

} // namespace ConvolutionUtil