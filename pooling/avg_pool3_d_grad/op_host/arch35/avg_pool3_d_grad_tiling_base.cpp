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
 * \file avg_pool3_d_grad_tiling_base.cpp
 * \brief 3D average pooling backward shared tiling base (arch35/runtime2.0).
 */

#include <cstdint>

#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "avg_pool3_d_grad_tiling_base.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static const int32_t ORIG_INPUT_SHAPE_INDEX = 0;
static const int32_t GRAD_INDEX = 1;
static const int32_t OUTPUT_INDEX = 0;

static const int32_t KERNEL_POS = 0;
static const int32_t STRIDE_POS = 1;
static const int32_t PADS_POS = 2;
static const int32_t CEIL_MODE_POS = 3;
static const int32_t COUNT_INCLUDE_PAD_POS = 4;
static const int32_t DIVISOR_OVERRIDE_POS = 5;
static const int32_t FORMAT_POS = 6;

static const int32_t ZERO_DIMS = 0;
static const int32_t DHW_DIMS_ = 3;
static const int32_t CDHW_DIMS_ = 4;
static const int32_t NCDHW_DIMS_ = 5;
static const int32_t PAD_DIMS_ = 6;

// Minimum workspace requested from the framework (16 MiB).
static constexpr int64_t kMinWorkspaceBytes = 16 * 1024 * 1024;

// dim positions within a 4-dim CDHW (channel-first) tensor
static const int32_t CDHW_C_DIM = 0;
static const int32_t CDHW_D_DIM = 1;
static const int32_t CDHW_H_DIM = 2;
static const int32_t CDHW_W_DIM = 3;
// dim positions within a 4-dim DHWC (channel-last) tensor
static const int32_t DHWC_D_DIM = 0;
static const int32_t DHWC_H_DIM = 1;
static const int32_t DHWC_W_DIM = 2;
static const int32_t DHWC_C_DIM = 3;

static const int32_t ONE = 1;

static inline bool IsInvalidType(const ge::DataType& dtype)
{
    static const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    return supportedDtype.count(dtype) == 0;
}

bool AvgPool3DGradTilingBase::IsGreaterThanInt32Max() const
{
    int64_t totalSize = inputData.batches * inputData.channels * inputData.inputShape[D_DIM] *
                        inputData.inputShape[H_DIM] * inputData.inputShape[W_DIM];
    return totalSize > static_cast<int64_t>(INT32_MAX);
}

bool AvgPool3DGradTilingBase::CheckInputShape()
{
    auto shapeTensor = context_->GetInputTensor(ORIG_INPUT_SHAPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeTensor);
    // orig_input_shape is a 1D tensor whose element count equals the dim num (4 or 5).
    auto shapeDim = shapeTensor->GetOriginShape().GetShapeSize();
    if (shapeDim != NCDHW_DIMS_ && shapeDim != CDHW_DIMS_) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "orig_input_shape",
                                                 std::to_string(shapeDim).c_str(), "shape dim must be 4 or 5");
        return false;
    }

    auto gradShape = context_->GetInputShape(GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradShape);
    auto outShape = context_->GetOutputShape(OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outShape);

    size_t gradDimNum = gradShape->GetStorageShape().GetDimNum();
    size_t outDimNum = outShape->GetStorageShape().GetDimNum();
    if (gradDimNum != NCDHW_DIMS_ && gradDimNum != CDHW_DIMS_) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "grads", std::to_string(gradDimNum).c_str(),
                                                 "shape dim should be 4 or 5");
        return false;
    }
    if (outDimNum != NCDHW_DIMS_ && outDimNum != CDHW_DIMS_) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "output", std::to_string(outDimNum).c_str(),
                                                 "shape dim should be 4 or 5");
        return false;
    }
    return true;
}

ge::graphStatus AvgPool3DGradTilingBase::CheckInputDtype()
{
    auto inputDesc = context_->GetInputDesc(GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto dtype = inputDesc->GetDataType();
    if (IsInvalidType(dtype)) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grads", Ops::Base::ToString(dtype).c_str(),
                                  "float16, bfloat16 and float32");
        return ge::GRAPH_FAILED;
    }
    inputData.dtypeSize = ge::GetSizeByDataType(dtype);
    if (inputData.dtypeSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dtypeSize",
                                              std::to_string(inputData.dtypeSize).c_str(),
                                              "dtype size must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3DGradTilingBase::CheckAttrShape()
{
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    auto kSizeDim = runtimeAttrs->GetListInt(KERNEL_POS)->GetSize();
    auto strideDim = runtimeAttrs->GetListInt(STRIDE_POS)->GetSize();
    auto padDim = runtimeAttrs->GetListInt(PADS_POS)->GetSize();
    if (kSizeDim != ONE_DIMS && kSizeDim != DHW_DIMS_ && kSizeDim != NCDHW_DIMS_) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "kernel_size", std::to_string(kSizeDim).c_str(),
                                     "1, 3 or 5");
        return ge::GRAPH_FAILED;
    }
    if (strideDim != ONE_DIMS && strideDim != DHW_DIMS_ && strideDim != NCDHW_DIMS_) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "stride", std::to_string(strideDim).c_str(), "1, 3 or 5");
        return ge::GRAPH_FAILED;
    }
    if (padDim != ONE_DIMS && padDim != DHW_DIMS_ && padDim != PAD_DIMS_) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "pads", std::to_string(padDim).c_str(), "1, 3 or 6");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3DGradTilingBase::CheckAttrVal()
{
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputFormat);
    std::string formatStr(inputFormat);
    if (formatStr != "NCDHW" && formatStr != "NDHWC") {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "data_format", formatStr.c_str(), "NCDHW and NDHWC");
        return ge::GRAPH_FAILED;
    }
    // ceil_mode / count_include_pad / divisor_override value checks are trivial bools/ints.
    return ge::GRAPH_SUCCESS;
}

// Dimension mapping for the grads storage shape, resolved by the grads' own rank and the
// runtime data_format (the grads tensor is physically arranged in that layout).
static void SetGradDims(const ge::Format format, const int64_t gradDimNum, AvgPool3DCommon& gradDims)
{
    if (format == ge::Format::FORMAT_NCDHW) { // channel-first
        if (gradDimNum == NCDHW_DIMS_) {      // 5D: [N, C, D, H, W]
            gradDims.nDim = 0;
            gradDims.cDim = 1;
            gradDims.dDim = 2;
            gradDims.hDim = 3;
            gradDims.wDim = 4;
        } else { // 4D: [C, D, H, W]
            gradDims.cDim = CDHW_C_DIM;
            gradDims.dDim = CDHW_D_DIM;
            gradDims.hDim = CDHW_H_DIM;
            gradDims.wDim = CDHW_W_DIM;
        }
    } else if (gradDimNum == NCDHW_DIMS_) { // NDHWC 5D: [N, D, H, W, C]
        gradDims.nDim = 0;
        gradDims.dDim = 1;
        gradDims.hDim = 2;
        gradDims.wDim = 3;
        gradDims.cDim = 4;
    } else { // NDHWC 4D: [D, H, W, C]
        gradDims.dDim = DHWC_D_DIM;
        gradDims.hDim = DHWC_H_DIM;
        gradDims.wDim = DHWC_W_DIM;
        gradDims.cDim = DHWC_C_DIM;
    }
}

// Whether the output shape, interpreted with the given output dims, reproduces the
// orig_input_shape values parsed with origDims. The n-axis is checked only when both are 5D.
static bool OutShapeMatches(const int32_t* shapeValue, const AvgPool3DCommon& origDims, const bool is5d,
                            const gert::Shape& outShape, const AvgPool3DCommon& outDims)
{
    return shapeValue[origDims.cDim] == outShape.GetDim(outDims.cDim) &&
           shapeValue[origDims.dDim] == outShape.GetDim(outDims.dDim) &&
           shapeValue[origDims.hDim] == outShape.GetDim(outDims.hDim) &&
           shapeValue[origDims.wDim] == outShape.GetDim(outDims.wDim) &&
           (!is5d || outShape.GetDimNum() != NCDHW_DIMS_ || shapeValue[origDims.nDim] == outShape.GetDim(outDims.nDim));
}

// The aclnn layer merges N and C into a single trailing channel for 5D inputs, so the
// output may be declared as [1, ..., N*C] instead of the separate [N,C,...] size list.
// Accept that merged representation when it aligns with the grads' leading dim (physical
// batch) and D/H/W still match.
static bool OutShapeMergedMatches(const int32_t* shapeValue, const AvgPool3DCommon& origDims, const bool is5d,
                                  const gert::Shape& outShape, const AvgPool3DCommon& outDims,
                                  const gert::Shape& gradShape)
{
    if (!is5d || outShape.GetDimNum() != NCDHW_DIMS_) {
        return false;
    }
    const int64_t outN = outShape.GetDim(outDims.nDim);
    const int64_t outC = outShape.GetDim(outDims.cDim);
    return outN == gradShape.GetDim(0) &&
           outC * outN == static_cast<int64_t>(shapeValue[origDims.nDim]) * shapeValue[origDims.cDim] &&
           shapeValue[origDims.dDim] == outShape.GetDim(outDims.dDim) &&
           shapeValue[origDims.hDim] == outShape.GetDim(outDims.hDim) &&
           shapeValue[origDims.wDim] == outShape.GetDim(outDims.wDim);
}

// orig_input_shape is a channel-first size list (4D [C,D,H,W], 5D [N,C,D,H,W]), i.e. its
// D/H/W are the trailing three entries. The kernel front-end feeds native data_format-ordered
// lists for 5D NDHWC ([N,D,H,W,C]), so decide by comparing orig against the grads' NDHWC
// layout (C is the trailing channel element).
static AvgPool3DCommon ResolveOrigDims(const ge::Format format, const int32_t* shapeValue, const int32_t shapeDim,
                                       const gert::Shape& gradShape)
{
    if (shapeDim == NCDHW_DIMS_ && format == ge::Format::FORMAT_NDHWC && gradShape.GetDimNum() == NCDHW_DIMS_) {
        AvgPool3DCommon ndhwc5;
        SetGradDims(ge::Format::FORMAT_NDHWC, NCDHW_DIMS_, ndhwc5);
        if (shapeValue[ndhwc5.nDim] == gradShape.GetDim(ndhwc5.nDim) &&
            shapeValue[ndhwc5.cDim] == gradShape.GetDim(ndhwc5.cDim)) {
            return ndhwc5;
        }
    }
    AvgPool3DCommon origDims;
    SetGradDims(ge::Format::FORMAT_NCDHW, shapeDim, origDims);
    return origDims;
}

void AvgPool3DGradTilingBase::SetBatchChannelInfo(const ge::Format format, const bool is5d, const int32_t* shapeValue,
                                                  const AvgPool3DCommon& origDims, const gert::Shape& gradShape)
{
    if (format == ge::Format::FORMAT_NCDHW) {
        inputData.batches = is5d ? shapeValue[origDims.nDim] * shapeValue[origDims.cDim] : shapeValue[origDims.cDim];
        inputData.channels = ONE;
    } else {
        // NDHWC: physical leading dim comes from the grads (N*C may be merged into the
        // trailing channel for 5D inputs), C from the grads' trailing channel element.
        inputData.batches = is5d ? gradShape.GetDim(0) : ONE;
        inputData.channels = gradShape.GetDim(gradShape.GetDimNum() - 1);
    }
}

ge::graphStatus AvgPool3DGradTilingBase::SetInputParams()
{
    auto shapeTensor = context_->GetInputTensor(ORIG_INPUT_SHAPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeTensor);
    auto shapeValue = shapeTensor->GetData<int32_t>();
    if (shapeValue == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto shapeDim = shapeTensor->GetOriginShape().GetShapeSize();
    const bool is5d = (shapeDim == NCDHW_DIMS_);

    auto gradShape = context_->GetInputShape(GRAD_INDEX)->GetStorageShape();
    auto outShape = context_->GetOutputShape(OUTPUT_INDEX)->GetStorageShape();

    const AvgPool3DCommon origDims = ResolveOrigDims(inputData.inputFormat, shapeValue, shapeDim, gradShape);

    // The grads storage shape follows the runtime data_format for its own rank, so it may
    // have a different rank than the size list (e.g. 4D input vs 5D NDHWC transposed grads).
    AvgPool3DCommon gradDims;
    SetGradDims(inputData.inputFormat, gradShape.GetDimNum(), gradDims);

    // The output may be declared in the runtime layout or with N*C merged into a single
    // trailing channel; accept whichever mapping reproduces orig_input_shape.
    AvgPool3DCommon outDimsFmt;
    SetGradDims(inputData.inputFormat, outShape.GetDimNum(), outDimsFmt);
    AvgPool3DCommon outDims = outDimsFmt;
    if (!OutShapeMatches(shapeValue, origDims, is5d, outShape, outDimsFmt) &&
        !OutShapeMergedMatches(shapeValue, origDims, is5d, outShape, outDimsFmt, gradShape)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "orig_input_shape and output", "shape mismatch",
                                               "orig_input_shape should be the same as output shape");
        return ge::GRAPH_FAILED;
    }

    SetBatchChannelInfo(inputData.inputFormat, is5d, shapeValue, origDims, gradShape);

    inputData.inputShape = {shapeValue[origDims.dDim], shapeValue[origDims.hDim], shapeValue[origDims.wDim]};
    inputData.gradShape = {gradShape.GetDim(gradDims.dDim), gradShape.GetDim(gradDims.hDim),
                           gradShape.GetDim(gradDims.wDim)};
    inputData.outShape = {outShape.GetDim(outDims.dDim), outShape.GetDim(outDims.hDim), outShape.GetDim(outDims.wDim)};
    return ge::GRAPH_SUCCESS;
}

void AvgPool3DGradTilingBase::SetKernelSizeInfo(const gert::RuntimeAttrs* runtimeAttrs, const AvgPool3DCommon& commInfo)
{
    auto kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    auto kSizeDim = kernelSize->GetSize();
    int64_t dKernelSize = ONE;
    int64_t hKernelSize = ONE;
    int64_t wKernelSize = ONE;
    if (kSizeDim == ONE_DIMS) {
        dKernelSize = kernelSize->GetData()[ZERO_DIMS];
        hKernelSize = dKernelSize;
        wKernelSize = dKernelSize;
    } else if (kSizeDim == DHW_DIMS_) {
        dKernelSize = kernelSize->GetData()[D_DIM];
        hKernelSize = kernelSize->GetData()[H_DIM];
        wKernelSize = kernelSize->GetData()[W_DIM];
    } else {
        dKernelSize = kernelSize->GetData()[commInfo.dDim];
        hKernelSize = kernelSize->GetData()[commInfo.hDim];
        wKernelSize = kernelSize->GetData()[commInfo.wDim];
    }
    inputData.kernelSize = {dKernelSize, hKernelSize, wKernelSize};
}

void AvgPool3DGradTilingBase::SetStrideInfo(const gert::RuntimeAttrs* runtimeAttrs, const AvgPool3DCommon& commInfo)
{
    auto stride = runtimeAttrs->GetListInt(STRIDE_POS);
    auto strideDim = stride->GetSize();
    int64_t dStride = ONE;
    int64_t hStride = ONE;
    int64_t wStride = ONE;
    if (strideDim == ONE_DIMS) {
        dStride = stride->GetData()[ZERO_DIMS];
        hStride = dStride;
        wStride = dStride;
    } else if (strideDim == DHW_DIMS_) {
        dStride = stride->GetData()[D_DIM];
        hStride = stride->GetData()[H_DIM];
        wStride = stride->GetData()[W_DIM];
    } else {
        dStride = stride->GetData()[commInfo.dDim];
        hStride = stride->GetData()[commInfo.hDim];
        wStride = stride->GetData()[commInfo.wDim];
    }
    inputData.stride = {dStride, hStride, wStride};
}

void AvgPool3DGradTilingBase::SetPadInfo(const gert::RuntimeAttrs* runtimeAttrs)
{
    auto padding = runtimeAttrs->GetListInt(PADS_POS);
    auto padDim = padding->GetSize();
    int64_t frontPad = 0;
    int64_t backendPad = 0;
    int64_t topPad = 0;
    int64_t bottomPad = 0;
    int64_t leftPad = 0;
    int64_t rightPad = 0;
    if (padDim == ONE_DIMS) {
        frontPad = padding->GetData()[FRONT_PAD_INDEX];
        backendPad = frontPad;
        topPad = frontPad;
        bottomPad = frontPad;
        leftPad = frontPad;
        rightPad = frontPad;
    } else if (padDim == DHW_DIMS_) {
        frontPad = padding->GetData()[FRONT_PAD_INDEX];
        backendPad = frontPad;
        topPad = padding->GetData()[BACKEND_PAD_INDEX];
        bottomPad = topPad;
        leftPad = padding->GetData()[TOP_PAD_INDEX];
        rightPad = leftPad;
    } else {
        frontPad = padding->GetData()[FRONT_PAD_INDEX];
        backendPad = padding->GetData()[BACKEND_PAD_INDEX];
        topPad = padding->GetData()[TOP_PAD_INDEX];
        bottomPad = padding->GetData()[BOTTOM_PAD_INDEX];
        leftPad = padding->GetData()[LEFT_PAD_INDEX];
        rightPad = padding->GetData()[RIGHT_PAD_INDEX];
    }
    inputData.pad = {frontPad, backendPad, topPad, bottomPad, leftPad, rightPad};
}

void AvgPool3DGradTilingBase::SetMiscAttrs(const gert::RuntimeAttrs* runtimeAttrs)
{
    inputData.ceilMode = false;
    const bool* ceilMode = runtimeAttrs->GetAttrPointer<bool>(CEIL_MODE_POS);
    if (ceilMode != nullptr) {
        inputData.ceilMode = *ceilMode;
    }
    // 3D uses count_include_pad directly (default true = include padding).
    // Note: 2D avg_pool_v2_grad uses exclusive, where countIncludePad = !exclusive.
    inputData.countIncludePad = true;
    const bool* countIncludePad = runtimeAttrs->GetAttrPointer<bool>(COUNT_INCLUDE_PAD_POS);
    if (countIncludePad != nullptr) {
        inputData.countIncludePad = *countIncludePad;
    }
    inputData.divisorOverride = 0;
    const int64_t* divisorOverride = runtimeAttrs->GetAttrPointer<int64_t>(DIVISOR_OVERRIDE_POS);
    if (divisorOverride != nullptr) {
        inputData.divisorOverride = *divisorOverride;
    }
}

ge::graphStatus AvgPool3DGradTilingBase::SetAttrParams()
{
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);

    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    std::string formatStr = (inputFormat != nullptr) ? std::string(inputFormat) : "NDHWC";
    inputData.inputFormat = (formatStr == "NCDHW") ? ge::Format::FORMAT_NCDHW : ge::Format::FORMAT_NDHWC;
    AvgPool3DCommon commInfo;
    SetGradDims(inputData.inputFormat, NCDHW_DIMS_, commInfo);

    SetKernelSizeInfo(runtimeAttrs, commInfo);
    SetStrideInfo(runtimeAttrs, commInfo);
    SetPadInfo(runtimeAttrs);
    SetMiscAttrs(runtimeAttrs);
    return ge::GRAPH_SUCCESS;
}

bool AvgPool3DGradTilingBase::IsKernelStrideValid() const
{
    return inputData.kernelSize[D_DIM] > 0 && inputData.kernelSize[H_DIM] > 0 && inputData.kernelSize[W_DIM] > 0 &&
           inputData.stride[D_DIM] > 0 && inputData.stride[H_DIM] > 0 && inputData.stride[W_DIM] > 0;
}

bool AvgPool3DGradTilingBase::IsPadValid() const
{
    return inputData.pad[FRONT_PAD_INDEX] >= 0 && inputData.pad[BACKEND_PAD_INDEX] >= 0 &&
           inputData.pad[TOP_PAD_INDEX] >= 0 && inputData.pad[BOTTOM_PAD_INDEX] >= 0 &&
           inputData.pad[LEFT_PAD_INDEX] >= 0 && inputData.pad[RIGHT_PAD_INDEX] >= 0 &&
           inputData.pad[FRONT_PAD_INDEX] < inputData.kernelSize[D_DIM] &&
           inputData.pad[BACKEND_PAD_INDEX] < inputData.kernelSize[D_DIM] &&
           inputData.pad[TOP_PAD_INDEX] < inputData.kernelSize[H_DIM] &&
           inputData.pad[BOTTOM_PAD_INDEX] < inputData.kernelSize[H_DIM] &&
           inputData.pad[LEFT_PAD_INDEX] < inputData.kernelSize[W_DIM] &&
           inputData.pad[RIGHT_PAD_INDEX] < inputData.kernelSize[W_DIM];
}

void AvgPool3DGradTilingBase::ComputeExpectedShape(int64_t& expectedD, int64_t& expectedH, int64_t& expectedW) const
{
    // ceil_mode adds (stride - 1) to the numerator before the floor division.
    expectedD = (inputData.inputShape[D_DIM] - inputData.kernelSize[D_DIM] + inputData.pad[FRONT_PAD_INDEX] +
                 inputData.pad[BACKEND_PAD_INDEX] + (inputData.ceilMode ? inputData.stride[D_DIM] - 1 : 0)) /
                    inputData.stride[D_DIM] +
                1;
    expectedH = (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + inputData.pad[TOP_PAD_INDEX] +
                 inputData.pad[BOTTOM_PAD_INDEX] + (inputData.ceilMode ? inputData.stride[H_DIM] - 1 : 0)) /
                    inputData.stride[H_DIM] +
                1;
    expectedW = (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + inputData.pad[LEFT_PAD_INDEX] +
                 inputData.pad[RIGHT_PAD_INDEX] + (inputData.ceilMode ? inputData.stride[W_DIM] - 1 : 0)) /
                    inputData.stride[W_DIM] +
                1;
    if (!inputData.ceilMode) {
        return;
    }
    // Drop a trailing window that starts beyond the padded input extent (torch ceil rule).
    if ((expectedD - 1) * inputData.stride[D_DIM] >= inputData.inputShape[D_DIM] + inputData.pad[FRONT_PAD_INDEX]) {
        --expectedD;
    }
    if ((expectedH - 1) * inputData.stride[H_DIM] >= inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX]) {
        --expectedH;
    }
    if ((expectedW - 1) * inputData.stride[W_DIM] >= inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX]) {
        --expectedW;
    }
}

ge::graphStatus AvgPool3DGradTilingBase::CheckGradValid()
{
    if (!IsKernelStrideValid()) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "ksize and strides",
                                               "kernel/stride should be greater than 0",
                                               "ksize and strides must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (!IsPadValid()) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "pads",
                                               "pad should be >= 0 and smaller than kernel size",
                                               "pads must be >= 0 and smaller than kernel size");
        return ge::GRAPH_FAILED;
    }
    int64_t expectedD = 0;
    int64_t expectedH = 0;
    int64_t expectedW = 0;
    ComputeExpectedShape(expectedD, expectedH, expectedW);
    if (inputData.gradShape[D_DIM] != expectedD || inputData.gradShape[H_DIM] != expectedH ||
        inputData.gradShape[W_DIM] != expectedW) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "d-dim, h-dim and w-dim of grads",
            std::to_string(inputData.gradShape[D_DIM]) + ", " + std::to_string(inputData.gradShape[H_DIM]) + ", " +
                std::to_string(inputData.gradShape[W_DIM]),
            "grad shape in d-dim, h-dim and w-dim should be " + std::to_string(expectedD) + ", " +
                std::to_string(expectedH) + " and " + std::to_string(expectedW));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void AvgPool3DGradTilingBase::SetOtherInputParams()
{
    inputData.isInt32Meet = IsGreaterThanInt32Max() ? 0 : ONE;
    inputData.hasDivisor = inputData.divisorOverride ? ONE : 0;
}

ge::graphStatus AvgPool3DGradTilingBase::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const AvgPool3DGradCompileInfo*>(context_->GetCompileInfo());
        OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context_, "compile info is null"),
                        return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<int64_t>(ubSizePlatform);
    }
    OP_TILING_CHECK(coreNum == 0, CUBE_INNER_ERR_REPORT(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3DGradTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckInputDtype(), OP_LOGE(context_->GetNodeName(), "The input dtype is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputShape(), OP_LOGE(context_->GetNodeName(), "The input relationship is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckAttrShape(), OP_LOGE(context_->GetNodeName(), "The attr shape is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckAttrVal(), OP_LOGE(context_->GetNodeName(), "The attr value is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != SetAttrParams(), OP_LOGE(context_->GetNodeName(), "Set attr params failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != SetInputParams(), OP_LOGE(context_->GetNodeName(), "Set input params failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckGradValid(), OP_LOGE(context_->GetNodeName(), "The grad shape is invalid."),
                return ge::GRAPH_FAILED);
    SetOtherInputParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3DGradTilingBase::GetWorkspaceSize()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = kMinWorkspaceBytes;
    return ge::GRAPH_SUCCESS;
}

bool AvgPool3DGradTilingBase::IsCapable() { return false; }

ge::graphStatus AvgPool3DGradTilingBase::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AvgPool3DGradTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AvgPool3DGradTilingBase::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t AvgPool3DGradTilingBase::GetTilingKey() const { return 0; }

} // namespace optiling
