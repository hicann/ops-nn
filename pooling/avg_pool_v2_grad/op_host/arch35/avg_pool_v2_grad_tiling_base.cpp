/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool_v2_grad_tiling_base.cpp
 * \brief
 */

#include <cstdint>
#include "tiling_base/tiling_templates_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "avg_pool_v2_grad_tiling_base.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{
static const int32_t INPUT_IDX_X = 0;

static const int32_t KERNEL_POS = 0;
static const int32_t STRIDE_POS = 1;
static const int32_t PADDING_MODE_POS = 2;
static const int32_t PADS_POS = 3;
static const int32_t FORMAT_POS = 4;
static const int32_t GLOBAL_POOLING_POS = 5;
static const int32_t CEIL_MODE_POS = 6;
static const int32_t EXCLUSIVE_POS = 7;
static const int32_t DIVISOR_OVERRIDE_POS = 8;

static const int32_t AVG_POOL_GRAD_DIM_ZERO = 0; 
static const int32_t AVG_POOL_GRAD_DIM_ONE = 1;
static const int32_t AVG_POOL_GRAD_DIM_TWO = 2;
static const int32_t AVG_POOL_GRAD_DIM_THREE = 3;
static const int32_t INDEX_GRAD = 1;

static const int32_t ONE = 1;
static const int32_t TWO = 2;
constexpr size_t ORIG_INPUT_SHAPE_INDEX = 0;

static bool IsInvalidType(const DataType dtype)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    bool dtypeInValid = (supportedDtype.count(dtype) == 0);
    return dtypeInValid;
}

static bool IsInvalidPaddingMode(std::string padMode)
{
    const std::set<std::string> supportedPadModeList = {"SAME", "VALID", "CALCULATED"};
    bool padModeInValid = (supportedPadModeList.count(padMode) == 0);
    return padModeInValid;
}

static inline bool IsGreaterThanInt32Max(const AvgPoolV2GradInputInfo& inputData, AvgPoolV2GradCommon& commInfo)
{   
    int64_t totalSize = inputData.batches * inputData.channels * inputData.inputShape[H_DIM] * inputData.inputShape[W_DIM];
    return totalSize > static_cast<int64_t>(INT32_MAX);
}

static ge::graphStatus GetPadInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                  AvgPoolV2GradInputInfo& inputData, const AvgPoolV2GradCommon& commInfo)
{   

    if (commInfo.padModeStr == "VALID") {
        inputData.pad = {0, 0, 0, 0};  // top, bottom, left, right
    } else if (commInfo.padModeStr == "SAME") {
        int64_t hPadNeed = std::max(int64_t{0}, (inputData.gradShape[H_DIM] - 1) * inputData.stride[H_DIM] +
                                                inputData.kernelSize[H_DIM] - inputData.inputShape[H_DIM]);
        int64_t topPad = hPadNeed / TWO;
        int64_t bottomPad = hPadNeed - topPad;
        
        int64_t wPadNeed = std::max(int64_t{0}, (inputData.gradShape[W_DIM] - 1) * inputData.stride[W_DIM] +
                                                inputData.kernelSize[W_DIM] - inputData.inputShape[W_DIM]);
        int64_t leftPad = wPadNeed / TWO;
        int64_t rightPad = wPadNeed - leftPad;
        
        inputData.pad = {topPad, bottomPad, leftPad, rightPad};
    } else if (commInfo.padModeStr == "CALCULATED") {
        auto padding = runtimeAttrs->GetListInt(PADS_POS);
        OPS_CHECK_NULL_WITH_CONTEXT(context, padding);
        auto paddingDim = padding->GetSize();

        OP_TILING_CHECK(paddingDim != ONE_DIMS && paddingDim != HW_DIMS && paddingDim != NCHW_DIMS,
            VECTOR_INNER_ERR_REPORT_TILIING(context, "AvgPoolV2Grad: pad list must have %d, %d, or %d elements ", ONE_DIMS, HW_DIMS, NCHW_DIMS),
             return ge::GRAPH_FAILED);
        int64_t topPad;
        int64_t bottomPad;
        int64_t leftPad;
        int64_t rightPad;
        if (paddingDim == ONE_DIMS) {
            topPad = padding->GetData()[TOP_PAD_INDEX];
            bottomPad = padding->GetData()[TOP_PAD_INDEX];
            leftPad = padding->GetData()[TOP_PAD_INDEX];
            rightPad = padding->GetData()[TOP_PAD_INDEX];
        } else if (paddingDim == HW_DIMS) {
            topPad = padding->GetData()[TOP_PAD_INDEX];
            bottomPad = padding->GetData()[TOP_PAD_INDEX];
            leftPad = padding->GetData()[BOTTOM_PAD_INDEX];
            rightPad = padding->GetData()[BOTTOM_PAD_INDEX];
        } else {
            topPad = padding->GetData()[TOP_PAD_INDEX];
            bottomPad = padding->GetData()[BOTTOM_PAD_INDEX];
            leftPad = padding->GetData()[LEFT_PAD_INDEX];
            rightPad = padding->GetData()[RIGHT_PAD_INDEX];
        }
        OP_CHECK_IF(
            topPad * TWO > inputData.kernelSize[H_DIM] || topPad < 0 || bottomPad * TWO > inputData.kernelSize[H_DIM] || bottomPad < 0 ||
                leftPad * TWO > inputData.kernelSize[W_DIM] || leftPad < 0 || rightPad * TWO > inputData.kernelSize[W_DIM] || rightPad < 0,
            OP_LOGE(
                context->GetNodeName(),
                "AvgPoolV2Grad: not support pad shape [%ld, %ld, %ld, %ld] kernel shape [%ld, %ld], pad should \
                be greater than or equal to 0 and smaller than half of the corresponding kernel size",
                topPad, bottomPad, leftPad, rightPad, inputData.kernelSize[H_DIM], inputData.kernelSize[W_DIM]),
            return ge::GRAPH_FAILED);
        inputData.pad = {topPad, bottomPad, leftPad, rightPad};
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(context, "AvgPoolV2Grad: not support padmode %s", commInfo.padModeStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetStrideInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                    AvgPoolV2GradInputInfo& inputData, const AvgPoolV2GradCommon& commInfo)
{
    auto stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context, stride);
    auto strideDim = stride->GetSize();
    OP_TILING_CHECK(strideDim != ONE_DIMS && strideDim != HW_DIMS && strideDim != NCHW_DIMS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "AvgPoolV2Grad: stride must have %d, %d, or %d elements ",
                                                    ONE_DIMS, HW_DIMS, NCHW_DIMS),
                    return ge::GRAPH_FAILED);

    int64_t hStride = ONE;
    int64_t wStride = ONE;
    if (strideDim == ONE_DIMS) {  
        hStride = stride->GetData()[AVG_POOL_GRAD_DIM_ZERO];
        wStride = stride->GetData()[AVG_POOL_GRAD_DIM_ZERO];
    } else if (strideDim == HW_DIMS) {
        hStride = stride->GetData()[AVG_POOL_GRAD_DIM_ZERO];
        wStride = stride->GetData()[AVG_POOL_GRAD_DIM_ONE];
    } else if (strideDim == NCHW_DIMS) {
        hStride = stride->GetData()[commInfo.hDim];
        wStride = stride->GetData()[commInfo.wDim];
    }
    inputData.stride = {hStride, wStride};
    OP_TILING_CHECK(hStride <= 0 || wStride <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                        "AvgPoolV2Grad: The stride of the H and W dimensions should be greater than 0, not support [%ld, %ld]",
                         hStride, wStride),
                    return ge::GRAPH_FAILED); 
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetKernelKsizeInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                AvgPoolV2GradInputInfo& inputData, const AvgPoolV2GradCommon& commInfo)
{
    auto kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context, kernelSize);
    auto kSizeDim = kernelSize->GetSize();
    OP_TILING_CHECK(
        kSizeDim != ONE_DIMS && kSizeDim != HW_DIMS && kSizeDim != NCHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context, "AvgPoolV2Grad: kernel_size must have %d, %d, or %d elements ", 
                                        ONE_DIMS, HW_DIMS, NCHW_DIMS),
        return ge::GRAPH_FAILED);
    int64_t hKernelSize = 1;
    int64_t wKernelSize = 1;
    if (kSizeDim == ONE_DIMS) {
        hKernelSize = kernelSize->GetData()[AVG_POOL_GRAD_DIM_ZERO];
        wKernelSize = kernelSize->GetData()[AVG_POOL_GRAD_DIM_ZERO];
    } else if (kSizeDim == HW_DIMS) {
        hKernelSize = kernelSize->GetData()[AVG_POOL_GRAD_DIM_ZERO];
        wKernelSize = kernelSize->GetData()[AVG_POOL_GRAD_DIM_ONE];
    } else if (kSizeDim == NCHW_DIMS) {
        hKernelSize = kernelSize->GetData()[commInfo.hDim];
        wKernelSize = kernelSize->GetData()[commInfo.wDim];
    }
    inputData.kernelSize = {hKernelSize, wKernelSize};
    
    OP_TILING_CHECK(hKernelSize <= 0 || wKernelSize <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                        "AvgPoolV2Grad: The ksize of the H and W dimensions should be greater than 0, not support [%ld, %ld]",
                        hKernelSize, wKernelSize),
                    return ge::GRAPH_FAILED); 
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(gert::TilingContext* context, gert::Shape& gradShape, gert::Shape& outputShape)
{
    OP_TILING_CHECK(
        gradShape.GetDimNum() != NCHW_DIMS && gradShape.GetDimNum() != CHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "AvgPoolV2Grad: input shape dim = %zu, should be equal 3 or 4",
                                        gradShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        outputShape.GetDimNum() != NCHW_DIMS && outputShape.GetDimNum() != CHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "AvgPoolV2Grad: output shape dim = %zu, should be equal 3 or 4",
                                        outputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    if (gradShape.GetShapeSize() == 0 && outputShape.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(gradShape.GetShapeSize() <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "AvgPoolV2Grad: input shape size %ld less than zero failed",
                                                    gradShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(outputShape.GetShapeSize() <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "AvgPoolV2Grad: output shape size %ld less than zero failed",
                                                    outputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED); 
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                AvgPoolV2GradInputInfo& inputData, AvgPoolV2GradCommon& commInfo)
{
    // 输入值依赖input
    auto inputShape0 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape0);
    auto shapeDim = inputShape0->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(
        shapeDim != NCHW_DIMS && shapeDim != CHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context, "inputShapeDim must be 3 or 4, shapeDim: %ld", shapeDim),
        return ge::GRAPH_FAILED);
    // input_grad
    auto inputShape1 = context->GetInputShape(INDEX_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape1);
    auto gradShape = EnsureNotScalar(inputShape1->GetStorageShape());
    // output_grad
    auto outX = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outX);
    auto outShape = EnsureNotScalar(outX->GetStorageShape());

    auto inputDesc = context->GetInputDesc(INDEX_GRAD);
    OPS_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    auto dtype = inputDesc->GetDataType();
    if (IsInvalidType(dtype)) {
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "AvgPoolV2Grad: invalid dtype");
        return ge::GRAPH_FAILED;
    }
    inputData.dtypeSize = ge::GetSizeByDataType(dtype);
    OP_TILING_CHECK(
        inputData.dtypeSize <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context, "inputData.dtypeSize must be greater than 0, dtypeSize: %ld", inputData.dtypeSize),
        return ge::GRAPH_FAILED);
    // 校验是否是3/4维
    OP_TILING_CHECK(CheckShape(context, gradShape, outShape) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "AvgPoolV2Grad: check shape failed"),
                    return ge::GRAPH_FAILED);
    // 值依赖转换
    const gert::Tensor* shapeTensor = context->GetInputTensor(ORIG_INPUT_SHAPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeTensor);
    const int32_t* shapeValue = shapeTensor->GetData<int32_t>();
    if (shapeValue == nullptr) {
        return ge::GRAPH_FAILED;
    }
    std::string inputFormatStr("NHWC");
    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    if (inputFormat != nullptr) {
        inputFormatStr = inputFormat;
    }
    if (inputFormatStr == "NCHW") {
        inputData.inputFormat = ge::Format::FORMAT_NCHW;
    } else if (inputFormatStr == "NHWC") {
        inputData.inputFormat = ge::Format::FORMAT_NHWC;
    } else if (shapeDim == CHW_DIMS) {
        commInfo.cDim = AVG_POOL_GRAD_DIM_ZERO;
        commInfo.hDim = AVG_POOL_GRAD_DIM_ONE;
        commInfo.wDim = AVG_POOL_GRAD_DIM_TWO;
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(context, 
            "AvgPoolV2Grad: only support NCHW、NHWC, not support format %s",
            inputFormatStr.c_str());
        return ge::GRAPH_FAILED;
    } 

    if (inputData.inputFormat == ge::Format::FORMAT_NCHW) {
        commInfo.nDim = AVG_POOL_GRAD_DIM_ZERO;
        commInfo.cDim = AVG_POOL_GRAD_DIM_ONE;
        commInfo.hDim = AVG_POOL_GRAD_DIM_TWO;
        commInfo.wDim = AVG_POOL_GRAD_DIM_THREE;
        inputData.batches = shapeValue[commInfo.nDim] * shapeValue[commInfo.cDim];
        inputData.channels = ONE;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NHWC) {
        commInfo.nDim = AVG_POOL_GRAD_DIM_ZERO;
        commInfo.cDim = AVG_POOL_GRAD_DIM_THREE;
        commInfo.hDim = AVG_POOL_GRAD_DIM_ONE;
        commInfo.wDim = AVG_POOL_GRAD_DIM_TWO;
        inputData.batches = shapeValue[commInfo.nDim];
        inputData.channels = shapeValue[commInfo.cDim];
    } else if (shapeDim == CHW_DIMS) {
        inputData.batches = ONE;
        inputData.channels = shapeValue[commInfo.cDim];
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                        "AvgPoolV2Grad: only support NCHW and NHWC, not support format.");
        return ge::GRAPH_FAILED;
    }
    inputData.inputShape = {shapeValue[commInfo.hDim], shapeValue[commInfo.wDim]};
    inputData.gradShape = {gradShape.GetDim(commInfo.hDim), gradShape.GetDim(commInfo.wDim)};
    if (shapeDim == NCHW_DIMS) {
        OP_TILING_CHECK(
        shapeValue[commInfo.nDim] != outShape.GetDim(commInfo.nDim),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), 
        "AvgPoolV2Grad: input n-dim shape value is %d, but output n-dim shape value is %ld, should be same ", shapeValue[commInfo.nDim], outShape.GetDim(commInfo.nDim)),
        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(
        shapeValue[commInfo.cDim] != outShape.GetDim(commInfo.cDim),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), 
        "AvgPoolV2Grad: input c-dim shape value is %d, but output c-dim shape value is %ld, should be same ", shapeValue[commInfo.cDim], outShape.GetDim(commInfo.cDim)),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        shapeValue[commInfo.hDim] != outShape.GetDim(commInfo.hDim),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), 
        "AvgPoolV2Grad: input h-dim shape value is %d, but output h-dim shape value is %ld, should be same ", shapeValue[commInfo.hDim], outShape.GetDim(commInfo.hDim)),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        shapeValue[commInfo.wDim] != outShape.GetDim(commInfo.wDim),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), 
        "AvgPoolV2Grad: input w-dim shape value is %d, but output w-dim shape value is %ld, should be same ", shapeValue[commInfo.hDim], outShape.GetDim(commInfo.hDim)),
        return ge::GRAPH_FAILED);
    inputData.outShape = {outShape.GetDim(commInfo.hDim), outShape.GetDim(commInfo.wDim)};
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrsInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                            AvgPoolV2GradInputInfo& inputData, AvgPoolV2GradCommon& commInfo)
{
    const char* padMode = runtimeAttrs->GetAttrPointer<char>(PADDING_MODE_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context, padMode);
    commInfo.padModeStr = padMode;
    OP_TILING_CHECK(
        IsInvalidPaddingMode(commInfo.padModeStr),
        VECTOR_INNER_ERR_REPORT_TILIING(context, "AvgPoolV2Grad: not support padmode %s", commInfo.padModeStr.c_str()),
        return ge::GRAPH_FAILED);
    const bool* ceilMode = runtimeAttrs->GetAttrPointer<bool>(CEIL_MODE_POS);
    if (ceilMode != nullptr) {
        inputData.ceilMode = *ceilMode;
    }
    const bool* exclusive = runtimeAttrs->GetAttrPointer<bool>(EXCLUSIVE_POS);
    if (exclusive != nullptr) {
        inputData.countIncludePad = !(*exclusive);
    }
    const int64_t* divisorOverride = runtimeAttrs->GetAttrPointer<int64_t>(DIVISOR_OVERRIDE_POS);
    if (divisorOverride != nullptr) {
        inputData.divisorOverride = *divisorOverride;
    }
    const bool* globalPooling = runtimeAttrs->GetAttrPointer<bool>(GLOBAL_POOLING_POS);
    if (globalPooling != nullptr) {
        inputData.globalPooling = *globalPooling;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGradShapeForValid(gert::TilingContext* context, AvgPoolV2GradInputInfo& inputData)
{
    int64_t expectedH = (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + inputData.stride[H_DIM]) / 
                        inputData.stride[H_DIM];
    int64_t expectedW = (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + inputData.stride[W_DIM]) / 
                        inputData.stride[W_DIM];
    if (inputData.gradShape[H_DIM] != expectedH || inputData.gradShape[W_DIM] != expectedW) {
        VECTOR_INNER_ERR_REPORT_TILIING(context,
                                        "AvgPoolV2Grad: when padmode is VALID, the gradshape in h-dim and w-dim should be [%ld] [%ld], but got [%ld] [%ld]",
                                        expectedH, expectedW, inputData.gradShape[H_DIM],
                                        inputData.gradShape[W_DIM]);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGradShapeForSame(gert::TilingContext* context, AvgPoolV2GradInputInfo& inputData)
{
    int64_t expectedH = (inputData.inputShape[H_DIM] + inputData.stride[H_DIM] - 1) / inputData.stride[H_DIM];
    int64_t expectedW = (inputData.inputShape[W_DIM] + inputData.stride[W_DIM] - 1) / inputData.stride[W_DIM];
    if (inputData.gradShape[H_DIM] != expectedH || inputData.gradShape[W_DIM] != expectedW) {
        VECTOR_INNER_ERR_REPORT_TILIING(context,
                                        "AvgPoolV2Grad: when padmode is SAME, the gradshape in h-dim and w-dim should be [%ld] [%ld], but got [%ld] [%ld]",
                                        expectedH, expectedW, inputData.gradShape[H_DIM],
                                        inputData.gradShape[W_DIM]);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGradShapeForCalculated(gert::TilingContext* context, AvgPoolV2GradInputInfo& inputData)
{
    int64_t topPad = inputData.pad[TOP_PAD_INDEX];
    int64_t bottomPad = inputData.pad[BOTTOM_PAD_INDEX];
    int64_t leftPad = inputData.pad[LEFT_PAD_INDEX];
    int64_t rightPad = inputData.pad[RIGHT_PAD_INDEX];
    int64_t expectedH = (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + topPad + bottomPad) / inputData.stride[H_DIM] + 1;
    int64_t expectedW = (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + leftPad + rightPad) / inputData.stride[W_DIM] + 1;
    if (inputData.ceilMode) {
        expectedH = (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + topPad + bottomPad + inputData.stride[H_DIM] - 1) / 
                        inputData.stride[H_DIM] + 1;
        expectedW = (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + leftPad + rightPad + inputData.stride[W_DIM] - 1) / 
                        inputData.stride[W_DIM] + 1;
        if ((expectedH - 1) * inputData.stride[H_DIM] >= inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX]) {
            expectedH = expectedH - 1;
        }
        if ((expectedW - 1) * inputData.stride[W_DIM] >= inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX]) {
            expectedW = expectedW - 1;
        }
    } 
    if (inputData.gradShape[H_DIM] != expectedH || inputData.gradShape[W_DIM] != expectedW) {
        VECTOR_INNER_ERR_REPORT_TILIING(context,
                                        "AvgPoolV2Grad: when padmode is Calculated, the gradshape in h-dim and w-dim should be [%ld] [%ld], but got [%ld] [%ld]",
                                        expectedH, expectedW, inputData.gradShape[H_DIM],
                                        inputData.gradShape[W_DIM]);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGradShape(gert::TilingContext* context, AvgPoolV2GradInputInfo& inputData,
                                                 const AvgPoolV2GradCommon& commInfo)
{
    if (commInfo.padModeStr == "VALID") {
        return CheckGradShapeForValid(context, inputData);
    } else if (commInfo.padModeStr == "SAME") {
        return CheckGradShapeForSame(context, inputData);
    } else if (commInfo.padModeStr == "CALCULATED") {
        return CheckGradShapeForCalculated(context, inputData);
    }
    VECTOR_INNER_ERR_REPORT_TILIING(context, 
                                    "AvgPoolV2Grad: unsupported pad mode [%s], only VALID, SAME and CALCULATED are supported", 
                                    commInfo.padModeStr.c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus GetAvgPoolV2GradPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum)
{
    auto platformPtr = context->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const AvgPoolV2GradCompileInfo*>(context->GetCompileInfo());
        OP_TILING_CHECK(
            compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "compile info is null"),
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

    OP_TILING_CHECK(
        coreNum == 0, CUBE_INNER_ERR_REPORT(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAvgPoolV2GradShapeAttrsInfo(gert::TilingContext *context, AvgPoolV2GradInputInfo& inputData)
{
    auto runtimeAttrs = context->GetAttrs();
    AvgPoolV2GradCommon commInfo;
    OPS_CHECK_NULL_WITH_CONTEXT(context, runtimeAttrs);

    OP_TILING_CHECK(GetAttrsInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "GetAttrsInfo fail."), 
                    return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(GetShapeAndDtype(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "GetShapeAndDtype fail."), 
                    return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(GetKernelKsizeInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "GetKernelKsizeInfo fail."), 
                    return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(GetStrideInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "GetStrideInfo fail."), 
                    return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(GetPadInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "GetPadInfo fail."), 
                    return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(CheckGradShape(context, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context, "CheckGradShape fail."), 
                    return ge::GRAPH_FAILED);
    
    if (IsGreaterThanInt32Max(inputData, commInfo)) {
        inputData.isInt32Meet = 0;
    } else {
        inputData.isInt32Meet = ONE;
    }

    if (!inputData.divisorOverride) {
        inputData.hasDivisor = 0;
    } else {
        inputData.hasDivisor = ONE;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling