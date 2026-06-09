/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file avg_pool_v2_grad_tiling_base.cpp
 * \brief
 */

#include <cstdint>
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "avg_pool_v2_grad_tiling_base.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
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
    int64_t totalSize =
        inputData.batches * inputData.channels * inputData.inputShape[H_DIM] * inputData.inputShape[W_DIM];
    return totalSize > static_cast<int64_t>(INT32_MAX);
}

static ge::graphStatus GetPadInfo(
    gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs, AvgPoolV2GradInputInfo& inputData,
    const AvgPoolV2GradCommon& commInfo)
{
    if (commInfo.padModeStr == "VALID") {
        inputData.pad = {0, 0, 0, 0}; // top, bottom, left, right
    } else if (commInfo.padModeStr == "SAME") {
        int64_t hPadNeed = std::max(
            int64_t{0}, (inputData.gradShape[H_DIM] - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM] -
                            inputData.inputShape[H_DIM]);
        int64_t topPad = hPadNeed / TWO;
        int64_t bottomPad = hPadNeed - topPad;

        int64_t wPadNeed = std::max(
            int64_t{0}, (inputData.gradShape[W_DIM] - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM] -
                            inputData.inputShape[W_DIM]);
        int64_t leftPad = wPadNeed / TWO;
        int64_t rightPad = wPadNeed - leftPad;

        inputData.pad = {topPad, bottomPad, leftPad, rightPad};
    } else if (commInfo.padModeStr == "CALCULATED") {
        auto padding = runtimeAttrs->GetListInt(PADS_POS);
        OPS_CHECK_NULL_WITH_CONTEXT(context, padding);
        auto paddingDim = padding->GetSize();

        if (paddingDim != ONE_DIMS && paddingDim != HW_DIMS && paddingDim != NCHW_DIMS) {
            OP_LOGE_FOR_INVALID_LISTSIZE(
                context->GetNodeName(), "pad", std::to_string(paddingDim).c_str(), "1, 2 or 4");
            return ge::GRAPH_FAILED;
        }
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
        if (topPad * TWO > inputData.kernelSize[H_DIM] || topPad < 0 || bottomPad * TWO > inputData.kernelSize[H_DIM] ||
            bottomPad < 0 || leftPad * TWO > inputData.kernelSize[W_DIM] || leftPad < 0 ||
            rightPad * TWO > inputData.kernelSize[W_DIM] || rightPad < 0) {
            std::string valMsg = std::to_string(topPad) + ", " + std::to_string(bottomPad) + ", " +
                                 std::to_string(leftPad) + " and " + std::to_string(rightPad) + " with kernel " +
                                 std::to_string(inputData.kernelSize[H_DIM]) + " and " +
                                 std::to_string(inputData.kernelSize[W_DIM]);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "pad", valMsg.c_str(),
                "pad should be greater than or equal to 0 and smaller than half of the corresponding kernel size");
            return ge::GRAPH_FAILED;
        }
        inputData.pad = {topPad, bottomPad, leftPad, rightPad};
    } else {
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), "padding_mode", commInfo.padModeStr.c_str(), "SAME, VALID and CALCULATED");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetStrideInfo(
    gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs, AvgPoolV2GradInputInfo& inputData,
    const AvgPoolV2GradCommon& commInfo)
{
    auto stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context, stride);
    auto strideDim = stride->GetSize();
    if (strideDim != ZERO_DIMS && strideDim != ONE_DIMS && strideDim != HW_DIMS && strideDim != NCHW_DIMS) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            context->GetNodeName(), "stride", std::to_string(strideDim).c_str(), "0, 1, 2 or 4");
        return ge::GRAPH_FAILED;
    }

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
    } else if (strideDim == ZERO_DIMS) {
        hStride = inputData.kernelSize[H_DIM];
        wStride = inputData.kernelSize[W_DIM];
    }
    inputData.stride = {hStride, wStride};
    if (hStride <= 0 || wStride <= 0) {
        std::string strideMsg = std::to_string(hStride) + " and " + std::to_string(wStride);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "hStride and wStride", strideMsg.c_str(),
            "stride of H and W dimensions should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetKernelKsizeInfo(
    gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs, AvgPoolV2GradInputInfo& inputData,
    const AvgPoolV2GradCommon& commInfo)
{
    auto kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context, kernelSize);
    auto kSizeDim = kernelSize->GetSize();
    if (kSizeDim != ONE_DIMS && kSizeDim != HW_DIMS && kSizeDim != NCHW_DIMS) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            context->GetNodeName(), "kernel_size", std::to_string(kSizeDim).c_str(), "1, 2 or 4");
        return ge::GRAPH_FAILED;
    }
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

    if (hKernelSize <= 0 || wKernelSize <= 0) {
        std::string ksizeMsg = std::to_string(hKernelSize) + " and " + std::to_string(wKernelSize);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "hKernelSize and wKernelSize", ksizeMsg.c_str(),
            "ksize of H and W dimensions should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(gert::TilingContext* context, gert::Shape& gradShape, gert::Shape& outputShape)
{
    if (gradShape.GetDimNum() != NCHW_DIMS && gradShape.GetDimNum() != CHW_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "input_grad", std::to_string(gradShape.GetDimNum()).c_str(),
            "shape dim should be 3 or 4");
        return ge::GRAPH_FAILED;
    }
    if (outputShape.GetDimNum() != NCHW_DIMS && outputShape.GetDimNum() != CHW_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "output", std::to_string(outputShape.GetDimNum()).c_str(),
            "shape dim should be 3 or 4");
        return ge::GRAPH_FAILED;
    }
    if (gradShape.GetShapeSize() == 0 && outputShape.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (gradShape.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(
            context->GetNodeName(), "input_grad", std::to_string(gradShape.GetShapeSize()).c_str(), "greater than 0");
        return ge::GRAPH_FAILED;
    }

    if (outputShape.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(
            context->GetNodeName(), "output", std::to_string(outputShape.GetShapeSize()).c_str(), "greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAndDtype(
    gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs, AvgPoolV2GradInputInfo& inputData,
    AvgPoolV2GradCommon& commInfo)
{
    // 输入值依赖input
    auto inputShape0 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape0);
    auto shapeDim = inputShape0->GetStorageShape().GetDim(0);
    if (shapeDim != NCHW_DIMS && shapeDim != CHW_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "orig_input_shape", std::to_string(shapeDim).c_str(), "shape dim must be 3 or 4");
        return ge::GRAPH_FAILED;
    }
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
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "input_grad", Ops::Base::ToString(dtype).c_str(), "float16, bfloat16 and float32");
        return ge::GRAPH_FAILED;
    }
    inputData.dtypeSize = ge::GetSizeByDataType(dtype);
    if (inputData.dtypeSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "dtypeSize", std::to_string(inputData.dtypeSize).c_str(),
            "dtype size must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    // 校验是否是3/4维
    if (CheckShape(context, gradShape, outShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "shape", "check_failed", "check shape validation failed");
        return ge::GRAPH_FAILED;
    }
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
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "input_format", inputFormatStr.c_str(), "NCHW and NHWC");
        return ge::GRAPH_FAILED;
    }

    if (inputData.inputFormat == ge::Format::FORMAT_NCHW) {
        if (shapeDim == CHW_DIMS) {
            commInfo.cDim = AVG_POOL_GRAD_DIM_ZERO;
            commInfo.hDim = AVG_POOL_GRAD_DIM_ONE;
            commInfo.wDim = AVG_POOL_GRAD_DIM_TWO;
            inputData.batches = shapeValue[commInfo.cDim];
        } else {
            commInfo.nDim = AVG_POOL_GRAD_DIM_ZERO;
            commInfo.cDim = AVG_POOL_GRAD_DIM_ONE;
            commInfo.hDim = AVG_POOL_GRAD_DIM_TWO;
            commInfo.wDim = AVG_POOL_GRAD_DIM_THREE;
            inputData.batches = shapeValue[commInfo.nDim] * shapeValue[commInfo.cDim];
        }
        inputData.channels = ONE;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NHWC) {
        if (shapeDim == CHW_DIMS) {
            commInfo.cDim = AVG_POOL_GRAD_DIM_TWO;
            commInfo.hDim = AVG_POOL_GRAD_DIM_ZERO;
            commInfo.wDim = AVG_POOL_GRAD_DIM_ONE;
            inputData.batches = ONE;
        } else {
            commInfo.nDim = AVG_POOL_GRAD_DIM_ZERO;
            commInfo.cDim = AVG_POOL_GRAD_DIM_THREE;
            commInfo.hDim = AVG_POOL_GRAD_DIM_ONE;
            commInfo.wDim = AVG_POOL_GRAD_DIM_TWO;
            inputData.batches = shapeValue[commInfo.nDim];
        }
        inputData.channels = shapeValue[commInfo.cDim];
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(
            context->GetNodeName(), "input_format", Ops::Base::ToString(inputData.inputFormat).c_str(),
            "NCHW and NHWC");
        return ge::GRAPH_FAILED;
    }
    inputData.inputShape = {shapeValue[commInfo.hDim], shapeValue[commInfo.wDim]};
    inputData.gradShape = {gradShape.GetDim(commInfo.hDim), gradShape.GetDim(commInfo.wDim)};
    if (shapeDim == NCHW_DIMS) {
        if (shapeValue[commInfo.nDim] != outShape.GetDim(commInfo.nDim)) {
            std::string valMsg = "input " + std::to_string(shapeValue[commInfo.nDim]) + " and output " +
                                 std::to_string(outShape.GetDim(commInfo.nDim));
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "n-dim of input and output", valMsg.c_str(),
                "n-dim of input and output should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    if (shapeValue[commInfo.cDim] != outShape.GetDim(commInfo.cDim)) {
        std::string valMsg = "input " + std::to_string(shapeValue[commInfo.cDim]) + " and output " +
                             std::to_string(outShape.GetDim(commInfo.cDim));
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "c-dim of input and output", valMsg.c_str(),
            "c-dim of input and output should be the same");
        return ge::GRAPH_FAILED;
    }
    if (shapeValue[commInfo.hDim] != outShape.GetDim(commInfo.hDim)) {
        std::string valMsg = "input " + std::to_string(shapeValue[commInfo.hDim]) + " and output " +
                             std::to_string(outShape.GetDim(commInfo.hDim));
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "h-dim of input and output", valMsg.c_str(),
            "h-dim of input and output should be the same");
        return ge::GRAPH_FAILED;
    }
    if (shapeValue[commInfo.wDim] != outShape.GetDim(commInfo.wDim)) {
        std::string valMsg = "input " + std::to_string(shapeValue[commInfo.wDim]) + " and output " +
                             std::to_string(outShape.GetDim(commInfo.wDim));
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "w-dim of input and output", valMsg.c_str(),
            "w-dim of input and output should be the same");
        return ge::GRAPH_FAILED;
    }
    inputData.outShape = {outShape.GetDim(commInfo.hDim), outShape.GetDim(commInfo.wDim)};
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrsInfo(
    gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs, AvgPoolV2GradInputInfo& inputData,
    AvgPoolV2GradCommon& commInfo)
{
    const char* padMode = runtimeAttrs->GetAttrPointer<char>(PADDING_MODE_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context, padMode);
    commInfo.padModeStr = padMode;
    if (IsInvalidPaddingMode(commInfo.padModeStr)) {
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), "padding_mode", commInfo.padModeStr.c_str(), "SAME, VALID and CALCULATED");
        return ge::GRAPH_FAILED;
    }
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
    int64_t expectedH =
        (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + inputData.stride[H_DIM]) / inputData.stride[H_DIM];
    int64_t expectedW =
        (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + inputData.stride[W_DIM]) / inputData.stride[W_DIM];
    if (inputData.gradShape[H_DIM] != expectedH || inputData.gradShape[W_DIM] != expectedW) {
        std::string shapeMsg =
            std::to_string(inputData.gradShape[H_DIM]) + " and " + std::to_string(inputData.gradShape[W_DIM]);
        std::string reasonMsg = "when padmode is VALID, grad shape in h-dim and w-dim should be " +
                                std::to_string(expectedH) + " and " + std::to_string(expectedW);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "h-dim and w-dim of input_grad", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGradShapeForSame(gert::TilingContext* context, AvgPoolV2GradInputInfo& inputData)
{
    int64_t expectedH = (inputData.inputShape[H_DIM] + inputData.stride[H_DIM] - 1) / inputData.stride[H_DIM];
    int64_t expectedW = (inputData.inputShape[W_DIM] + inputData.stride[W_DIM] - 1) / inputData.stride[W_DIM];
    if (inputData.gradShape[H_DIM] != expectedH || inputData.gradShape[W_DIM] != expectedW) {
        std::string shapeMsg =
            std::to_string(inputData.gradShape[H_DIM]) + " and " + std::to_string(inputData.gradShape[W_DIM]);
        std::string reasonMsg = "when padmode is SAME, grad shape in h-dim and w-dim should be " +
                                std::to_string(expectedH) + " and " + std::to_string(expectedW);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "h-dim and w-dim of input_grad", shapeMsg.c_str(), reasonMsg.c_str());
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
    int64_t expectedH =
        (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + topPad + bottomPad) / inputData.stride[H_DIM] + 1;
    int64_t expectedW =
        (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + leftPad + rightPad) / inputData.stride[W_DIM] + 1;
    if (inputData.ceilMode) {
        expectedH = (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + topPad + bottomPad +
                     inputData.stride[H_DIM] - 1) /
                        inputData.stride[H_DIM] +
                    1;
        expectedW = (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + leftPad + rightPad +
                     inputData.stride[W_DIM] - 1) /
                        inputData.stride[W_DIM] +
                    1;
        if ((expectedH - 1) * inputData.stride[H_DIM] >= inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX]) {
            expectedH = expectedH - 1;
        }
        if ((expectedW - 1) * inputData.stride[W_DIM] >= inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX]) {
            expectedW = expectedW - 1;
        }
    }
    if (inputData.gradShape[H_DIM] != expectedH || inputData.gradShape[W_DIM] != expectedW) {
        std::string shapeMsg =
            std::to_string(inputData.gradShape[H_DIM]) + " and " + std::to_string(inputData.gradShape[W_DIM]);
        std::string reasonMsg = "when padmode is CALCULATED, grad shape in h-dim and w-dim should be " +
                                std::to_string(expectedH) + " and " + std::to_string(expectedW);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "h-dim and w-dim of input_grad", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGradShape(
    gert::TilingContext* context, AvgPoolV2GradInputInfo& inputData, const AvgPoolV2GradCommon& commInfo)
{
    if (commInfo.padModeStr == "VALID") {
        return CheckGradShapeForValid(context, inputData);
    } else if (commInfo.padModeStr == "SAME") {
        return CheckGradShapeForSame(context, inputData);
    } else if (commInfo.padModeStr == "CALCULATED") {
        return CheckGradShapeForCalculated(context, inputData);
    }
    OP_LOGE_FOR_INVALID_VALUE(
        context->GetNodeName(), "padding_mode", commInfo.padModeStr.c_str(), "SAME, VALID and CALCULATED");
    return ge::GRAPH_FAILED;
}

ge::graphStatus GetAvgPoolV2GradPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum)
{
    auto platformPtr = context->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const AvgPoolV2GradCompileInfo*>(context->GetCompileInfo());
        OP_TILING_CHECK(
            compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<int64_t>(ubSizePlatform);
    }

    OP_TILING_CHECK(coreNum == 0, CUBE_INNER_ERR_REPORT(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAvgPoolV2GradShapeAttrsInfo(gert::TilingContext* context, AvgPoolV2GradInputInfo& inputData)
{
    auto runtimeAttrs = context->GetAttrs();
    AvgPoolV2GradCommon commInfo;
    OPS_CHECK_NULL_WITH_CONTEXT(context, runtimeAttrs);
    if (GetAttrsInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attrs", "check_failed", "GetAttrsInfo validation failed");
        return ge::GRAPH_FAILED;
    }
    if (GetShapeAndDtype(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "shape_and_dtype", "check_failed", "GetShapeAndDtype validation failed");
        return ge::GRAPH_FAILED;
    }
    if (GetKernelKsizeInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "kernel_size", "check_failed", "GetKernelKsizeInfo validation failed");
        return ge::GRAPH_FAILED;
    }
    if (GetStrideInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "stride", "check_failed", "GetStrideInfo validation failed");
        return ge::GRAPH_FAILED;
    }
    if (GetPadInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "pad", "check_failed", "GetPadInfo validation failed");
        return ge::GRAPH_FAILED;
    }
    if (inputData.globalPooling) {
        if ((inputData.inputShape[0] != 0 && inputData.gradShape[0] != 1) ||
            (inputData.inputShape[1] != 0 && inputData.gradShape[1] != 1)) {
            std::string shapeMsg =
                std::to_string(inputData.gradShape[0]) + " and " + std::to_string(inputData.gradShape[1]);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "h-dim and w-dim of input_grad", shapeMsg.c_str(),
                "when global_pooling is true, grad shape in h-dim and w-dim must be 1 if the size of corresponding "
                "input shape is not zero");
            return ge::GRAPH_FAILED;
        }
        inputData.pad = {0, 0, 0, 0};
        inputData.stride = inputData.inputShape;
        inputData.kernelSize = inputData.inputShape;
        if (inputData.divisorOverride == 0) {
            inputData.divisorOverride = inputData.kernelSize[0] * inputData.kernelSize[1];
        }
    }
    if (CheckGradShape(context, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "grad_shape", "check_failed", "CheckGradShape validation failed");
        return ge::GRAPH_FAILED;
    }
    inputData.isInt32Meet = IsGreaterThanInt32Max(inputData, commInfo) ? 0 : ONE;
    inputData.hasDivisor = inputData.divisorOverride ? ONE : 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling