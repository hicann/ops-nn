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
 * \file avg_pool_v2_common_tiling.cpp
 * \brief
 */
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "log/log.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "avg_pool_v2_common_tiling.h"

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

static const int32_t MP_AVG_2D_DIM_ZERO = 0;
static const int32_t MP_AVG_2D_DIM_ONE = 1;
static const int32_t MP_AVG_2D_DIM_TWO = 2;
static const int32_t MP_AVG_2D_DIM_THREE = 3;

static const int32_t DIGIT_TWO = 2;

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

static ge::graphStatus CheckShape(gert::TilingContext* context, gert::Shape& inputShape, gert::Shape& outputShape,
                                  const ge::Format& inputFormat)
{
    if (inputShape.GetDimNum() != NCHW_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                 std::to_string(inputShape.GetDimNum()).c_str(), "shape dim must be 4");
        return ge::GRAPH_FAILED;
    }
    if (outputShape.GetDimNum() != NCHW_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "y", std::to_string(outputShape.GetDimNum()).c_str(), "shape dim must be 4");
        return ge::GRAPH_FAILED;
    }
    if (inputShape.GetShapeSize() == 0 && outputShape.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (inputShape.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "x", std::to_string(inputShape.GetShapeSize()).c_str(), "shape size must be > 0");
        return ge::GRAPH_FAILED;
    }

    if (outputShape.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "y", std::to_string(outputShape.GetShapeSize()).c_str(), "shape size must be > 0");
        return ge::GRAPH_FAILED;
    }

    int32_t nDim = MP_AVG_2D_DIM_ZERO;
    int32_t cDim = MP_AVG_2D_DIM_ONE;
    if (inputFormat == ge::Format::FORMAT_NHWC) {
        nDim = MP_AVG_2D_DIM_ZERO;
        cDim = MP_AVG_2D_DIM_THREE;
    }
    if (inputShape.GetDim(nDim) != outputShape.GetDim(nDim)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x, y",
            (std::to_string(inputShape.GetDim(nDim)) + ", " + std::to_string(outputShape.GetDim(nDim))).c_str(),
            "n-dim must be equal in x and y shapes");
        return ge::GRAPH_FAILED;
    }

    if (inputShape.GetDim(cDim) != outputShape.GetDim(cDim)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x, y",
            (std::to_string(inputShape.GetDim(cDim)) + ", " + std::to_string(outputShape.GetDim(cDim))).c_str(),
            "c-dim must be equal in x and y shapes");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context, AvgPoolInputInfo& inputData,
                                        AvgPoolV2Common& commInfo)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShape = Ops::NN::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    auto outX = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outX);
    auto outShape = Ops::NN::OpTiling::EnsureNotScalar(outX->GetStorageShape());
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dtype = inputDesc->GetDataType();
    if (IsInvalidType(dtype)) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", std::to_string(static_cast<int32_t>(dtype)).c_str(),
                                  "[DT_FLOAT, DT_FLOAT16, DT_BF16]");
        return ge::GRAPH_FAILED;
    }
    inputData.dtypeSize = ge::GetSizeByDataType(dtype);
    if (inputData.dtypeSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dtypeSize",
                                              std::to_string(inputData.dtypeSize).c_str(), "dtypeSize must be > 0");
        return ge::GRAPH_FAILED;
    }
    if (CheckShape(context, inputShape, outShape, inputData.inputFormat) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "x", std::to_string(inputShape.GetDimNum()).c_str(), "shape validation failed");
        return ge::GRAPH_FAILED;
    }
    if (inputData.inputFormat == ge::Format::FORMAT_NCHW) {
        commInfo.nDim = MP_AVG_2D_DIM_ZERO;
        commInfo.cDim = MP_AVG_2D_DIM_ONE;
        commInfo.hDim = MP_AVG_2D_DIM_TWO;
        commInfo.wDim = MP_AVG_2D_DIM_THREE;
        inputData.batches = inputShape.GetDim(commInfo.nDim) * inputShape.GetDim(commInfo.cDim);
        inputData.channels = 1;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NHWC) {
        commInfo.nDim = MP_AVG_2D_DIM_ZERO;
        commInfo.hDim = MP_AVG_2D_DIM_ONE;
        commInfo.wDim = MP_AVG_2D_DIM_TWO;
        commInfo.cDim = MP_AVG_2D_DIM_THREE;
        inputData.batches = inputShape.GetDim(commInfo.nDim);
        inputData.channels = inputShape.GetDim(commInfo.cDim);
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x",
                                   std::to_string(static_cast<int32_t>(inputData.inputFormat)).c_str(), "NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }
    inputData.inputShape[H_DIM] = inputShape.GetDim(commInfo.hDim);
    inputData.inputShape[W_DIM] = inputShape.GetDim(commInfo.wDim);
    inputData.outShape[H_DIM] = outShape.GetDim(commInfo.hDim);
    inputData.outShape[W_DIM] = outShape.GetDim(commInfo.wDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetStrideInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                     AvgPoolInputInfo& inputData, const AvgPoolV2Common& commInfo)
{
    auto stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, stride);
    auto strideDim = stride->GetSize();
    if (strideDim != ONE_DIMS && strideDim != HW_DIMS && strideDim != NCHW_DIMS) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "stride", std::to_string(strideDim).c_str(), "1, 2, or 4");
        return ge::GRAPH_FAILED;
    }

    int64_t hStride = 1;
    int64_t wStride = 1;
    if (strideDim == ONE_DIMS) {
        hStride = stride->GetData()[MP_AVG_2D_DIM_ZERO];
        wStride = stride->GetData()[MP_AVG_2D_DIM_ZERO];
    } else if (strideDim == HW_DIMS) {
        hStride = stride->GetData()[MP_AVG_2D_DIM_ZERO];
        wStride = stride->GetData()[MP_AVG_2D_DIM_ONE];
    } else if (strideDim == NCHW_DIMS) {
        hStride = stride->GetData()[commInfo.hDim];
        wStride = stride->GetData()[commInfo.wDim];
    }
    inputData.stride[H_DIM] = hStride;
    inputData.stride[W_DIM] = wStride;
    if (hStride <= 0 || wStride <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "hStride, wStride",
                                               (std::to_string(hStride) + ", " + std::to_string(wStride)).c_str(),
                                               "stride values must be > 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetKernelKsizeInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                          AvgPoolInputInfo& inputData, const AvgPoolV2Common& commInfo)
{
    auto kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, kernelSize);
    auto kSizeDim = kernelSize->GetSize();
    if (kSizeDim != ONE_DIMS && kSizeDim != HW_DIMS && kSizeDim != NCHW_DIMS) {
        OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "kernel_size", std::to_string(kSizeDim).c_str(),
                                     "1, 2, or 4");
        return ge::GRAPH_FAILED;
    }
    int64_t hKernelSize = 1;
    int64_t wKernelSize = 1;
    if (kSizeDim == ONE_DIMS) {
        hKernelSize = kernelSize->GetData()[MP_AVG_2D_DIM_ZERO];
        wKernelSize = kernelSize->GetData()[MP_AVG_2D_DIM_ZERO];
    } else if (kSizeDim == HW_DIMS) {
        hKernelSize = kernelSize->GetData()[MP_AVG_2D_DIM_ZERO];
        wKernelSize = kernelSize->GetData()[MP_AVG_2D_DIM_ONE];
    } else if (kSizeDim == NCHW_DIMS) {
        hKernelSize = kernelSize->GetData()[commInfo.hDim];
        wKernelSize = kernelSize->GetData()[commInfo.wDim];
    }
    inputData.kernelSize[H_DIM] = hKernelSize;
    inputData.kernelSize[W_DIM] = wKernelSize;

    if (hKernelSize <= 0 || wKernelSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "hKernelSize, wKernelSize",
            (std::to_string(hKernelSize) + ", " + std::to_string(wKernelSize)).c_str(),
            "kernel size values must be > 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPadInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                  AvgPoolInputInfo& inputData, const AvgPoolV2Common& commInfo)
{
    if (commInfo.padModeStr == "VALID") {
        inputData.pad[TOP_PAD_INDEX] = 0;
        inputData.pad[BOTTOM_PAD_INDEX] = 0;
        inputData.pad[LEFT_PAD_INDEX] = 0;
        inputData.pad[RIGHT_PAD_INDEX] = 0; // top, bottom, left, right
        if (inputData.ceilMode) {
            int64_t bottomPad = std::max(int64_t{0}, (inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] +
                                                         inputData.kernelSize[H_DIM] - inputData.inputShape[H_DIM]);
            int64_t rightPad = std::max(int64_t{0}, (inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] +
                                                        inputData.kernelSize[W_DIM] - inputData.inputShape[W_DIM]);
            inputData.pad[TOP_PAD_INDEX] = 0;
            inputData.pad[BOTTOM_PAD_INDEX] = bottomPad;
            inputData.pad[LEFT_PAD_INDEX] = 0;
            inputData.pad[RIGHT_PAD_INDEX] = rightPad;
        }
    } else if (commInfo.padModeStr == "SAME") {
        int64_t hPadNeed = std::max(int64_t{0}, (inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] +
                                                    inputData.kernelSize[H_DIM] - inputData.inputShape[H_DIM]);
        int64_t topPad = hPadNeed / DIGIT_TWO;
        int64_t bottomPad = hPadNeed - topPad;

        int64_t wPadNeed = std::max(int64_t{0}, (inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] +
                                                    inputData.kernelSize[W_DIM] - inputData.inputShape[W_DIM]);
        int64_t leftPad = wPadNeed / DIGIT_TWO;
        int64_t rightPad = wPadNeed - leftPad;

        inputData.pad[TOP_PAD_INDEX] = topPad;
        inputData.pad[BOTTOM_PAD_INDEX] = bottomPad;
        inputData.pad[LEFT_PAD_INDEX] = leftPad;
        inputData.pad[RIGHT_PAD_INDEX] = rightPad;
    } else if (commInfo.padModeStr == "CALCULATED") {
        auto padding = runtimeAttrs->GetListInt(PADS_POS);
        OP_CHECK_NULL_WITH_CONTEXT(context, padding);
        OP_CHECK_IF(padding->GetSize() != 4,
                    OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "pad",
                                                 std::to_string(padding->GetSize()).c_str(), "4"),
                    return ge::GRAPH_FAILED);
        int64_t topPad = padding->GetData()[TOP_PAD_INDEX];
        int64_t bottomPad = padding->GetData()[BOTTOM_PAD_INDEX];
        int64_t leftPad = padding->GetData()[LEFT_PAD_INDEX];
        int64_t rightPad = padding->GetData()[RIGHT_PAD_INDEX];
        OP_CHECK_IF(
            topPad >= inputData.kernelSize[H_DIM] || bottomPad >= inputData.kernelSize[H_DIM],
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "topPad, bottomPad",
                                                   (std::to_string(topPad) + ", " + std::to_string(bottomPad)).c_str(),
                                                   "topPad and bottomPad must be less than H kernel_size"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            leftPad >= inputData.kernelSize[W_DIM] || rightPad >= inputData.kernelSize[W_DIM],
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "leftPad, rightPad",
                                                   (std::to_string(leftPad) + ", " + std::to_string(rightPad)).c_str(),
                                                   "leftPad and rightPad must be less than W kernel_size"),
            return ge::GRAPH_FAILED);
        inputData.pad[TOP_PAD_INDEX] = topPad;
        inputData.pad[BOTTOM_PAD_INDEX] = bottomPad;
        inputData.pad[LEFT_PAD_INDEX] = leftPad;
        inputData.pad[RIGHT_PAD_INDEX] = rightPad;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "padding_mode", commInfo.padModeStr.c_str(),
                                              "padding_mode must be VALID, SAME or CALCULATED");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrsInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                    AvgPoolInputInfo& inputData, AvgPoolV2Common& commInfo)
{
    const char* padMode = runtimeAttrs->GetAttrPointer<char>(PADDING_MODE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, padMode);
    commInfo.padModeStr = padMode;
    if (IsInvalidPaddingMode(commInfo.padModeStr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "padding_mode", commInfo.padModeStr.c_str(),
                                              "padding_mode must be SAME, VALID or CALCULATED");
        return ge::GRAPH_FAILED;
    }
    const bool* globalPooling = runtimeAttrs->GetAttrPointer<bool>(GLOBAL_POOLING_POS);
    if (globalPooling != nullptr) {
        inputData.globalPooling = *globalPooling;
    }
    const bool* ceilMode = runtimeAttrs->GetAttrPointer<bool>(CEIL_MODE_POS);
    if (ceilMode != nullptr) {
        inputData.ceilMode = *ceilMode;
    }
    const bool* countIncludePad = runtimeAttrs->GetAttrPointer<bool>(EXCLUSIVE_POS);
    if (countIncludePad != nullptr) {
        inputData.countIncludePad = !(*countIncludePad);
    }
    const int64_t* divisorOverride = runtimeAttrs->GetAttrPointer<int64_t>(DIVISOR_OVERRIDE_POS);
    if (divisorOverride != nullptr) {
        inputData.divisorOverride = *divisorOverride;
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
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x", inputFormatStr.c_str(), "NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShapeForValid(gert::TilingContext* context, AvgPoolInputInfo& inputData)
{
    int64_t expectedH = (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + inputData.stride[H_DIM]) /
                        inputData.stride[H_DIM];
    int64_t expectedW = (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + inputData.stride[W_DIM]) /
                        inputData.stride[W_DIM];
    if (inputData.ceilMode) {
        expectedH = (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + inputData.stride[H_DIM] - 1) /
                        inputData.stride[H_DIM] +
                    1;
        expectedW = (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + inputData.stride[W_DIM] - 1) /
                        inputData.stride[W_DIM] +
                    1;
    }
    if (inputData.outShape[H_DIM] != expectedH || inputData.outShape[W_DIM] != expectedW) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "y",
            (std::to_string(inputData.outShape[H_DIM]) + ", " + std::to_string(inputData.outShape[W_DIM])).c_str(),
            "output shape mismatch under VALID padding");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShapeForSame(gert::TilingContext* context, AvgPoolInputInfo& inputData)
{
    int64_t expectedH = (inputData.inputShape[H_DIM] + inputData.stride[H_DIM] - 1) / inputData.stride[H_DIM];
    int64_t expectedW = (inputData.inputShape[W_DIM] + inputData.stride[W_DIM] - 1) / inputData.stride[W_DIM];
    if (inputData.outShape[H_DIM] != expectedH || inputData.outShape[W_DIM] != expectedW) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "y",
            (std::to_string(inputData.outShape[H_DIM]) + ", " + std::to_string(inputData.outShape[W_DIM])).c_str(),
            "output shape mismatch under SAME padding");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShapeForCalculated(gert::TilingContext* context, AvgPoolInputInfo& inputData)
{
    int64_t topPad = inputData.pad[TOP_PAD_INDEX];
    int64_t bottomPad = inputData.pad[BOTTOM_PAD_INDEX];
    int64_t leftPad = inputData.pad[LEFT_PAD_INDEX];
    int64_t rightPad = inputData.pad[RIGHT_PAD_INDEX];
    int64_t expectedH = (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + topPad + bottomPad) /
                            inputData.stride[H_DIM] +
                        1;
    int64_t expectedW = (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + leftPad + rightPad) /
                            inputData.stride[W_DIM] +
                        1;
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
    if (inputData.outShape[H_DIM] != expectedH || inputData.outShape[W_DIM] != expectedW) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "y",
            (std::to_string(inputData.outShape[H_DIM]) + ", " + std::to_string(inputData.outShape[W_DIM])).c_str(),
            "output shape mismatch under CALCULATED padding");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShape(gert::TilingContext* context, AvgPoolInputInfo& inputData,
                                        const AvgPoolV2Common& commInfo)
{
    if (commInfo.padModeStr == "VALID") {
        return CheckOutPutShapeForValid(context, inputData);
    } else if (commInfo.padModeStr == "SAME") {
        return CheckOutPutShapeForSame(context, inputData);
    } else if (commInfo.padModeStr == "CALCULATED") {
        return CheckOutPutShapeForCalculated(context, inputData);
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "padding_mode", commInfo.padModeStr.c_str(),
                                          "padding_mode must be VALID, SAME or CALCULATED");
    return ge::GRAPH_FAILED;
}

static void RefineShape(AvgPoolInputInfo& inputData)
{
    if (inputData.outShape[H_DIM] == 1) {
        if (inputData.kernelSize[H_DIM] >= inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX]) {
            inputData.kernelSize[H_DIM] = inputData.inputShape[H_DIM];
            inputData.pad[TOP_PAD_INDEX] = 0;
            inputData.pad[BOTTOM_PAD_INDEX] = 0;
        } else {
            inputData.kernelSize[H_DIM] = inputData.kernelSize[H_DIM] - inputData.pad[TOP_PAD_INDEX];
            inputData.pad[TOP_PAD_INDEX] = 0;
            inputData.pad[BOTTOM_PAD_INDEX] = 0;
        }
        inputData.stride[H_DIM] = inputData.kernelSize[H_DIM];
    }
    if (inputData.outShape[W_DIM] == 1) {
        if (inputData.kernelSize[W_DIM] >= inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX]) {
            inputData.kernelSize[W_DIM] = inputData.inputShape[W_DIM];
            inputData.pad[LEFT_PAD_INDEX] = 0;
            inputData.pad[RIGHT_PAD_INDEX] = 0;
        } else {
            inputData.kernelSize[W_DIM] = inputData.kernelSize[W_DIM] - inputData.pad[LEFT_PAD_INDEX];
            inputData.pad[LEFT_PAD_INDEX] = 0;
            inputData.pad[RIGHT_PAD_INDEX] = 0;
        }
        inputData.stride[W_DIM] = inputData.kernelSize[W_DIM];
    }
}

ge::graphStatus GetAvgPoolV2ShapeAttrsInfo(gert::TilingContext* context, AvgPoolInputInfo& inputData)
{
    auto runtimeAttrs = context->GetAttrs();
    AvgPoolV2Common commInfo;
    OP_CHECK_NULL_WITH_CONTEXT(context, runtimeAttrs);
    if (GetAttrsInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attrs", "check_failed", "GetAttrsInfo failed");
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAndDtype(context, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "shape_and_dtype", "check_failed",
                                              "GetShapeAndDtype failed");
        return ge::GRAPH_FAILED;
    }

    if (GetKernelKsizeInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "kernel_size", "check_failed",
                                              "GetKernelKsizeInfo failed");
        return ge::GRAPH_FAILED;
    }

    if (GetStrideInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "stride", "check_failed", "GetStrideInfo failed");
        return ge::GRAPH_FAILED;
    }

    if (GetPadInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "pad", "check_failed", "GetPadInfo failed");
        return ge::GRAPH_FAILED;
    }

    if (inputData.globalPooling) {
        if ((inputData.inputShape[0] != 0 && inputData.outShape[0] != 1) ||
            (inputData.inputShape[1] != 0 && inputData.outShape[1] != 1)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "inputShape[h_dim], inputShape[w_dim]",
                (std::to_string(inputData.inputShape[0]) + ", " + std::to_string(inputData.inputShape[1])).c_str(),
                "global_pooling=true requires outputShape[h]=1 and outputShape[w]=1 when input is non-zero");
            return ge::GRAPH_FAILED;
        }
        inputData.pad[TOP_PAD_INDEX] = 0;
        inputData.pad[BOTTOM_PAD_INDEX] = 0;
        inputData.pad[LEFT_PAD_INDEX] = 0;
        inputData.pad[RIGHT_PAD_INDEX] = 0;
        inputData.stride = inputData.inputShape;
        inputData.kernelSize = inputData.inputShape;
        if (inputData.divisorOverride == 0) {
            inputData.divisorOverride = inputData.kernelSize[0] * inputData.kernelSize[1];
        }
        return ge::GRAPH_SUCCESS;
    }
    if (CheckOutPutShape(context, inputData, commInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output_shape", "check_failed",
                                              "CheckOutPutShape failed");
        return ge::GRAPH_FAILED;
    }

    if (inputData.divisorOverride) {
        RefineShape(inputData);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAvgPoolV2PlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum)
{
    auto platformPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    ubSize = static_cast<uint64_t>(ubSizePlatform);

    OP_CHECK_IF(coreNum == 0, CUBE_INNER_ERR_REPORT(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling