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
 * \file avg_pool_tiling_common.cpp
 * \brief
 */
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "util/platform_util.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "avg_pool_tiling_common.h"

using namespace AscendC;
using namespace ge;
 
namespace optiling
{
static const int32_t INPUT_IDX_X = 0;

static const int32_t KERNEL_POS = 0;
static const int32_t STRIDE_POS = 1;
static const int32_t PADDING_POS = 2;
static const int32_t FORMAT_POS = 3;

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
    const std::set<std::string> supportedPadModeList = {"SAME", "VALID"};
    bool padModeInValid = (supportedPadModeList.count(padMode) == 0);
    return padModeInValid;
}

static ge::graphStatus CheckShape(gert::TilingContext* context, gert::Shape& inputShape, gert::Shape& outputShape,
                        const ge::Format& inputFormat)
{
    OP_CHECK_IF(
        inputShape.GetDimNum() != NCHW_DIMS,
        OP_LOGE(context->GetNodeName(), "AvgPool: input shape dim = %zu, should be equal 4",
                                        inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outputShape.GetDimNum() != NCHW_DIMS,
        OP_LOGE(context->GetNodeName(), "AvgPool: output shape dim = %zu, should be equal 4",
                                        outputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    if (inputShape.GetShapeSize() == 0 && outputShape.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(inputShape.GetShapeSize() <= 0,
                    OP_LOGE(context->GetNodeName(),
                                                    "AvgPool: input shape size %ld less than zero failed",
                                                    inputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF(outputShape.GetShapeSize() <= 0,
                    OP_LOGE(context->GetNodeName(),
                                                    "AvgPool: output shape size %ld less than zero failed",
                                                    outputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);

    int32_t nDim = MP_AVG_2D_DIM_ZERO;
    int32_t cDim = MP_AVG_2D_DIM_ONE;
    if (inputFormat == ge::Format::FORMAT_NHWC) {
        nDim = MP_AVG_2D_DIM_ZERO;
        cDim = MP_AVG_2D_DIM_THREE;
    }
    OP_CHECK_IF(
        inputShape.GetDim(nDim) != outputShape.GetDim(nDim),
        OP_LOGE(context->GetNodeName(),
                                        "AvgPool: the size of dim-n should be equal in inputShape and outShape, but get input [%ld], output [%ld]",
            inputShape.GetDim(nDim), outputShape.GetDim(nDim)),
        return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(
        inputShape.GetDim(cDim) != outputShape.GetDim(cDim),
        OP_LOGE(context->GetNodeName(),
                                        "AvgPool: the size of dim-c should be equal in inputShape and outShape, but get input [%ld], output [%ld]",
                                        inputShape.GetDim(cDim), outputShape.GetDim(cDim)),
        return ge::GRAPH_FAILED);  
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context, AvgPoolInputInfo& inputData, AvgPoolCommon& commInfo)
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
        OP_LOGE(context->GetNodeName(), "AvgPool: invalid dtype");
        return ge::GRAPH_FAILED;
    }
    inputData.dtypeSize = ge::GetSizeByDataType(dtype);
    OP_CHECK_IF(
        inputData.dtypeSize <= 0,
        OP_LOGE(context, "inputData.dtypeSize must be greater than 0, dtypeSize: %ld", inputData.dtypeSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape(context, inputShape, outShape, inputData.inputFormat) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "AvgPool: check shape failed"),
                    return ge::GRAPH_FAILED);
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
        OP_LOGE(context->GetNodeName(),
                                        "AvgPool: only support NCHW and NHWC, not support format.");
        return ge::GRAPH_FAILED;
    }
    inputData.inputShape[H_DIM] = inputShape.GetDim(commInfo.hDim);
    inputData.inputShape[W_DIM] = inputShape.GetDim(commInfo.wDim);
    inputData.outShape[H_DIM] = outShape.GetDim(commInfo.hDim);
    inputData.outShape[W_DIM] = outShape.GetDim(commInfo.wDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetStrideInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                    AvgPoolInputInfo& inputData, const AvgPoolCommon& commInfo)
{
    auto stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, stride);
    auto strideDim = stride->GetSize();
    OP_CHECK_IF(strideDim != ONE_DIMS && strideDim != HW_DIMS && strideDim != NCHW_DIMS,
                    OP_LOGE(context, "AvgPool: stride must have %d, %d, or %d elements ",
                                                    ONE_DIMS, HW_DIMS, NCHW_DIMS),
                    return ge::GRAPH_FAILED);

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
    inputData.stride = {hStride, wStride};
    OP_CHECK_IF(hStride <= 0 || wStride <= 0,
                    OP_LOGE(context->GetNodeName(),
                        "AvgPool: The stride of the H and W dimensions should be greater than 0, not support [%ld, %ld]",
                         hStride, wStride),
                    return ge::GRAPH_FAILED); 
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetKernelKsizeInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                AvgPoolInputInfo& inputData, const AvgPoolCommon& commInfo)
{
    auto kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, kernelSize);
    auto kSizeDim = kernelSize->GetSize();
    OP_CHECK_IF(
        kSizeDim != ONE_DIMS && kSizeDim != HW_DIMS && kSizeDim != NCHW_DIMS,
        OP_LOGE(context, "AvgPool: kernel_size must have %d, %d, or %d elements ", 
                                        ONE_DIMS, HW_DIMS, NCHW_DIMS),
        return ge::GRAPH_FAILED);
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
    inputData.kernelSize = {hKernelSize, wKernelSize};
    
    OP_CHECK_IF(hKernelSize <= 0 || wKernelSize <= 0,
                    OP_LOGE(context->GetNodeName(),
                        "AvgPool: The ksize of the H and W dimensions should be greater than 0, not support [%ld, %ld]",
                        hKernelSize, wKernelSize),
                    return ge::GRAPH_FAILED); 
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPadInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                                  AvgPoolInputInfo& inputData, const AvgPoolCommon& commInfo)
{
    if (commInfo.padModeStr == "VALID") {
        inputData.pad = {0, 0, 0, 0};  // top, bottom, left, right
    } else if (commInfo.padModeStr == "SAME") {
        int64_t hPadNeed = std::max(int64_t{0}, (inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] +
                                                inputData.kernelSize[H_DIM] - inputData.inputShape[H_DIM]);
        int64_t topPad = hPadNeed / DIGIT_TWO;
        int64_t bottomPad = hPadNeed - topPad;
        
        int64_t wPadNeed = std::max(int64_t{0}, (inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] +
                                                inputData.kernelSize[W_DIM] - inputData.inputShape[W_DIM]);
        int64_t leftPad = wPadNeed / DIGIT_TWO;
        int64_t rightPad = wPadNeed - leftPad;
        
        inputData.pad = {topPad, bottomPad, leftPad, rightPad};
    } else {
        OP_LOGE(context, "AvgPool: not support padmode %s", commInfo.padModeStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrsInfo(gert::TilingContext* context, const gert::RuntimeAttrs* runtimeAttrs,
                            AvgPoolInputInfo& inputData, AvgPoolCommon& commInfo)
{
    const char* padMode = runtimeAttrs->GetAttrPointer<char>(PADDING_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, padMode);
    commInfo.padModeStr = padMode;
    OP_CHECK_IF(
        IsInvalidPaddingMode(commInfo.padModeStr),
        OP_LOGE(context, "AvgPool: not support padmode %s", commInfo.padModeStr.c_str()),
        return ge::GRAPH_FAILED);
    inputData.ceilMode = false;
    inputData.countIncludePad = false;
    inputData.divisorOverride = 0;
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
        OP_LOGE(context, 
            "AvgPool: only support NCHW and NHWC, not support format %s",
            inputFormatStr.c_str());
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
    if (inputData.outShape[H_DIM] != expectedH || inputData.outShape[W_DIM] != expectedW) {
        OP_LOGE(context,
                                        "AvgPool: when padmode is VALID, the outputshape in \
h-dim and w-dim should be [%ld] [%ld], but got [%ld] [%ld]",
                                        expectedH, expectedW, inputData.outShape[H_DIM],
                                        inputData.outShape[W_DIM]);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShapeForSame(gert::TilingContext* context, AvgPoolInputInfo& inputData)
{
    int64_t expectedH = (inputData.inputShape[H_DIM] + inputData.stride[H_DIM] - 1) / inputData.stride[H_DIM];
    int64_t expectedW = (inputData.inputShape[W_DIM] + inputData.stride[W_DIM] - 1) / inputData.stride[W_DIM];
    if (inputData.outShape[H_DIM] != expectedH || inputData.outShape[W_DIM] != expectedW) {
        OP_LOGE(context,
                                        "AvgPool: when padmode is SAME, the outputshape in \
h-dim and w-dim should be [%ld] [%ld], but got [%ld] [%ld]",
                                        expectedH, expectedW, inputData.outShape[H_DIM],
                                        inputData.outShape[W_DIM]);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShape(gert::TilingContext* context, AvgPoolInputInfo& inputData,
                                                 const AvgPoolCommon& commInfo)
{
    if (commInfo.padModeStr == "VALID") {
        return CheckOutPutShapeForValid(context, inputData);
    } else if (commInfo.padModeStr == "SAME") {
        return CheckOutPutShapeForSame(context, inputData);
    }
    OP_LOGE(context, 
                                    "AvgPool: unsupported pad mode [%s], only VALID and SAME are supported", 
                                    commInfo.padModeStr.c_str());
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

ge::graphStatus GetAvgPoolShapeAttrsInfo(gert::TilingContext* context, AvgPoolInputInfo& inputData)
{
    auto runtimeAttrs = context->GetAttrs();
    AvgPoolCommon commInfo;
    OP_CHECK_NULL_WITH_CONTEXT(context, runtimeAttrs);
    OP_CHECK_IF(GetAttrsInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "GetAttrsInfo fail."), 
                    return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(GetShapeAndDtype(context, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "GetShapeAndDtype fail."), 
                    return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(GetKernelKsizeInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "GetKernelKsizeInfo fail."), 
                    return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(GetStrideInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "GetStrideInfo fail."), 
                    return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(GetPadInfo(context, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "GetPadInfo fail."), 
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutPutShape(context, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "CheckOutPutShape fail."), 
                    return ge::GRAPH_FAILED);

    if (!inputData.divisorOverride || !inputData.countIncludePad) {
        RefineShape(inputData);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAvgPoolPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum)
{
    auto platformPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatform;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    ubSize = static_cast<uint64_t>(ubSizePlatform);
    
    OP_CHECK_IF(coreNum == 0, 
                    CUBE_INNER_ERR_REPORT(context, "coreNum is 0"), 
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling