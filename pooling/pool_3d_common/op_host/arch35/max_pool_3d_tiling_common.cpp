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
 * \file max_pool_3d_tiling_common.cpp
 * \brief
 */

#include "exe_graph/runtime/shape.h"
#include "pool_tiling_templates_registry.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"
#include "max_pool_3d_tiling_common.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{
const int32_t INPUT_IDX_X = 0;

const int32_t KERNEL_POS = 0;
const int32_t STRIDE_POS = 1;
const int32_t PADDING_MODE_POS = 2;
const int32_t PADDING_POS = 3;
const int32_t DILATION_POS = 4;
const int32_t CEIL_POS = 5;
const int32_t FORMAT_POS = 6;

static const int32_t MP_MAX_3D_DIM_ZERO = 0;
static const int32_t MP_MAX_3D_DIM_ONE = 1;
static const int32_t MP_MAX_3D_DIM_TWO = 2;
static const int32_t MP_MAX_3D_DIM_THREE = 3;
static const int32_t MP_MAX_3D_DIM_FOUR = 4;

static const int32_t DIGIT_TWO = 2;

ge::graphStatus GetMaxPool3DPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum)
{
    auto platformPtr = context->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const MaxPool3DCompileInfo*>(context->GetCompileInfo());
        OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "compile info is null"),
                        return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;

        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNum();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<uint64_t>(ubSizePlatform);
    }
    OP_TILING_CHECK(coreNum == 0, CUBE_INNER_ERR_REPORT(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool IsInvalidType(const DataType dtype)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    bool dtypeInValid = (supportedDtype.count(dtype) == 0);
    return dtypeInValid;
}

static bool IsInvalidPaddingMode(std::string padMode)
{
    const std::set<std::string> supportedPadModelIST = {"CALCULATED", "SAME", "VALID"};
    bool padModeInValid = (supportedPadModelIST.count(padMode) == 0);
    return padModeInValid;
}

static ge::graphStatus CheckShape(gert::TilingContext* context_, gert::Shape& inputShape, gert::Shape& outputShape,
                                  const ge::Format& inputFormat)
{
    OP_TILING_CHECK(
        inputShape.GetDimNum() != NCDHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3D: input shape dim = %zu, should be equal 5",
                                        inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        outputShape.GetDimNum() != NCDHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3D: output shape dim = %zu, should be equal 5",
                                        outputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    if (inputShape.GetShapeSize() == 0 && outputShape.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }

    OP_TILING_CHECK(inputShape.GetShapeSize() <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "MaxPool3D: input shape size %ld less than zero failed",
                                                    inputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(outputShape.GetShapeSize() <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "MaxPool3D: output shape size %ld less than zero failed",
                                                    outputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);

    int32_t nDim = MP_MAX_3D_DIM_ZERO;
    int32_t cDim = MP_MAX_3D_DIM_ONE;
    if (inputFormat == ge::Format::FORMAT_NDHWC) {
        nDim = MP_MAX_3D_DIM_ZERO;
        cDim = MP_MAX_3D_DIM_FOUR;
    }
    OP_TILING_CHECK(inputShape.GetDim(nDim) != outputShape.GetDim(nDim),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "MaxPool3D: the size of dim-n should be equal in inputShape and \
outShape, but get input [%ld], output [%ld]",
                                                    inputShape.GetDim(nDim), outputShape.GetDim(nDim)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputShape.GetDim(cDim) != outputShape.GetDim(cDim),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "MaxPool3D: the size of dim-c should be equal in inputShape and \
outShape, but get input [%ld], output [%ld]",
                                                    inputShape.GetDim(cDim), outputShape.GetDim(cDim)),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context_, Pool3DInputInfo& inputData,
                                        MaxPool3DCommon& commInfo)
{
    auto inputX = context_->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = Ops::NN::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    auto outX = context_->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = Ops::NN::OpTiling::EnsureNotScalar(outX->GetStorageShape());
    auto inputDesc = context_->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto dtype = inputDesc->GetDataType();
    if (IsInvalidType(dtype)) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3D: invalid dtype %s",
                                        Ops::Base::ToString(dtype).c_str());
        return ge::GRAPH_FAILED;
    }
    inputData.dtypeSize = ge::GetSizeByDataType(dtype);
    OP_TILING_CHECK(inputData.dtypeSize <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "dtypeSize must be greater than 0, dtypeSize: %ld",
                                                    inputData.dtypeSize), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckShape(context_, inputShape, outShape, inputData.inputFormat) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3D: check shape failed"),
                    return ge::GRAPH_FAILED);
    if (inputData.inputFormat == ge::Format::FORMAT_NCDHW) {
        commInfo.nDim = MP_MAX_3D_DIM_ZERO;
        commInfo.cDim = MP_MAX_3D_DIM_ONE;
        commInfo.dDim = MP_MAX_3D_DIM_TWO;
        commInfo.hDim = MP_MAX_3D_DIM_THREE;
        commInfo.wDim = MP_MAX_3D_DIM_FOUR;
        inputData.batches = inputShape.GetDim(commInfo.nDim) * inputShape.GetDim(commInfo.cDim);
        inputData.channels = 1;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NDHWC) {
        commInfo.nDim = MP_MAX_3D_DIM_ZERO;
        commInfo.dDim = MP_MAX_3D_DIM_ONE;
        commInfo.hDim = MP_MAX_3D_DIM_TWO;
        commInfo.wDim = MP_MAX_3D_DIM_THREE;
        commInfo.cDim = MP_MAX_3D_DIM_FOUR;
        inputData.batches = inputShape.GetDim(commInfo.nDim);
        inputData.channels = inputShape.GetDim(commInfo.cDim);
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "MaxPool3D: only support NCDHW and NDHWC, not support format %s.",
                                        ge::TypeUtils::FormatToSerialString(inputData.inputFormat).c_str());
        return ge::GRAPH_FAILED;
    }
    inputData.inputShape = {inputShape.GetDim(commInfo.dDim), inputShape.GetDim(commInfo.hDim),
                            inputShape.GetDim(commInfo.wDim)};
    inputData.outShape = {outShape.GetDim(commInfo.dDim), outShape.GetDim(commInfo.hDim),
                          outShape.GetDim(commInfo.wDim)};
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDilationInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                                       Pool3DInputInfo& inputData, const MaxPool3DCommon& commInfo)
{
    auto dilation = runtimeAttrs->GetListInt(DILATION_POS);
    if (dilation == nullptr) {
        inputData.dilation = {1, 1, 1};
    } else {
        auto dilationDim = dilation->GetSize();
        OP_TILING_CHECK(
            dilationDim != ONE_DIMS && dilationDim != DHW_DIMS && dilationDim != NCDHW_DIMS,
            VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: dilation must have %d, %d, or %d elements ", ONE_DIMS,
                                            DHW_DIMS, NCDHW_DIMS),
            return ge::GRAPH_FAILED);

        int64_t dDilation = 1;
        int64_t hDilation = 1;
        int64_t wDilation = 1;
        if (dilationDim == ONE_DIMS) {
            dDilation = dilation->GetData()[MP_MAX_3D_DIM_ZERO];
            hDilation = dilation->GetData()[MP_MAX_3D_DIM_ZERO];
            wDilation = dilation->GetData()[MP_MAX_3D_DIM_ZERO];
        } else if (dilationDim == DHW_DIMS) {
            dDilation = dilation->GetData()[MP_MAX_3D_DIM_ZERO];
            hDilation = dilation->GetData()[MP_MAX_3D_DIM_ONE];
            wDilation = dilation->GetData()[MP_MAX_3D_DIM_TWO];
        } else if (dilationDim == NCDHW_DIMS) {
            dDilation = dilation->GetData()[commInfo.dDim];
            hDilation = dilation->GetData()[commInfo.hDim];
            wDilation = dilation->GetData()[commInfo.wDim];
        }
        OP_TILING_CHECK(dDilation <= 0 || hDilation <= 0 || wDilation <= 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "MaxPool3D: not support dilation shape [%ld, %ld, %ld]",
                                                        dDilation, hDilation, wDilation),
                        return ge::GRAPH_FAILED);
        inputData.dilation = {dDilation, hDilation, wDilation};
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetStrideInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                                     Pool3DInputInfo& inputData, const MaxPool3DCommon& commInfo)
{
    auto stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, stride);
    auto strideDim = stride->GetSize();
    OP_TILING_CHECK(strideDim != ONE_DIMS && strideDim != DHW_DIMS && strideDim != NCDHW_DIMS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: stride must have %d, %d, or %d elements ",
                                                    ONE_DIMS, DHW_DIMS, NCDHW_DIMS),
                    return ge::GRAPH_FAILED);

    int64_t dStride = 1;
    int64_t hStride = 1;
    int64_t wStride = 1;
    if (strideDim == ONE_DIMS) {
        dStride = stride->GetData()[MP_MAX_3D_DIM_ZERO];
        hStride = stride->GetData()[MP_MAX_3D_DIM_ZERO];
        wStride = stride->GetData()[MP_MAX_3D_DIM_ZERO];
    } else if (strideDim == DHW_DIMS) {
        dStride = stride->GetData()[MP_MAX_3D_DIM_ZERO];
        hStride = stride->GetData()[MP_MAX_3D_DIM_ONE];
        wStride = stride->GetData()[MP_MAX_3D_DIM_TWO];
    } else if (strideDim == NCDHW_DIMS) {
        dStride = stride->GetData()[commInfo.dDim];
        hStride = stride->GetData()[commInfo.hDim];
        wStride = stride->GetData()[commInfo.wDim];
        int64_t nStride = stride->GetData()[commInfo.nDim];
        int64_t cStride = stride->GetData()[commInfo.cDim];
        OP_TILING_CHECK(nStride != 1 || cStride != 1,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            context_->GetNodeName(),
                            "MaxPool3D: The stride of the N and C dimensions should be 1, not support [%ld, %ld]",
                            nStride, cStride),
                        return ge::GRAPH_FAILED);
    }
    inputData.stride = {dStride, hStride, wStride};
    OP_TILING_CHECK(
        dStride <= 0 || hStride <= 0 || wStride <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(),
            "MaxPool3D: The stride of the D, H and W dimensions should be greater than 0, not support [%ld, %ld, %ld]",
            dStride, hStride, wStride),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetKernelKsizeInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                                          Pool3DInputInfo& inputData, const MaxPool3DCommon& commInfo)
{
    auto kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    auto kSzieDim = kernelSize->GetSize();
    OP_TILING_CHECK(
        kSzieDim != ONE_DIMS && kSzieDim != DHW_DIMS && kSzieDim != NCDHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: kernel_size must have %d, %d, or %d elements ", ONE_DIMS,
                                        DHW_DIMS, NCDHW_DIMS),
        return ge::GRAPH_FAILED);
    int64_t dKernelSize = 1;
    int64_t hKernelSize = 1;
    int64_t wKernelSize = 1;
    if (kSzieDim == ONE_DIMS) {
        dKernelSize = kernelSize->GetData()[MP_MAX_3D_DIM_ZERO];
        hKernelSize = kernelSize->GetData()[MP_MAX_3D_DIM_ZERO];
        wKernelSize = kernelSize->GetData()[MP_MAX_3D_DIM_ZERO];
    } else if (kSzieDim == DHW_DIMS) {
        dKernelSize = kernelSize->GetData()[MP_MAX_3D_DIM_ZERO];
        hKernelSize = kernelSize->GetData()[MP_MAX_3D_DIM_ONE];
        wKernelSize = kernelSize->GetData()[MP_MAX_3D_DIM_TWO];
    } else if (kSzieDim == NCDHW_DIMS) {
        dKernelSize = kernelSize->GetData()[commInfo.dDim];
        hKernelSize = kernelSize->GetData()[commInfo.hDim];
        wKernelSize = kernelSize->GetData()[commInfo.wDim];
        int64_t nKernelSize = kernelSize->GetData()[commInfo.nDim];
        int64_t cKernelSize = kernelSize->GetData()[commInfo.cDim];
        OP_TILING_CHECK(nKernelSize != 1 || cKernelSize != 1,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            context_->GetNodeName(),
                            "MaxPool3D: The ksize of the N and C dimensions should be 1, not support [%ld, %ld]",
                            nKernelSize, cKernelSize),
                        return ge::GRAPH_FAILED);
    }
    inputData.kernelSize = {dKernelSize, hKernelSize, wKernelSize};
    OP_TILING_CHECK(
        dKernelSize <= 0 || hKernelSize <= 0 || wKernelSize <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(),
            "MaxPool3D: The ksize of the D, H and W dimensions should be greater than 0, not support [%ld, %ld, %ld]",
            dKernelSize, hKernelSize, wKernelSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPadInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                                  Pool3DInputInfo& inputData, const MaxPool3DCommon& commInfo)
{
    if (commInfo.padModeStr == "CALCULATED") {
        auto padding = runtimeAttrs->GetListInt(PADDING_POS);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, padding);
        OP_TILING_CHECK(padding->GetSize() != PAD_DIMS,
            VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: pad list must have %d elements ", PAD_DIMS),
            return ge::GRAPH_FAILED);
        int64_t frontPad = padding->GetData()[FRONT_PAD_INDEX];
        int64_t backendPad = padding->GetData()[BACKEND_PAD_INDEX];
        int64_t topPad = padding->GetData()[TOP_PAD_INDEX];
        int64_t bottomPad = padding->GetData()[BOTTOM_PAD_INDEX];
        int64_t leftPad = padding->GetData()[LEFT_PAD_INDEX];
        int64_t rightPad = padding->GetData()[RIGHT_PAD_INDEX];
        inputData.pad = {frontPad, backendPad, topPad, bottomPad, leftPad, rightPad};
        int64_t fixdkD = (inputData.kernelSize[D_DIM] - 1) * inputData.dilation[D_DIM] + 1;
        int64_t fixdkH = (inputData.kernelSize[H_DIM] - 1) * inputData.dilation[H_DIM] + 1;
        int64_t fixdKW = (inputData.kernelSize[W_DIM] - 1) * inputData.dilation[W_DIM] + 1;
        OP_TILING_CHECK(
            frontPad >= fixdkD || frontPad < 0 || backendPad >= fixdkD || backendPad < 0 || topPad >= fixdkH ||
                topPad < 0 || bottomPad >= fixdkH || bottomPad < 0 || leftPad >= fixdKW || leftPad < 0 ||
                rightPad >= fixdKW || rightPad < 0,
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                "MaxPool3D: not support pad shape [%ld, %ld, %ld, %ld, %ld, %ld] kernel shape [%ld, %ld, %ld], pad should \
be greater than or equal to 0 and smaller than the corresponding kernel size",
                frontPad, backendPad, topPad, bottomPad, leftPad, rightPad, fixdkD, fixdkH, fixdKW), return ge::GRAPH_FAILED);
    } else if (commInfo.padModeStr == "VALID") {
        inputData.pad = {0, 0, 0, 0, 0, 0};
    } else if (commInfo.padModeStr == "SAME") {
        int64_t dPadNeed = std::max(int64_t{0}, (inputData.outShape[D_DIM] - 1) * inputData.stride[D_DIM] +
                                                    (inputData.kernelSize[D_DIM] - 1) * inputData.dilation[D_DIM] + 1 -
                                                    inputData.inputShape[D_DIM]);
        int64_t frontPad = dPadNeed / DIGIT_TWO;
        int64_t backendPad = dPadNeed - frontPad;
        int64_t hPadNeed = std::max(int64_t{0}, (inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] +
                                                    (inputData.kernelSize[H_DIM] - 1) * inputData.dilation[H_DIM] + 1 -
                                                    inputData.inputShape[H_DIM]);
        int64_t topPad = hPadNeed / DIGIT_TWO;
        int64_t bottomPad = hPadNeed - topPad;
        int64_t wPadNeed = std::max(int64_t{0}, (inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] +
                                                    (inputData.kernelSize[W_DIM] - 1) * inputData.dilation[W_DIM] + 1 -
                                                    inputData.inputShape[W_DIM]);
        int64_t leftPad = wPadNeed / DIGIT_TWO;
        int64_t rightPad = wPadNeed - leftPad;
        inputData.pad = {frontPad, backendPad, topPad, bottomPad, leftPad, rightPad};
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: not support padmode %s", commInfo.padModeStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrsInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                                    Pool3DInputInfo& inputData, MaxPool3DCommon& commInfo)
{
    const char* padMode = runtimeAttrs->GetAttrPointer<char>(PADDING_MODE_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padMode);
    commInfo.padModeStr = padMode;
    OP_TILING_CHECK(
        IsInvalidPaddingMode(commInfo.padModeStr),
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: not support padmode %s", commInfo.padModeStr.c_str()),
        return ge::GRAPH_FAILED);
    inputData.ceilMode = false;
    const bool* ceilModePtr = runtimeAttrs->GetAttrPointer<bool>(CEIL_POS);
    if (ceilModePtr != nullptr) {
        inputData.ceilMode = *ceilModePtr;
    }

    std::string inputFormatStr("NDHWC");
    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    inputFormatStr = inputFormat;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputFormat);
    if (inputFormatStr == "NCDHW") {
        inputData.inputFormat = ge::Format::FORMAT_NCDHW;
    } else if (inputFormatStr == "NDHWC") {
        inputData.inputFormat = ge::Format::FORMAT_NDHWC;
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: only support NCDHW and NDHWC, not support format %s",
                                        inputFormatStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShapeForValid(gert::TilingContext* context_, Pool3DInputInfo& inputData)
{
    int64_t expectedD = (inputData.inputShape[D_DIM] - (inputData.kernelSize[D_DIM] - 1) * inputData.dilation[D_DIM] -
                         1 + inputData.stride[D_DIM]) /
                        inputData.stride[D_DIM];
    int64_t expectedH = (inputData.inputShape[H_DIM] - (inputData.kernelSize[H_DIM] - 1) * inputData.dilation[H_DIM] -
                         1 + inputData.stride[H_DIM]) /
                        inputData.stride[H_DIM];
    int64_t expectedW = (inputData.inputShape[W_DIM] - (inputData.kernelSize[W_DIM] - 1) * inputData.dilation[W_DIM] -
                         1 + inputData.stride[W_DIM]) /
                        inputData.stride[W_DIM];
    if (inputData.outShape[D_DIM] != expectedD || inputData.outShape[H_DIM] != expectedH ||
        inputData.outShape[W_DIM] != expectedW) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_,
                                        "MaxPool3D: when padmode is VALID, the outputshape in \
d-dim, h-dim and w-dim should be [%ld] [%ld] [%ld], but got [%ld] [%ld] [%ld]",
                                        expectedD, expectedH, expectedW, inputData.outShape[D_DIM],
                                        inputData.inputShape[H_DIM], inputData.inputShape[W_DIM]);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShapeForSame(gert::TilingContext* context_, Pool3DInputInfo& inputData)
{
    int64_t expectedD = (inputData.inputShape[D_DIM] + inputData.stride[D_DIM] - 1) / inputData.stride[D_DIM];
    int64_t expectedH = (inputData.inputShape[H_DIM] + inputData.stride[H_DIM] - 1) / inputData.stride[H_DIM];
    int64_t expectedW = (inputData.inputShape[W_DIM] + inputData.stride[W_DIM] - 1) / inputData.stride[W_DIM];
    if (inputData.outShape[D_DIM] != expectedD || inputData.outShape[H_DIM] != expectedH ||
        inputData.outShape[W_DIM] != expectedW) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_,
                                        "MaxPool3D: when padmode is SAME, the outputshape in \
d-dim, h-dim and w-dim should be [%ld] [%ld] [%ld], but got [%ld] [%ld] [%ld]",
                                        expectedD, expectedH, expectedW, inputData.outShape[D_DIM],
                                        inputData.outShape[H_DIM], inputData.outShape[W_DIM]);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

int64_t InferCalculateOutShape(const int64_t ksize, const int64_t padL, const int64_t padR, const int64_t stride,
                               const bool ceilMode, const int64_t dilation, int64_t dimSize)
{
    int64_t tmpTotalInput = dimSize + padL + padR - (ksize - 1) * dilation - 1;
    if (ceilMode) {
        tmpTotalInput += stride - 1;
    }
    int64_t outputSize = DivRtn(tmpTotalInput, stride) + 1;

    if (ceilMode) {
        if ((outputSize - 1) * stride >= dimSize + padL) {
            --outputSize;
        }
    }
    return outputSize;
}

static ge::graphStatus CheckOutPutShapeForCalculated(gert::TilingContext* context_, Pool3DInputInfo& inputData)
{
    int64_t expectedD = InferCalculateOutShape(
        inputData.kernelSize[D_DIM], inputData.pad[FRONT_PAD_INDEX], inputData.pad[BACKEND_PAD_INDEX],
        inputData.stride[D_DIM], inputData.ceilMode, inputData.dilation[D_DIM], inputData.inputShape[D_DIM]);

    int64_t expectedH = InferCalculateOutShape(
        inputData.kernelSize[H_DIM], inputData.pad[TOP_PAD_INDEX], inputData.pad[BOTTOM_PAD_INDEX],
        inputData.stride[H_DIM], inputData.ceilMode, inputData.dilation[H_DIM], inputData.inputShape[H_DIM]);

    int64_t expectedW = InferCalculateOutShape(
        inputData.kernelSize[W_DIM], inputData.pad[LEFT_PAD_INDEX], inputData.pad[RIGHT_PAD_INDEX],
        inputData.stride[W_DIM], inputData.ceilMode, inputData.dilation[W_DIM], inputData.inputShape[W_DIM]);
    if (inputData.outShape[D_DIM] != expectedD || inputData.outShape[H_DIM] != expectedH ||
        inputData.outShape[W_DIM] != expectedW) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_,
                                        "MaxPool3D: when padmode is CALCULATED, the outputshape in \
d-dim, h-dim and w-dim should be [%ld] [%ld] [%ld], but got [%ld] [%ld] [%ld]",
                                        expectedD, expectedH, expectedW, inputData.outShape[D_DIM],
                                        inputData.outShape[H_DIM], inputData.outShape[W_DIM]);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShape(gert::TilingContext* context_, Pool3DInputInfo& inputData,
                                        const MaxPool3DCommon& commInfo)
{
    if (commInfo.padModeStr == "CALCULATED") {
        return CheckOutPutShapeForCalculated(context_, inputData);
    } else if (commInfo.padModeStr == "VALID") {
        return CheckOutPutShapeForValid(context_, inputData);
    } else if (commInfo.padModeStr == "SAME") {
        return CheckOutPutShapeForSame(context_, inputData);
    }
    return ge::GRAPH_SUCCESS;
}

static void RefineShape(Pool3DInputInfo& inputData)
{
    if (inputData.outShape[D_DIM] == 1 && inputData.dilation[D_DIM] == 1) {
        inputData.kernelSize[D_DIM] =
            std::min(inputData.kernelSize[D_DIM] - inputData.pad[FRONT_PAD_INDEX], inputData.inputShape[D_DIM]);
        inputData.pad[FRONT_PAD_INDEX] = 0;
        inputData.pad[BACKEND_PAD_INDEX] = 0;
        inputData.stride[D_DIM] = inputData.inputShape[D_DIM];
    }

    if (inputData.outShape[H_DIM] == 1 && inputData.dilation[H_DIM] == 1) {
        inputData.kernelSize[H_DIM] =
            std::min(inputData.kernelSize[H_DIM] - inputData.pad[TOP_PAD_INDEX], inputData.inputShape[H_DIM]);
        inputData.pad[TOP_PAD_INDEX] = 0;
        inputData.pad[BOTTOM_PAD_INDEX] = 0;
        inputData.stride[H_DIM] = inputData.inputShape[H_DIM];
    }

    if (inputData.outShape[W_DIM] == 1 && inputData.dilation[W_DIM] == 1) {
        inputData.kernelSize[W_DIM] =
            std::min(inputData.kernelSize[W_DIM] - inputData.pad[LEFT_PAD_INDEX], inputData.inputShape[W_DIM]);
        inputData.pad[LEFT_PAD_INDEX] = 0;
        inputData.pad[RIGHT_PAD_INDEX] = 0;
        inputData.stride[W_DIM] = inputData.inputShape[W_DIM];
    }
}

ge::graphStatus GetMaxPool3DShapeAttrsInfo(gert::TilingContext* context_, Pool3DInputInfo& inputData)
{
    inputData.poolMode = OP_TYPE_MAX_POOL_3D;
    auto runtimeAttrs = context_->GetAttrs();
    MaxPool3DCommon commInfo;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    OP_TILING_CHECK(GetAttrsInfo(context_, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetAttrsInfo fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetShapeAndDtype(context_, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetShapeAndDtype fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetKernelKsizeInfo(context_, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetKernelKsizeInfo fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetStrideInfo(context_, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetStrideInfo fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetDilationInfo(context_, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetDilationInfo fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetPadInfo(context_, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetPadInfo fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutPutShape(context_, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "CheckOutPutShape fail."), return ge::GRAPH_FAILED);
    RefineShape(inputData);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling