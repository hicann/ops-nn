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
 * \file avg_pool_3d_tiling_common.cpp
 * \brief
 */

#include "exe_graph/runtime/shape.h"
#include "pool_tiling_templates_registry.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "avg_pool_3d_tiling_common.h"
#include "pooling/avg_pool3_d/op_host/avg_pool_cube_tiling.h"

using namespace AscendC;
using namespace ge;

 
namespace optiling
{
static const int32_t INPUT_IDX_X = 0;

static const int32_t KERNEL_POS = 0;
static const int32_t STRIDE_POS = 1;
static const int32_t PADDING_POS = 2;
static const int32_t CEIL_POS = 3;
static const int32_t COUNT_INCLUDE_PAD_POS = 4;
static const int32_t DIVISOR_OVERRIDE_POS = 5;
static const int32_t FORMAT_POS = 6;

static const int32_t MP_AVG_3D_DIM_ZERO = 0;
static const int32_t MP_AVG_3D_DIM_ONE = 1;
static const int32_t MP_AVG_3D_DIM_TWO = 2;
static const int32_t MP_AVG_3D_DIM_THREE = 3;
static const int32_t MP_AVG_3D_DIM_FOUR = 4;

static const gert::Shape g_vec_1_shape = {1};

static const gert::Shape& EnsureNotScalar(const gert::Shape &inShape) {
  if (inShape.IsScalar()) {
    return g_vec_1_shape;
  }
  return inShape;
}

static bool IsInvalidType(const DataType dtype)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    bool dtypeInValid = (supportedDtype.count(dtype) == 0);
    return dtypeInValid;
}

static ge::graphStatus CheckShape(gert::TilingContext* context_, gert::Shape& inputShape, gert::Shape& outputShape,
                        const ge::Format& inputFormat)
{
    OP_TILING_CHECK(
        inputShape.GetDimNum() != NCDHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "AvgPool3D: input shape dim = %zu, should be equal 5",
                                        inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        outputShape.GetDimNum() != NCDHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "AvgPool3D: output shape dim = %zu, should be equal 5",
                                        outputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    if (inputShape.GetShapeSize() == 0 && outputShape.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }

    OP_TILING_CHECK(inputShape.GetShapeSize() <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "AvgPool3D: input shape size %ld less than zero failed",
                                                    inputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(outputShape.GetShapeSize() <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "AvgPool3D: output shape size %ld less than zero failed",
                                                    outputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);

    int32_t nDim = MP_AVG_3D_DIM_ZERO;
    int32_t cDim = MP_AVG_3D_DIM_ONE;
    if (inputFormat == ge::Format::FORMAT_NDHWC) {
        nDim = MP_AVG_3D_DIM_ZERO;
        cDim = MP_AVG_3D_DIM_FOUR;
    }
    OP_TILING_CHECK(
        inputShape.GetDim(nDim) != outputShape.GetDim(nDim),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "AvgPool3D: the size of dim-n should be equal in inputShape and \
outShape, but get input [%ld], output [%ld]",
                                        inputShape.GetDim(nDim), outputShape.GetDim(nDim)),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        inputShape.GetDim(cDim) != outputShape.GetDim(cDim),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "AvgPool3D: the size of dim-c should be equal in inputShape and \
outShape, but get input [%ld], output [%ld]",
                                        inputShape.GetDim(cDim), outputShape.GetDim(cDim)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
 
static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context_, Pool3DInputInfo& inputData, AvgPool3DCommon& commInfo)
{
    auto inputX = context_->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    auto outX = context_->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = EnsureNotScalar(outX->GetStorageShape());
    auto inputDesc = context_->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto dtype = inputDesc->GetDataType();
    if (IsInvalidType(dtype)) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "AvgPool3D: invalid dtype");
        return ge::GRAPH_FAILED;
    }
    inputData.dtypeSize = ge::GetSizeByDataType(dtype);
    OP_TILING_CHECK(
        inputData.dtypeSize <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "inputData.dtypeSize must be greater than 0, dtypeSize: %ld", inputData.dtypeSize),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckShape(context_, inputShape, outShape, inputData.inputFormat) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "AvgPool3D: check shape failed"),
                    return ge::GRAPH_FAILED);
    if (inputData.inputFormat == ge::Format::FORMAT_NCDHW) {
        commInfo.nDim = MP_AVG_3D_DIM_ZERO;
        commInfo.cDim = MP_AVG_3D_DIM_ONE;
        commInfo.dDim = MP_AVG_3D_DIM_TWO;
        commInfo.hDim = MP_AVG_3D_DIM_THREE;
        commInfo.wDim = MP_AVG_3D_DIM_FOUR;
        inputData.batches = inputShape.GetDim(commInfo.nDim) * inputShape.GetDim(commInfo.cDim);
        inputData.channels = 1;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NDHWC) {
        commInfo.nDim = MP_AVG_3D_DIM_ZERO;
        commInfo.dDim = MP_AVG_3D_DIM_ONE;
        commInfo.hDim = MP_AVG_3D_DIM_TWO;
        commInfo.wDim = MP_AVG_3D_DIM_THREE;
        commInfo.cDim = MP_AVG_3D_DIM_FOUR;
        inputData.batches = inputShape.GetDim(commInfo.nDim);
        inputData.channels = inputShape.GetDim(commInfo.cDim);
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "AvgPool3D: only support NCDHW and NDHWC, not support format.");
        return ge::GRAPH_FAILED;
    }
    inputData.inputShape = {inputShape.GetDim(commInfo.dDim), inputShape.GetDim(commInfo.hDim),
                            inputShape.GetDim(commInfo.wDim)};
    inputData.outShape = {outShape.GetDim(commInfo.dDim), outShape.GetDim(commInfo.hDim),
                        outShape.GetDim(commInfo.wDim)};
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetStrideInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                                    Pool3DInputInfo& inputData, const AvgPool3DCommon& commInfo)
{
    auto stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, stride);
    auto strideDim = stride->GetSize();
    OP_TILING_CHECK(strideDim != ONE_DIMS && strideDim != DHW_DIMS && strideDim != NCDHW_DIMS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "AvgPool3D: stride must have %d, %d, or %d elements ",
                                                    ONE_DIMS, DHW_DIMS, NCDHW_DIMS),
                    return ge::GRAPH_FAILED);

    int64_t dStride = 1;
    int64_t hStride = 1;
    int64_t wStride = 1;
    if (strideDim == ONE_DIMS) {
        dStride = stride->GetData()[MP_AVG_3D_DIM_ZERO];
        hStride = stride->GetData()[MP_AVG_3D_DIM_ZERO];
        wStride = stride->GetData()[MP_AVG_3D_DIM_ZERO];
    } else if (strideDim == DHW_DIMS) {
        dStride = stride->GetData()[MP_AVG_3D_DIM_ZERO];
        hStride = stride->GetData()[MP_AVG_3D_DIM_ONE];
        wStride = stride->GetData()[MP_AVG_3D_DIM_TWO];
    } else if (strideDim == NCDHW_DIMS) {
        dStride = stride->GetData()[commInfo.dDim];
        hStride = stride->GetData()[commInfo.hDim];
        wStride = stride->GetData()[commInfo.wDim];
    }
    inputData.stride = {dStride, hStride, wStride};
    OP_TILING_CHECK(
        dStride <= 0 || hStride <= 0 || wStride <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(),
            "AvgPool3D: The stride of the D, H and W dimensions should be greater than 0, not support [%ld, %ld, %ld]",
            dStride, hStride, wStride),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetKernelKsizeInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                                Pool3DInputInfo& inputData,const AvgPool3DCommon& commInfo)
{
    auto kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    auto kSzieDim = kernelSize->GetSize();
    OP_TILING_CHECK(
        kSzieDim != ONE_DIMS && kSzieDim != DHW_DIMS && kSzieDim != NCDHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "AvgPool3D: kernel_size must have %d, %d, or %d elements ", ONE_DIMS,
                                        DHW_DIMS, NCDHW_DIMS),
        return ge::GRAPH_FAILED);
    int64_t dKernelSize = 1;
    int64_t hKernelSize = 1;
    int64_t wKernelSize = 1;
    if (kSzieDim == ONE_DIMS) {
        dKernelSize = kernelSize->GetData()[MP_AVG_3D_DIM_ZERO];
        hKernelSize = kernelSize->GetData()[MP_AVG_3D_DIM_ZERO];
        wKernelSize = kernelSize->GetData()[MP_AVG_3D_DIM_ZERO];
    } else if (kSzieDim == DHW_DIMS) {
        dKernelSize = kernelSize->GetData()[MP_AVG_3D_DIM_ZERO];
        hKernelSize = kernelSize->GetData()[MP_AVG_3D_DIM_ONE];
        wKernelSize = kernelSize->GetData()[MP_AVG_3D_DIM_TWO];
    } else if (kSzieDim == NCDHW_DIMS) {
        dKernelSize = kernelSize->GetData()[commInfo.dDim];
        hKernelSize = kernelSize->GetData()[commInfo.hDim];
        wKernelSize = kernelSize->GetData()[commInfo.wDim];
    }
    inputData.kernelSize = {dKernelSize, hKernelSize, wKernelSize};
    OP_TILING_CHECK(
        dKernelSize <= 0 || hKernelSize <= 0 || wKernelSize <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(),
            "AvgPool3D: The ksize of the D, H and W dimensions should be greater than 0, not support [%ld, %ld, %ld]",
            dKernelSize, hKernelSize, wKernelSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPadInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                        Pool3DInputInfo& inputData, const AvgPool3DCommon& commInfo)
{
    auto padding = runtimeAttrs->GetListInt(PADDING_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padding);
    OP_TILING_CHECK(
        padding->GetSize() != PAD_DIMS && padding->GetSize() != ONE_DIMS && padding->GetSize() != DHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "AvgPool3D: pad list must have 1 dim 3 dims and 6 dims, \
                                        but got %d dims", static_cast<int32_t>(padding->GetSize())), return ge::GRAPH_FAILED);
    int64_t frontPad = 0;
    int64_t backendPad = 0;
    int64_t topPad = 0;
    int64_t bottomPad = 0;
    int64_t leftPad = 0;
    int64_t rightPad = 0;
    if(padding->GetSize() == ONE_DIMS) {
        frontPad = padding->GetData()[FRONT_PAD_INDEX];
        backendPad = frontPad;
        topPad = frontPad;
        bottomPad = frontPad;
        leftPad = frontPad;
        rightPad = frontPad;
    } else if (padding->GetSize() == DHW_DIMS) {
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
    OP_TILING_CHECK(
        frontPad >= inputData.kernelSize[D_DIM] || frontPad < 0 || backendPad >= inputData.kernelSize[D_DIM] ||
            backendPad < 0 || topPad >= inputData.kernelSize[H_DIM] || topPad < 0 ||
            bottomPad >= inputData.kernelSize[H_DIM] || bottomPad < 0 || leftPad >= inputData.kernelSize[W_DIM] ||
            leftPad < 0 || rightPad >= inputData.kernelSize[W_DIM] || rightPad < 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(),
            "AvgPool3D: not support pad shape [%ld, %ld, %ld, %ld, %ld, %ld] kernel shape [%ld, %ld, %ld], pad should \
be greater than or equal to 0 and smaller than the corresponding kernel size",
            frontPad, backendPad, topPad, bottomPad, leftPad, rightPad, inputData.kernelSize[D_DIM],
            inputData.kernelSize[H_DIM], inputData.kernelSize[W_DIM]),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrsInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                            Pool3DInputInfo& inputData, AvgPool3DCommon& commInfo)
{
    inputData.ceilMode = false;
    const bool* ceilModePtr = runtimeAttrs->GetAttrPointer<bool>(CEIL_POS);
    if (ceilModePtr != nullptr) {
        inputData.ceilMode = *ceilModePtr;
    }

    inputData.countIncludePad = true;
    const bool* countPadPtr = runtimeAttrs->GetAttrPointer<bool>(COUNT_INCLUDE_PAD_POS);
    if (countPadPtr != nullptr) {
        inputData.countIncludePad = *countPadPtr;
    }

    inputData.divisorOverride = 0;
    const int64_t* divisorPtr = runtimeAttrs->GetAttrPointer<int64_t>(DIVISOR_OVERRIDE_POS);
    if (divisorPtr != nullptr) {
        inputData.divisorOverride = *divisorPtr;
    }

    inputData.dilation = {1, 1, 1};
    inputData.poolMode = OP_TYPE_AVG_POOL_3D;

    std::string inputFormatStr("NDHWC");
    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    if (inputFormat != nullptr) {
        inputFormatStr = inputFormat;
    }
    if (inputFormatStr == "NCDHW") {
        inputData.inputFormat = ge::Format::FORMAT_NCDHW;
    } else if (inputFormatStr == "NDHWC") {
        inputData.inputFormat = ge::Format::FORMAT_NDHWC;
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "AvgPool3D: only support NCDHW and NDHWC, not support format %s",
                                        inputFormatStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

int64_t InferCalculateOutShape(const int64_t ksize, const int64_t padL, const int64_t padR,
    const int64_t stride, const bool ceilMode, int64_t dimSize)
{
    int64_t tmpTotalInput = dimSize + padL + padR - ksize;
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

ge::graphStatus CheckOutPutShape(gert::TilingContext* context_, Pool3DInputInfo& inputData, const AvgPool3DCommon& commInfo)
{
    int64_t expectedD = 0;
    int64_t expectedH = 0;
    int64_t expectedW = 0;

    expectedD = InferCalculateOutShape(inputData.kernelSize[D_DIM], inputData.pad[FRONT_PAD_INDEX], 
                                      inputData.pad[BACKEND_PAD_INDEX], inputData.stride[D_DIM], 
                                      inputData.ceilMode, inputData.inputShape[D_DIM]);
    expectedH = InferCalculateOutShape(inputData.kernelSize[H_DIM], inputData.pad[TOP_PAD_INDEX], 
                                      inputData.pad[BOTTOM_PAD_INDEX], inputData.stride[H_DIM], 
                                      inputData.ceilMode, inputData.inputShape[H_DIM]);
    expectedW = InferCalculateOutShape(inputData.kernelSize[W_DIM], inputData.pad[LEFT_PAD_INDEX], 
                                      inputData.pad[RIGHT_PAD_INDEX], inputData.stride[W_DIM], 
                                      inputData.ceilMode, inputData.inputShape[W_DIM]);
    if (expectedD != inputData.outShape[D_DIM] || expectedH != inputData.outShape[H_DIM] || expectedW != inputData.outShape[W_DIM]) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_, 
                                       "AvgPool3D: output shape in d-dim h-dim and w-dim should be [%ld, %ld, %ld], but got [%ld, %ld, %ld]",
                                        expectedD, expectedH, expectedW, inputData.outShape[D_DIM],
                                        inputData.outShape[H_DIM], inputData.outShape[W_DIM]);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void RefineShape(Pool3DInputInfo& inputData)
{
    if (inputData.outShape[D_DIM] == 1) {
        if (inputData.kernelSize[D_DIM] >= inputData.inputShape[D_DIM] + inputData.pad[FRONT_PAD_INDEX]) {
            inputData.kernelSize[D_DIM] = inputData.inputShape[D_DIM];
            inputData.pad[FRONT_PAD_INDEX] = 0;
            inputData.pad[BACKEND_PAD_INDEX] = 0;
        } else {
            inputData.kernelSize[D_DIM] = inputData.kernelSize[D_DIM] - inputData.pad[FRONT_PAD_INDEX];
            inputData.pad[FRONT_PAD_INDEX] = 0;
            inputData.pad[BACKEND_PAD_INDEX] = 0;
        }
        inputData.stride[D_DIM] = inputData.kernelSize[D_DIM];
    }

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

ge::graphStatus GetAvgPool3DShapeAttrsInfo(gert::TilingContext* context_, Pool3DInputInfo& inputData)
{
    auto runtimeAttrs = context_->GetAttrs();
    AvgPool3DCommon commInfo;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    OP_TILING_CHECK(GetAttrsInfo(context_, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetAttrsInfo fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetShapeAndDtype(context_, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetShapeAndDtype fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetKernelKsizeInfo(context_, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetKernelKsizeInfo fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetStrideInfo(context_, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetStrideInfo fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetPadInfo(context_, runtimeAttrs, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "GetPadInfo fail."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutPutShape(context_, inputData, commInfo) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "CheckOutPutShape fail."), return ge::GRAPH_FAILED);
    
    if (!inputData.divisorOverride || !inputData.countIncludePad) {
        RefineShape(inputData);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAvgPool3DPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum)
{
    auto platformPtr = context->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const optiling::avgPool3DTilingCompileInfo::CubeCompileInfo *>(context->GetCompileInfo());
        OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "compile info is null"),
                        return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->core_num;

        ubSize = compileInfoPtr->ub_size;
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

}  // namespace optiling