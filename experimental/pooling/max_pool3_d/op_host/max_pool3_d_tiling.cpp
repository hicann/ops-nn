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
 * \file max_pool3_d_tiling.cpp
 * \brief MaxPool3D independent AscendC tiling.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include "exe_graph/runtime/shape.h"
#include "error_util.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "util/platform_util.h"
#include "../op_kernel/max_pool3_d_tiling_data.h"
#include "../op_kernel/max_pool3_d_tiling_key.h"

namespace optiling {
constexpr int32_t DHW_DIMS = 3;
constexpr int32_t PAD_DIMS = 6;
constexpr int32_t ONE_DIMS = 1;
constexpr int32_t NCDHW_DIMS = 5;
constexpr int32_t NDC1HWC0_DIMS = 6;

constexpr uint32_t D_DIM = 0;
constexpr uint32_t H_DIM = 1;
constexpr uint32_t W_DIM = 2;
constexpr uint32_t FRONT_PAD_INDEX = 0;
constexpr uint32_t BACKEND_PAD_INDEX = 1;
constexpr uint32_t TOP_PAD_INDEX = 2;
constexpr uint32_t BOTTOM_PAD_INDEX = 3;
constexpr uint32_t LEFT_PAD_INDEX = 4;
constexpr uint32_t RIGHT_PAD_INDEX = 5;

struct Pool3DInputInfo {
    std::array<int64_t, DHW_DIMS> inputShape{};
    std::array<int64_t, DHW_DIMS> outShape{};
    std::array<int64_t, DHW_DIMS> kernelSize{};
    std::array<int64_t, DHW_DIMS> stride{};
    std::array<int64_t, PAD_DIMS> pad{};
    std::array<int64_t, DHW_DIMS> dilation{};
    bool ceilMode = false;
    ge::Format inputFormat = ge::Format::FORMAT_RESERVED;
    int64_t dtypeSize = 0;
    bool isBfloat16 = false;
};

struct MaxPool3DCommon {
    int64_t nDim = 0;
    int64_t cDim = 0;
    int64_t dDim = 0;
    int64_t hDim = 0;
    int64_t wDim = 0;
    std::string padModeStr;
};

struct MaxPool3DCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

BEGIN_TILING_DATA_DEF(MaxPool3DTilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalOut);
TILING_DATA_FIELD_DEF(uint64_t, normalCoreOut);
TILING_DATA_FIELD_DEF(uint64_t, splitOut);
TILING_DATA_FIELD_DEF(uint64_t, splitQuantum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, inD);
TILING_DATA_FIELD_DEF(int64_t, inH);
TILING_DATA_FIELD_DEF(int64_t, inW);
TILING_DATA_FIELD_DEF(int64_t, c);
TILING_DATA_FIELD_DEF(int64_t, outD);
TILING_DATA_FIELD_DEF(int64_t, outH);
TILING_DATA_FIELD_DEF(int64_t, outW);
TILING_DATA_FIELD_DEF(int64_t, kD);
TILING_DATA_FIELD_DEF(int64_t, kH);
TILING_DATA_FIELD_DEF(int64_t, kW);
TILING_DATA_FIELD_DEF(int64_t, sD);
TILING_DATA_FIELD_DEF(int64_t, sH);
TILING_DATA_FIELD_DEF(int64_t, sW);
TILING_DATA_FIELD_DEF(int64_t, padFront);
TILING_DATA_FIELD_DEF(int64_t, padTop);
TILING_DATA_FIELD_DEF(int64_t, padLeft);
TILING_DATA_FIELD_DEF(int64_t, dilationD);
TILING_DATA_FIELD_DEF(int64_t, dilationH);
TILING_DATA_FIELD_DEF(int64_t, dilationW);
TILING_DATA_FIELD_DEF(uint32_t, dataFormat);
TILING_DATA_FIELD_DEF(uint32_t, outputLayout);
TILING_DATA_FIELD_DEF(int64_t, outputD);
TILING_DATA_FIELD_DEF(int64_t, outputH);
TILING_DATA_FIELD_DEF(int64_t, outputW);
TILING_DATA_FIELD_DEF(int64_t, outputC1);
TILING_DATA_FIELD_DEF(int64_t, outputC0);
TILING_DATA_FIELD_DEF(int64_t, outputC0Block);
TILING_DATA_FIELD_DEF(uint32_t, inputLayout);
TILING_DATA_FIELD_DEF(int64_t, inputC1);
TILING_DATA_FIELD_DEF(int64_t, inputC0);
TILING_DATA_FIELD_DEF(int64_t, inputC0Block);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint32_t, balancedSplit);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool3D, MaxPool3DTilingData);
} // namespace optiling
using namespace ge;

namespace optiling {
constexpr int32_t KERNEL_POS = 0;
constexpr int32_t STRIDE_POS = 1;
constexpr int32_t PADDING_MODE_POS = 2;
constexpr int32_t PADDING_POS = 3;
constexpr int32_t DILATION_POS = 4;
constexpr int32_t CEIL_POS = 5;
constexpr int32_t FORMAT_POS = 6;

constexpr int32_t MP_MAX_3D_DIM_ZERO = 0;
constexpr int32_t MP_MAX_3D_DIM_ONE = 1;
constexpr int32_t MP_MAX_3D_DIM_TWO = 2;
constexpr int32_t MP_MAX_3D_DIM_THREE = 3;
constexpr int32_t MP_MAX_3D_DIM_FOUR = 4;
constexpr int32_t DIGIT_TWO = 2;

static ge::graphStatus GetMaxPool3DPlatformInfo(gert::TilingContext* context, uint64_t& coreNum)
{
    if (context == nullptr) {
        OP_LOGE("MaxPool3D", "Tiling context is null.");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0U;
    auto platformPtr = context->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const MaxPool3DCompileInfo*>(context->GetCompileInfo());
        OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context, "compile info is null"),
                        return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;

        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform = 0U;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = ubSizePlatform;
    }
    OP_TILING_CHECK(coreNum == 0, CUBE_INNER_ERR_REPORT(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(ubSize == 0, CUBE_INNER_ERR_REPORT(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool IsInvalidType(const DataType dtype)
{
    return dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16;
}

static bool IsInvalidPaddingMode(const std::string& padMode)
{
    return padMode != "CALCULATED" && padMode != "SAME" && padMode != "VALID";
}

static ge::graphStatus CheckShapeSize(gert::TilingContext* context_, const gert::Shape& inputShape,
                                      const gert::Shape& outputShape)
{
    OP_TILING_CHECK(
        inputShape.GetDimNum() != NCDHW_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3D: input shape dim = %zu, should be equal 5",
                                        inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        outputShape.GetDimNum() != NCDHW_DIMS && outputShape.GetDimNum() != NDC1HWC0_DIMS,
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3D: output shape dim = %zu, should be 5 or 6",
                                        outputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    if (inputShape.GetShapeSize() == 0 && outputShape.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }

    OP_TILING_CHECK(inputShape.GetShapeSize() <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "MaxPool3D: input shape size %ld must be greater than zero",
                                                    inputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(outputShape.GetShapeSize() <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "MaxPool3D: output shape size %ld must be greater than zero",
                                                    outputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeChannels(gert::TilingContext* context_, const gert::Shape& inputShape,
                                          const gert::Shape& outputShape, const ge::Format& inputFormat)
{
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
    if (outputShape.GetDimNum() == NDC1HWC0_DIMS) {
        const int64_t outputChannels = outputShape.GetDim(MP_MAX_3D_DIM_TWO) * outputShape.GetDim(5);
        OP_TILING_CHECK(inputShape.GetDim(cDim) > outputChannels,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "MaxPool3D: NDC1HWC0 output channel capacity should cover "
                                                        "input C, but get input [%ld], output capacity [%ld]",
                                                        inputShape.GetDim(cDim), outputChannels),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(inputShape.GetDim(cDim) != outputShape.GetDim(cDim),
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                        "MaxPool3D: the size of dim-c should be equal in inputShape "
                                                        "and outShape, but get input [%ld], output [%ld]",
                                                        inputShape.GetDim(cDim), outputShape.GetDim(cDim)),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(gert::TilingContext* context_, gert::Shape& inputShape, gert::Shape& outputShape,
                                  const ge::Format& inputFormat)
{
    const ge::graphStatus sizeStatus = CheckShapeSize(context_, inputShape, outputShape);
    if (sizeStatus != ge::GRAPH_SUCCESS || (inputShape.GetShapeSize() == 0 && outputShape.GetShapeSize() == 0)) {
        return sizeStatus;
    }
    return CheckShapeChannels(context_, inputShape, outputShape, inputFormat);
}

static ge::graphStatus SetCommonDimIndices(gert::TilingContext* context_, const Pool3DInputInfo& inputData,
                                           MaxPool3DCommon& commInfo)
{
    if (inputData.inputFormat == ge::Format::FORMAT_NCDHW) {
        commInfo.nDim = MP_MAX_3D_DIM_ZERO;
        commInfo.cDim = MP_MAX_3D_DIM_ONE;
        commInfo.dDim = MP_MAX_3D_DIM_TWO;
        commInfo.hDim = MP_MAX_3D_DIM_THREE;
        commInfo.wDim = MP_MAX_3D_DIM_FOUR;
        return ge::GRAPH_SUCCESS;
    }
    if (inputData.inputFormat == ge::Format::FORMAT_NDHWC) {
        commInfo.nDim = MP_MAX_3D_DIM_ZERO;
        commInfo.dDim = MP_MAX_3D_DIM_ONE;
        commInfo.hDim = MP_MAX_3D_DIM_TWO;
        commInfo.wDim = MP_MAX_3D_DIM_THREE;
        commInfo.cDim = MP_MAX_3D_DIM_FOUR;
        return ge::GRAPH_SUCCESS;
    }
    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                    "MaxPool3D: only support NCDHW and NDHWC, not support format %s.",
                                    ge::TypeUtils::FormatToSerialString(inputData.inputFormat).c_str());
    return ge::GRAPH_FAILED;
}

static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context_, Pool3DInputInfo& inputData,
                                        MaxPool3DCommon& commInfo)
{
    auto inputX = context_->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = Ops::NN::OpTiling::EnsureNotScalar(inputX->GetOriginShape());
    if (inputShape.GetDimNum() != NCDHW_DIMS) {
        inputShape = Ops::NN::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    }
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
    inputData.isBfloat16 = dtype == ge::DT_BF16;
    OP_TILING_CHECK(inputData.dtypeSize <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "dtypeSize must be greater than 0, dtypeSize: %ld",
                                                    inputData.dtypeSize),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckShape(context_, inputShape, outShape, inputData.inputFormat) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3D: check shape failed"),
                    return ge::GRAPH_FAILED);
    const ge::graphStatus indexStatus = SetCommonDimIndices(context_, inputData, commInfo);
    if (indexStatus != ge::GRAPH_SUCCESS) {
        return indexStatus;
    }
    inputData.inputShape = {inputShape.GetDim(commInfo.dDim), inputShape.GetDim(commInfo.hDim),
                            inputShape.GetDim(commInfo.wDim)};
    if (outShape.GetDimNum() == NDC1HWC0_DIMS) {
        inputData.outShape = {outShape.GetDim(1), outShape.GetDim(3), outShape.GetDim(4)};
    } else {
        inputData.outShape = {outShape.GetDim(commInfo.dDim), outShape.GetDim(commInfo.hDim),
                              outShape.GetDim(commInfo.wDim)};
    }
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
        const int64_t* dilationData = dilation->GetData();
        OPS_CHECK_NULL_WITH_CONTEXT(context_, dilationData);

        int64_t dDilation = 1;
        int64_t hDilation = 1;
        int64_t wDilation = 1;
        if (dilationDim == ONE_DIMS) {
            dDilation = dilationData[MP_MAX_3D_DIM_ZERO];
            hDilation = dilationData[MP_MAX_3D_DIM_ZERO];
            wDilation = dilationData[MP_MAX_3D_DIM_ZERO];
        } else if (dilationDim == DHW_DIMS) {
            dDilation = dilationData[MP_MAX_3D_DIM_ZERO];
            hDilation = dilationData[MP_MAX_3D_DIM_ONE];
            wDilation = dilationData[MP_MAX_3D_DIM_TWO];
        } else if (dilationDim == NCDHW_DIMS) {
            dDilation = dilationData[commInfo.dDim];
            hDilation = dilationData[commInfo.hDim];
            wDilation = dilationData[commInfo.wDim];
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
    const int64_t* strideData = stride->GetData();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, strideData);

    int64_t dStride = 1;
    int64_t hStride = 1;
    int64_t wStride = 1;
    if (strideDim == ONE_DIMS) {
        dStride = strideData[MP_MAX_3D_DIM_ZERO];
        hStride = strideData[MP_MAX_3D_DIM_ZERO];
        wStride = strideData[MP_MAX_3D_DIM_ZERO];
    } else if (strideDim == DHW_DIMS) {
        dStride = strideData[MP_MAX_3D_DIM_ZERO];
        hStride = strideData[MP_MAX_3D_DIM_ONE];
        wStride = strideData[MP_MAX_3D_DIM_TWO];
    } else if (strideDim == NCDHW_DIMS) {
        dStride = strideData[commInfo.dDim];
        hStride = strideData[commInfo.hDim];
        wStride = strideData[commInfo.wDim];
        const int64_t nStride = strideData[commInfo.nDim];
        const int64_t cStride = strideData[commInfo.cDim];
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
    const int64_t* kernelSizeData = kernelSize->GetData();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, kernelSizeData);
    int64_t dKernelSize = 1;
    int64_t hKernelSize = 1;
    int64_t wKernelSize = 1;
    if (kSzieDim == ONE_DIMS) {
        dKernelSize = kernelSizeData[MP_MAX_3D_DIM_ZERO];
        hKernelSize = kernelSizeData[MP_MAX_3D_DIM_ZERO];
        wKernelSize = kernelSizeData[MP_MAX_3D_DIM_ZERO];
    } else if (kSzieDim == DHW_DIMS) {
        dKernelSize = kernelSizeData[MP_MAX_3D_DIM_ZERO];
        hKernelSize = kernelSizeData[MP_MAX_3D_DIM_ONE];
        wKernelSize = kernelSizeData[MP_MAX_3D_DIM_TWO];
    } else if (kSzieDim == NCDHW_DIMS) {
        dKernelSize = kernelSizeData[commInfo.dDim];
        hKernelSize = kernelSizeData[commInfo.hDim];
        wKernelSize = kernelSizeData[commInfo.wDim];
        const int64_t nKernelSize = kernelSizeData[commInfo.nDim];
        const int64_t cKernelSize = kernelSizeData[commInfo.cDim];
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

static ge::graphStatus GetCalculatedPadInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                                            Pool3DInputInfo& inputData)
{
    auto padding = runtimeAttrs->GetListInt(PADDING_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padding);
    OP_TILING_CHECK(padding->GetSize() != PAD_DIMS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: pad list must have %d elements ", PAD_DIMS),
                    return ge::GRAPH_FAILED);
    const int64_t* paddingData = padding->GetData();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, paddingData);
    const int64_t frontPad = paddingData[FRONT_PAD_INDEX];
    const int64_t backendPad = paddingData[BACKEND_PAD_INDEX];
    const int64_t topPad = paddingData[TOP_PAD_INDEX];
    const int64_t bottomPad = paddingData[BOTTOM_PAD_INDEX];
    const int64_t leftPad = paddingData[LEFT_PAD_INDEX];
    const int64_t rightPad = paddingData[RIGHT_PAD_INDEX];
    inputData.pad = {frontPad, backendPad, topPad, bottomPad, leftPad, rightPad};
    OP_TILING_CHECK(frontPad < 0 || backendPad < 0 || topPad < 0 || bottomPad < 0 || leftPad < 0 || rightPad < 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "MaxPool3D: not support pad shape [%ld, %ld, %ld, %ld, %ld, %ld], "
                                                    "pad should be greater than or equal to 0",
                                                    frontPad, backendPad, topPad, bottomPad, leftPad, rightPad),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetSamePadInfo(Pool3DInputInfo& inputData)
{
    const int64_t dPadNeed = std::max(int64_t{0}, (inputData.outShape[D_DIM] - 1) * inputData.stride[D_DIM] +
                                                      (inputData.kernelSize[D_DIM] - 1) * inputData.dilation[D_DIM] +
                                                      1 - inputData.inputShape[D_DIM]);
    const int64_t frontPad = dPadNeed / DIGIT_TWO;
    const int64_t backendPad = dPadNeed - frontPad;
    const int64_t hPadNeed = std::max(int64_t{0}, (inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] +
                                                      (inputData.kernelSize[H_DIM] - 1) * inputData.dilation[H_DIM] +
                                                      1 - inputData.inputShape[H_DIM]);
    const int64_t topPad = hPadNeed / DIGIT_TWO;
    const int64_t bottomPad = hPadNeed - topPad;
    const int64_t wPadNeed = std::max(int64_t{0}, (inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] +
                                                      (inputData.kernelSize[W_DIM] - 1) * inputData.dilation[W_DIM] +
                                                      1 - inputData.inputShape[W_DIM]);
    const int64_t leftPad = wPadNeed / DIGIT_TWO;
    const int64_t rightPad = wPadNeed - leftPad;
    inputData.pad = {frontPad, backendPad, topPad, bottomPad, leftPad, rightPad};
}

static ge::graphStatus GetPadInfo(gert::TilingContext* context_, const gert::RuntimeAttrs* runtimeAttrs,
                                  Pool3DInputInfo& inputData, const MaxPool3DCommon& commInfo)
{
    if (commInfo.padModeStr == "CALCULATED") {
        return GetCalculatedPadInfo(context_, runtimeAttrs, inputData);
    }
    if (commInfo.padModeStr == "VALID") {
        inputData.pad = {0, 0, 0, 0, 0, 0};
        return ge::GRAPH_SUCCESS;
    }
    if (commInfo.padModeStr == "SAME") {
        SetSamePadInfo(inputData);
        return ge::GRAPH_SUCCESS;
    }
    VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: not support padmode %s", commInfo.padModeStr.c_str());
    return ge::GRAPH_FAILED;
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
    const int32_t* ceilModePtr = runtimeAttrs->GetAttrPointer<int32_t>(CEIL_POS);
    if (ceilModePtr != nullptr) {
        inputData.ceilMode = (*ceilModePtr != 0);
    }

    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputFormat);
    const std::string inputFormatStr(inputFormat);
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
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: when padmode is VALID, the outputshape in \
d-dim, h-dim and w-dim should be [%ld] [%ld] [%ld], but got [%ld] [%ld] [%ld]",
                                        expectedD, expectedH, expectedW, inputData.outShape[D_DIM],
                                        inputData.outShape[H_DIM], inputData.outShape[W_DIM]);
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
        VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3D: when padmode is SAME, the outputshape in \
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
    if (commInfo.padModeStr == "VALID") {
        return CheckOutPutShapeForValid(context_, inputData);
    } else if (commInfo.padModeStr == "SAME") {
        return CheckOutPutShapeForSame(context_, inputData);
    }
    return ge::GRAPH_SUCCESS;
}

static void RefineShape(Pool3DInputInfo& inputData)
{
    if (inputData.outShape[D_DIM] == 1 && inputData.dilation[D_DIM] == 1) {
        inputData.kernelSize[D_DIM] = std::min(inputData.kernelSize[D_DIM] - inputData.pad[FRONT_PAD_INDEX],
                                               inputData.inputShape[D_DIM]);
        inputData.pad[FRONT_PAD_INDEX] = 0;
        inputData.pad[BACKEND_PAD_INDEX] = 0;
        inputData.stride[D_DIM] = inputData.inputShape[D_DIM];
    }

    if (inputData.outShape[H_DIM] == 1 && inputData.dilation[H_DIM] == 1) {
        inputData.kernelSize[H_DIM] = std::min(inputData.kernelSize[H_DIM] - inputData.pad[TOP_PAD_INDEX],
                                               inputData.inputShape[H_DIM]);
        inputData.pad[TOP_PAD_INDEX] = 0;
        inputData.pad[BOTTOM_PAD_INDEX] = 0;
        inputData.stride[H_DIM] = inputData.inputShape[H_DIM];
    }

    if (inputData.outShape[W_DIM] == 1 && inputData.dilation[W_DIM] == 1) {
        inputData.kernelSize[W_DIM] = std::min(inputData.kernelSize[W_DIM] - inputData.pad[LEFT_PAD_INDEX],
                                               inputData.inputShape[W_DIM]);
        inputData.pad[LEFT_PAD_INDEX] = 0;
        inputData.pad[RIGHT_PAD_INDEX] = 0;
        inputData.stride[W_DIM] = inputData.inputShape[W_DIM];
    }
}

static ge::graphStatus GetMaxPool3DShapeAttrsInfo(gert::TilingContext* context_, Pool3DInputInfo& inputData)
{
    if (context_ == nullptr) {
        OP_LOGE("MaxPool3D", "Tiling context is null.");
        return ge::GRAPH_FAILED;
    }
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

} // namespace optiling
using namespace AscendC;
namespace optiling {
namespace {
constexpr uint32_t FORMAT_NDHWC_VALUE = 0;
constexpr uint32_t FORMAT_NCDHW_VALUE = 1;
constexpr uint32_t LAYOUT_ND_VALUE = 0;
constexpr uint32_t LAYOUT_NDC1HWC0_VALUE = 1;
constexpr uint64_t OUTPUT_TILE_NUM = 5888;
constexpr uint64_t INPUT_TILE_NUM = 23040;
constexpr uint64_t NDHWC_STRIDE2_DTHENW_ROWS = 2;
constexpr uint64_t TINY_K3_VALID_ROWS_PER_CORE = 1;
constexpr uint64_t NCDHW_K1_HALF_MAX_DEPTH_GROUP = 4;
constexpr int64_t NDC1HWC0_LOGICAL_C0 = 16;

struct MaxPool3DLogicalDims {
    int64_t n = 0;
    int64_t c = 0;
    int64_t inD = 0;
    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outD = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    uint32_t outputLayout = LAYOUT_ND_VALUE;
    int64_t outputD = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    int64_t outputC1 = 1;
    int64_t outputC0 = 0;
    int64_t outputC0Block = 0;
    uint32_t inputLayout = LAYOUT_ND_VALUE;
    int64_t inputC1 = 1;
    int64_t inputC0 = 0;
    int64_t inputC0Block = 0;
};

static uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    if (divisor == 0U) {
        return 0U;
    }
    return (value + divisor - 1U) / divisor;
}

static uint64_t PositiveDim(int64_t value) { return value > 0 ? static_cast<uint64_t>(value) : 0U; }

static bool MatchesKernelStrideDilation(const Pool3DInputInfo& inputData, int64_t kD, int64_t kH, int64_t kW,
                                        int64_t sD, int64_t sH, int64_t sW, int64_t dilationD, int64_t dilationH,
                                        int64_t dilationW)
{
    return inputData.kernelSize[D_DIM] == kD && inputData.kernelSize[H_DIM] == kH &&
           inputData.kernelSize[W_DIM] == kW && inputData.stride[D_DIM] == sD && inputData.stride[H_DIM] == sH &&
           inputData.stride[W_DIM] == sW && inputData.dilation[D_DIM] == dilationD &&
           inputData.dilation[H_DIM] == dilationH && inputData.dilation[W_DIM] == dilationW;
}

static bool MatchesFrontTopLeftPad(const Pool3DInputInfo& inputData, int64_t front, int64_t top, int64_t left)
{
    return inputData.pad[FRONT_PAD_INDEX] == front && inputData.pad[TOP_PAD_INDEX] == top &&
           inputData.pad[LEFT_PAD_INDEX] == left;
}

static bool MatchesPoolSpec(const Pool3DInputInfo& inputData, int64_t kD, int64_t kH, int64_t kW, int64_t sD,
                            int64_t sH, int64_t sW, int64_t dilationD, int64_t dilationH, int64_t dilationW,
                            int64_t front, int64_t top, int64_t left)
{
    return MatchesKernelStrideDilation(inputData, kD, kH, kW, sD, sH, sW, dilationD, dilationH, dilationW) &&
           MatchesFrontTopLeftPad(inputData, front, top, left);
}

static int64_t ResolveNdc1hwc0Block(int64_t outputC0)
{
    if (outputC0 <= 0) {
        return NDC1HWC0_LOGICAL_C0;
    }
    return std::min<int64_t>(outputC0, NDC1HWC0_LOGICAL_C0);
}

static uint64_t SafeMul(uint64_t lhs, uint64_t rhs)
{
    if (lhs == 0U || rhs == 0U) {
        return 0U;
    }
    if (lhs > UINT64_MAX / rhs) {
        return UINT64_MAX;
    }
    return lhs * rhs;
}

static int64_t InferCalculatedOutDim(int64_t dimSize, int64_t ksize, int64_t padL, int64_t padR, int64_t stride,
                                     int64_t dilation, bool ceilMode)
{
    if (stride <= 0 || dimSize <= 0 || ksize <= 0 || dilation <= 0) {
        return 0;
    }
    int64_t tmpTotalInput = dimSize + padL + padR - (ksize - 1) * dilation - 1;
    if (ceilMode) {
        tmpTotalInput += stride - 1;
    }
    int64_t outputSize = tmpTotalInput / stride + 1;
    if (ceilMode && (outputSize - 1) * stride >= dimSize + padL) {
        --outputSize;
    }
    return std::max<int64_t>(outputSize, 0);
}

static void RefineLogicalOutDimsForNdc1hwc0(const Pool3DInputInfo& inputData, MaxPool3DLogicalDims& dims)
{
    if (dims.outputLayout != LAYOUT_NDC1HWC0_VALUE) {
        return;
    }
    dims.outD = std::min(dims.outputD,
                         InferCalculatedOutDim(dims.inD, inputData.kernelSize[D_DIM], inputData.pad[FRONT_PAD_INDEX],
                                               inputData.pad[BACKEND_PAD_INDEX], inputData.stride[D_DIM],
                                               inputData.dilation[D_DIM], inputData.ceilMode));
    dims.outH = std::min(dims.outputH,
                         InferCalculatedOutDim(dims.inH, inputData.kernelSize[H_DIM], inputData.pad[TOP_PAD_INDEX],
                                               inputData.pad[BOTTOM_PAD_INDEX], inputData.stride[H_DIM],
                                               inputData.dilation[H_DIM], inputData.ceilMode));
    dims.outW = std::min(dims.outputW,
                         InferCalculatedOutDim(dims.inW, inputData.kernelSize[W_DIM], inputData.pad[LEFT_PAD_INDEX],
                                               inputData.pad[RIGHT_PAD_INDEX], inputData.stride[W_DIM],
                                               inputData.dilation[W_DIM], inputData.ceilMode));
}

static uint64_t CalcTotalOut(const MaxPool3DLogicalDims& dims)
{
    if (dims.outputLayout == LAYOUT_NDC1HWC0_VALUE) {
        uint64_t total = PositiveDim(dims.n);
        total = SafeMul(total, PositiveDim(dims.outputD));
        total = SafeMul(total, PositiveDim(dims.outputC1));
        total = SafeMul(total, PositiveDim(dims.outputH));
        total = SafeMul(total, PositiveDim(dims.outputW));
        total = SafeMul(total, PositiveDim(dims.outputC0));
        return total;
    }
    uint64_t total = PositiveDim(dims.n);
    total = SafeMul(total, PositiveDim(dims.outD));
    total = SafeMul(total, PositiveDim(dims.outH));
    total = SafeMul(total, PositiveDim(dims.outW));
    total = SafeMul(total, PositiveDim(dims.c));
    return total;
}

static bool IsCompactNdc1hwc0Prefix(const MaxPool3DLogicalDims& dims)
{
    if (dims.outputLayout != LAYOUT_NDC1HWC0_VALUE || dims.c <= 0 || dims.outD <= 0 || dims.outH <= 0 ||
        dims.outW <= 0) {
        return false;
    }
    const uint64_t block = std::max<uint64_t>(PositiveDim(dims.outputC0Block > 0 ? dims.outputC0Block : dims.outputC0),
                                              1U);
    const uint64_t validC1 = CeilDiv(PositiveDim(dims.c), block);
    const uint64_t storageC0 = PositiveDim(dims.outputC0);
    const uint64_t storageC1 = PositiveDim(dims.outputC1);
    const bool packedC0Prefix = storageC0 >= block && storageC0 % block == 0U &&
                                SafeMul(storageC1, storageC0 / block) >= validC1;
    return dims.outputD >= dims.outD && dims.outputH >= dims.outH && dims.outputW >= dims.outW &&
           ((storageC0 == block && storageC1 >= validC1) || packedC0Prefix);
}

static uint64_t CalcCoreSplitOut(const MaxPool3DLogicalDims& dims)
{
    if (IsCompactNdc1hwc0Prefix(dims)) {
        const uint64_t block = std::max<uint64_t>(
            PositiveDim(dims.outputC0Block > 0 ? dims.outputC0Block : dims.outputC0), 1U);
        const uint64_t validC1 = CeilDiv(PositiveDim(dims.c), block);
        uint64_t total = PositiveDim(dims.n);
        total = SafeMul(total, PositiveDim(dims.outD));
        total = SafeMul(total, validC1);
        total = SafeMul(total, PositiveDim(dims.outH));
        total = SafeMul(total, PositiveDim(dims.outW));
        total = SafeMul(total, block);
        return total;
    }
    return CalcTotalOut(dims);
}

static bool IsPool2Stride2NoPad(const Pool3DInputInfo& inputData)
{
    return inputData.kernelSize[D_DIM] == 2 && inputData.kernelSize[H_DIM] == 2 && inputData.kernelSize[W_DIM] == 2 &&
           inputData.stride[D_DIM] == 2 && inputData.stride[H_DIM] == 2 && inputData.stride[W_DIM] == 2 &&
           inputData.dilation[D_DIM] == 1 && inputData.dilation[H_DIM] == 1 && inputData.dilation[W_DIM] == 1 &&
           inputData.pad[FRONT_PAD_INDEX] == 0 && inputData.pad[TOP_PAD_INDEX] == 0 &&
           inputData.pad[LEFT_PAD_INDEX] == 0;
}

static bool IsK1Stride1NoPad(const Pool3DInputInfo& inputData)
{
    return inputData.kernelSize[D_DIM] == 1 && inputData.kernelSize[H_DIM] == 1 && inputData.kernelSize[W_DIM] == 1 &&
           inputData.stride[D_DIM] == 1 && inputData.stride[H_DIM] == 1 && inputData.stride[W_DIM] == 1 &&
           inputData.dilation[D_DIM] == 1 && inputData.dilation[H_DIM] == 1 && inputData.dilation[W_DIM] == 1 &&
           inputData.pad[FRONT_PAD_INDEX] == 0 && inputData.pad[TOP_PAD_INDEX] == 0 &&
           inputData.pad[LEFT_PAD_INDEX] == 0;
}

static bool IsNdhwcD2H3W2Dil2Special(const Pool3DInputInfo& inputData)
{
    return inputData.inputFormat == ge::Format::FORMAT_NDHWC && inputData.kernelSize[D_DIM] == 2 &&
           inputData.kernelSize[H_DIM] == 3 && inputData.kernelSize[W_DIM] == 2 && inputData.stride[D_DIM] == 1 &&
           inputData.stride[H_DIM] == 2 && inputData.stride[W_DIM] == 1 && inputData.dilation[D_DIM] == 2 &&
           inputData.dilation[H_DIM] == 2 && inputData.dilation[W_DIM] == 1 && inputData.pad[FRONT_PAD_INDEX] >= 0 &&
           inputData.pad[TOP_PAD_INDEX] >= 0 && inputData.pad[LEFT_PAD_INDEX] >= 0;
}

static bool IsNdc1hwc0D2H3W2Dil2Special(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE &&
           (inputData.inputFormat == ge::Format::FORMAT_NDHWC || inputData.inputFormat == ge::Format::FORMAT_NCDHW) &&
           inputData.kernelSize[D_DIM] == 2 && inputData.kernelSize[H_DIM] == 3 && inputData.kernelSize[W_DIM] == 2 &&
           inputData.stride[D_DIM] == 1 && inputData.stride[H_DIM] == 2 && inputData.stride[W_DIM] == 1 &&
           inputData.dilation[D_DIM] == 2 && inputData.dilation[H_DIM] == 2 && inputData.dilation[W_DIM] == 1 &&
           inputData.pad[FRONT_PAD_INDEX] >= 0 && inputData.pad[TOP_PAD_INDEX] >= 0 &&
           inputData.pad[LEFT_PAD_INDEX] >= 0;
}

static bool IsNdc1hwc0HOnlyStride3ReusablePlane(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && inputData.kernelSize[D_DIM] == 1 &&
           inputData.kernelSize[H_DIM] == 3 && inputData.kernelSize[W_DIM] == 1 && inputData.stride[D_DIM] == 1 &&
           inputData.stride[H_DIM] == 3 && inputData.stride[W_DIM] == 1 && inputData.dilation[D_DIM] == 1 &&
           inputData.dilation[H_DIM] == 1 && inputData.dilation[W_DIM] == 1 && inputData.pad[FRONT_PAD_INDEX] == 0 &&
           inputData.pad[LEFT_PAD_INDEX] == 0 && dims.outW == dims.inW;
}

static bool IsNdc1hwc0TinyK3ValidSpecial(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    if (dims.outputLayout != LAYOUT_NDC1HWC0_VALUE || inputData.isBfloat16 ||
        (inputData.dtypeSize != 2 && inputData.dtypeSize != 4) ||
        (inputData.inputFormat != ge::Format::FORMAT_NCDHW && inputData.inputFormat != ge::Format::FORMAT_NDHWC)) {
        return false;
    }
    const uint64_t block = std::max<uint64_t>(PositiveDim(dims.outputC0Block > 0 ? dims.outputC0Block : dims.outputC0),
                                              1U);
    const uint64_t validC1 = CeilDiv(PositiveDim(dims.c), block);
    return block == NDC1HWC0_LOGICAL_C0 && validC1 == 1U && PositiveDim(dims.c) <= block &&
           (inputData.stride[H_DIM] == 1 || inputData.stride[H_DIM] == 3) &&
           MatchesPoolSpec(inputData, 3, 3, 3, 1, inputData.stride[H_DIM], 1, 1, 1, 1, 0, 0, 0) && dims.outD > 0 &&
           dims.outH == 1 && dims.outW > 0 && dims.inD == dims.outD + 2 && dims.inH == 3 && dims.inW == dims.outW + 2;
}

static bool IsNdhwcTinyK3Feature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return inputData.inputFormat == ge::Format::FORMAT_NDHWC && IsNdc1hwc0TinyK3ValidSpecial(inputData, dims);
}

static bool IsNcdhwTinyK3Feature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return inputData.inputFormat == ge::Format::FORMAT_NCDHW && IsNdc1hwc0TinyK3ValidSpecial(inputData, dims);
}

static bool IsTinyK3PhysicalRoute(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.inputLayout == LAYOUT_NDC1HWC0_VALUE && dims.outputD >= dims.outD && dims.outputH >= dims.outH &&
           dims.outputW >= dims.outW && IsNdc1hwc0TinyK3ValidSpecial(inputData, dims);
}

static bool IsHOnlyStride3PhysicalRoute(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    const uint64_t block = PositiveDim(dims.outputC0Block);
    const uint64_t rowElements = SafeMul(PositiveDim(dims.outW), block);
    const uint64_t planeElements = SafeMul(PositiveDim(dims.outH), rowElements);
    return !inputData.isBfloat16 && (inputData.dtypeSize == 2 || inputData.dtypeSize == 4) && block > 0U &&
           (inputData.inputFormat == ge::Format::FORMAT_NDHWC || inputData.inputFormat == ge::Format::FORMAT_NCDHW) &&
           dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && PositiveDim(dims.c) <= block &&
           PositiveDim(dims.outputC1) == 1U && dims.outD == dims.inD && dims.outW == dims.inW && rowElements > 0U &&
           planeElements <= OUTPUT_TILE_NUM && MatchesPoolSpec(inputData, 1, 3, 1, 1, 3, 1, 1, 1, 1, 0, 1, 0);
}

static bool IsPool2Stride2PhysicalRoute(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    if (inputData.isBfloat16 || (inputData.dtypeSize != 2 && inputData.dtypeSize != 4) ||
        dims.inputLayout != LAYOUT_NDC1HWC0_VALUE || dims.outputLayout != LAYOUT_NDC1HWC0_VALUE ||
        !IsPool2Stride2NoPad(inputData)) {
        return false;
    }
    const uint64_t inputBlock = PositiveDim(dims.inputC0Block);
    const bool supportedInputBlock = inputBlock == (inputData.dtypeSize == 4 ? 8U : 16U);
    const uint64_t inputRow = SafeMul(PositiveDim(dims.inW), inputBlock);
    const uint64_t outputBlock = PositiveDim(dims.outputC0Block);
    const uint64_t outputRow = SafeMul(PositiveDim(dims.outW), outputBlock);
    const bool profitableFamily = dims.n > 1 || dims.c >= 64 ||
                                  (inputData.dtypeSize == sizeof(float) && dims.c > NDC1HWC0_LOGICAL_C0);
    return profitableFamily && supportedInputBlock && inputRow > 0U && inputRow <= INPUT_TILE_NUM && outputRow > 0U &&
           outputRow <= OUTPUT_TILE_NUM && PositiveDim(dims.inputC1) == CeilDiv(PositiveDim(dims.c), inputBlock) &&
           dims.outD == (dims.inD + 1) / 2 && dims.outH == (dims.inH + 1) / 2 && dims.outW == (dims.inW + 1) / 2;
}

static bool IsK1IdentityPhysicalRoute(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    if ((inputData.dtypeSize != 2 && inputData.dtypeSize != 4) || inputData.isBfloat16 ||
        dims.inputLayout != LAYOUT_NDC1HWC0_VALUE || dims.outputLayout != LAYOUT_NDC1HWC0_VALUE ||
        (inputData.inputFormat != ge::Format::FORMAT_NCDHW && inputData.inputFormat != ge::Format::FORMAT_NDHWC) ||
        !IsK1Stride1NoPad(inputData) || PositiveDim(dims.inputC0Block) != (inputData.dtypeSize == 4 ? 8U : 16U)) {
        return false;
    }
    const uint64_t inputBlock = PositiveDim(dims.inputC0Block);
    const uint64_t inputC1 = PositiveDim(dims.inputC1);
    const uint64_t rowElements = SafeMul(SafeMul(PositiveDim(dims.outW), inputC1), inputBlock);
    return dims.outD == dims.inD && dims.outH == dims.inH && dims.outW == dims.inW && inputC1 > 0U &&
           SafeMul(inputC1, inputBlock) >= PositiveDim(dims.c) && rowElements > 0U && rowElements <= INPUT_TILE_NUM;
}

static bool IsNdc1hwc0NcdhwD3H3Dil2ReusablePlane(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && inputData.inputFormat == ge::Format::FORMAT_NCDHW &&
           inputData.kernelSize[D_DIM] == 3 && inputData.kernelSize[H_DIM] == 3 && inputData.kernelSize[W_DIM] == 1 &&
           inputData.stride[D_DIM] == 3 && inputData.stride[H_DIM] == 1 && inputData.stride[W_DIM] == 1 &&
           inputData.dilation[D_DIM] == 1 && inputData.dilation[H_DIM] == 2 && inputData.dilation[W_DIM] == 1 &&
           inputData.pad[FRONT_PAD_INDEX] == 0 && inputData.pad[TOP_PAD_INDEX] == 2 &&
           inputData.pad[LEFT_PAD_INDEX] == 0 && dims.outW == dims.inW;
}

static bool IsNdc1hwc0NdhwcD3H3Dil2ReusablePlane(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && inputData.inputFormat == ge::Format::FORMAT_NDHWC &&
           inputData.kernelSize[D_DIM] == 3 && inputData.kernelSize[H_DIM] == 3 && inputData.kernelSize[W_DIM] == 1 &&
           inputData.stride[D_DIM] == 3 && inputData.stride[H_DIM] == 1 && inputData.stride[W_DIM] == 1 &&
           inputData.dilation[D_DIM] == 1 && inputData.dilation[H_DIM] == 2 && inputData.dilation[W_DIM] == 1 &&
           inputData.pad[FRONT_PAD_INDEX] == 0 && inputData.pad[TOP_PAD_INDEX] == 2 &&
           inputData.pad[LEFT_PAD_INDEX] == 0 && dims.outW == dims.inW;
}

static bool IsD3H3Dil2PhysicalFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    const uint64_t expectedInputC0 = inputData.dtypeSize == sizeof(float) ? 8U : 16U;
    const uint64_t inputC0 = PositiveDim(dims.inputC0Block);
    const bool logicalInput = dims.inputLayout != LAYOUT_NDC1HWC0_VALUE;
    const bool supportedInputC0 = logicalInput || inputC0 == expectedInputC0 ||
                                  (inputData.dtypeSize == sizeof(float) && inputC0 == NDC1HWC0_LOGICAL_C0);
    const uint64_t outputBlock = PositiveDim(dims.outputC0Block);
    const uint64_t rowElements = SafeMul(PositiveDim(dims.outW), outputBlock);
    return !inputData.isBfloat16 && (inputData.dtypeSize == sizeof(uint16_t) || inputData.dtypeSize == sizeof(float)) &&
           dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && dims.n > 0 && dims.c > 0 && dims.inD > 0 && dims.inH > 0 &&
           dims.inW > 0 && dims.outD > 0 && dims.outH > 0 && dims.outW == dims.inW &&
           (logicalInput || PositiveDim(dims.inputC1) == 1U) && supportedInputC0 && PositiveDim(dims.outputC1) == 1U &&
           PositiveDim(dims.c) <= std::max<uint64_t>(inputC0, outputBlock) && rowElements <= OUTPUT_TILE_NUM &&
           (IsNdc1hwc0NcdhwD3H3Dil2ReusablePlane(inputData, dims) ||
            IsNdc1hwc0NdhwcD3H3Dil2ReusablePlane(inputData, dims));
}

static bool IsNcdhwPool2Feature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return inputData.dtypeSize == sizeof(float) && inputData.inputFormat == ge::Format::FORMAT_NCDHW &&
           dims.outputLayout == LAYOUT_ND_VALUE && dims.n == 1 && dims.c > 0 && dims.inD > 0 && dims.inH > 0 &&
           dims.inW > 0 && dims.outD == (dims.inD + 1) / 2 && dims.outH == (dims.inH + 1) / 2 &&
           dims.outW == (dims.inW + 1) / 2 && IsPool2Stride2NoPad(inputData);
}

static bool IsNdhwcCompactK1Feature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    const uint64_t block = PositiveDim(dims.outputC0Block);
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && inputData.inputFormat == ge::Format::FORMAT_NDHWC &&
           !inputData.isBfloat16 && (inputData.dtypeSize == 2 || inputData.dtypeSize == 4) && dims.n > 0 &&
           dims.c > 0 && PositiveDim(dims.c) <= block && dims.outD == dims.inD && dims.outH == dims.inH &&
           dims.outW == dims.inW && SafeMul(PositiveDim(dims.outW), block) <= OUTPUT_TILE_NUM &&
           IsK1Stride1NoPad(inputData);
}

static bool IsNcdhwFloatCompactK1Feature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    const uint64_t block = PositiveDim(dims.outputC0Block);
    return inputData.dtypeSize == sizeof(float) && !inputData.isBfloat16 &&
           inputData.inputFormat == ge::Format::FORMAT_NCDHW && dims.outputLayout == LAYOUT_NDC1HWC0_VALUE &&
           dims.n > 0 && dims.c > 0 && PositiveDim(dims.c) <= block && dims.outD == dims.inD && dims.outH == dims.inH &&
           dims.outW == dims.inW && SafeMul(PositiveDim(dims.outW), block) <= OUTPUT_TILE_NUM &&
           IsK1Stride1NoPad(inputData);
}

static bool IsNdhwcFloatK1DmaFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    if (inputData.dtypeSize != 4 || inputData.isBfloat16 || inputData.inputFormat != ge::Format::FORMAT_NDHWC ||
        dims.inputLayout == LAYOUT_NDC1HWC0_VALUE || dims.outputLayout != LAYOUT_NDC1HWC0_VALUE ||
        !IsK1Stride1NoPad(inputData)) {
        return false;
    }
    const uint64_t block = PositiveDim(dims.outputC0Block);
    const uint64_t validC1 = CeilDiv(PositiveDim(dims.c), std::max<uint64_t>(block, 1U));
    const uint64_t planeElements = SafeMul(SafeMul(PositiveDim(dims.outH), PositiveDim(dims.outW)),
                                           SafeMul(validC1, block));
    return block == NDC1HWC0_LOGICAL_C0 && dims.outD == dims.inD && dims.outH == dims.inH && dims.outW == dims.inW &&
           planeElements > 0U && planeElements <= OUTPUT_TILE_NUM;
}

static bool IsNcdhwSmallDepthStride2Feature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.outputLayout == LAYOUT_ND_VALUE && !inputData.isBfloat16 &&
           (inputData.dtypeSize == 2 || inputData.dtypeSize == 4) &&
           inputData.inputFormat == ge::Format::FORMAT_NCDHW && IsPool2Stride2NoPad(inputData) && dims.n == 1 &&
           dims.c > NDC1HWC0_LOGICAL_C0 && dims.inD > 0 && dims.outD > 0 && dims.outD <= 2 &&
           dims.outD == (dims.inD + 1) / 2 && dims.outH == (dims.inH + 1) / 2 && dims.outW == (dims.inW + 1) / 2;
}

static bool IsNdhwcD3H3BalancedFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && inputData.inputFormat == ge::Format::FORMAT_NDHWC &&
           IsD3H3Dil2PhysicalFeature(inputData, dims);
}

static bool IsNdhwcK1WideBalancedFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && inputData.inputFormat == ge::Format::FORMAT_NDHWC &&
           dims.c > NDC1HWC0_LOGICAL_C0 && IsNdhwcFloatK1DmaFeature(inputData, dims);
}

static bool IsNdhwcStride2BalancedFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && inputData.inputFormat == ge::Format::FORMAT_NDHWC &&
           dims.n > 1 && IsPool2Stride2PhysicalRoute(inputData, dims);
}

static bool IsNcdhwD2H3W2BalancedFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && inputData.inputFormat == ge::Format::FORMAT_NCDHW &&
           IsNdc1hwc0D2H3W2Dil2Special(inputData, dims);
}

static bool IsBalancedSplitFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return IsNcdhwSmallDepthStride2Feature(inputData, dims) || IsNdhwcD3H3BalancedFeature(inputData, dims) ||
           IsNdhwcCompactK1Feature(inputData, dims) || IsNdhwcK1WideBalancedFeature(inputData, dims) ||
           IsNdhwcStride2BalancedFeature(inputData, dims) || IsNcdhwD2H3W2BalancedFeature(inputData, dims) ||
           IsK1IdentityPhysicalRoute(inputData, dims);
}

static bool IsNdc1hwc0NcdhwD3W3DilD2Special(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && inputData.inputFormat == ge::Format::FORMAT_NCDHW &&
           MatchesPoolSpec(inputData, 3, 1, 3, 3, 1, 3, 2, 1, 1, 0, 0, 0) && dims.n > 0 && dims.c > 0 &&
           dims.outD == 1 && dims.outH == dims.inH && dims.outW > 0 &&
           SafeMul(PositiveDim(dims.c), PositiveDim(dims.inW)) <= INPUT_TILE_NUM;
}

static bool IsNdc1hwc0D3W3DilD2PhysicalFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    const bool validPhysicalChannel = dims.inputLayout != LAYOUT_NDC1HWC0_VALUE ||
                                      (inputData.dtypeSize == 4 &&
                                       ((PositiveDim(dims.inputC1) == 2U && PositiveDim(dims.inputC0Block) == 8U) ||
                                        (PositiveDim(dims.inputC1) == 1U && PositiveDim(dims.inputC0Block) == 16U))) ||
                                      (inputData.dtypeSize == 2 && PositiveDim(dims.inputC1) == 1U &&
                                       PositiveDim(dims.inputC0Block) == 16U);
    const uint64_t outputBlock = PositiveDim(dims.outputC0Block);
    const uint64_t outputRow = SafeMul(PositiveDim(dims.outW), outputBlock);
    return dims.outputLayout == LAYOUT_NDC1HWC0_VALUE &&
           (inputData.inputFormat == ge::Format::FORMAT_NCDHW || inputData.inputFormat == ge::Format::FORMAT_NDHWC) &&
           MatchesPoolSpec(inputData, 3, 1, 3, 3, 1, 3, 2, 1, 1, 0, 0, 0) && dims.n > 0 && dims.c > 0 &&
           dims.outD == 1 && dims.outH == dims.inH && dims.outW > 0 && outputRow > 0U && outputRow <= OUTPUT_TILE_NUM &&
           PositiveDim(dims.outputC1) == 1U && PositiveDim(dims.c) <= outputBlock && validPhysicalChannel;
}

static bool IsNdc1hwc0DilatedWPhysicalFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    const uint64_t expectedInputBlock = inputData.dtypeSize == 4 ? 8U : 16U;
    const uint64_t expectedOutputC0 = inputData.dtypeSize == 4 ? 32U : 16U;
    const uint64_t rowElements = SafeMul(PositiveDim(dims.outW), NDC1HWC0_LOGICAL_C0);
    return !inputData.isBfloat16 && (inputData.dtypeSize == 2 || inputData.dtypeSize == 4) &&
           dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && dims.n > 0 && dims.c > 0 &&
           PositiveDim(dims.c) <= NDC1HWC0_LOGICAL_C0 &&
           (dims.inputLayout != LAYOUT_NDC1HWC0_VALUE ||
            (PositiveDim(dims.inputC1) == 1U && PositiveDim(dims.inputC0Block) == expectedInputBlock)) &&
           PositiveDim(dims.outputC1) == 1U && PositiveDim(dims.outputC0) == expectedOutputC0 &&
           PositiveDim(dims.outputC0Block) == NDC1HWC0_LOGICAL_C0 && dims.outD > 0 && dims.outH > 0 && dims.outW > 0 &&
           rowElements <= OUTPUT_TILE_NUM && MatchesKernelStrideDilation(inputData, 1, 1, 3, 3, 3, 3, 1, 1, 2);
}

static bool IsNcdhwDilatedWFloatFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    const uint64_t block = PositiveDim(dims.outputC0Block);
    const uint64_t rowElements = SafeMul(PositiveDim(dims.outW), block);
    return !inputData.isBfloat16 && inputData.dtypeSize == sizeof(float) &&
           inputData.inputFormat == ge::Format::FORMAT_NCDHW && dims.inputLayout != LAYOUT_NDC1HWC0_VALUE &&
           dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && dims.n > 0 && dims.c > 0 && PositiveDim(dims.c) <= block &&
           PositiveDim(dims.outputC1) == 1U && PositiveDim(dims.outputC0) == 32U && block == NDC1HWC0_LOGICAL_C0 &&
           dims.outD > 0 && dims.outH > 0 && dims.outW > 0 && rowElements <= OUTPUT_TILE_NUM &&
           MatchesKernelStrideDilation(inputData, 1, 1, 3, 3, 3, 3, 1, 1, 2) && inputData.ceilMode &&
           MatchesFrontTopLeftPad(inputData, 2, 1, 1) && inputData.pad[BACKEND_PAD_INDEX] == 2 &&
           inputData.pad[BOTTOM_PAD_INDEX] == 1 && inputData.pad[RIGHT_PAD_INDEX] == 1;
}

static bool IsNdc1hwc0D2H3W2Dil2PhysicalFeature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    const uint64_t block = PositiveDim(dims.outputC0Block);
    const uint64_t planeElements = SafeMul(SafeMul(PositiveDim(dims.outH), PositiveDim(dims.outW)), block);
    return inputData.dtypeSize == 2 && !inputData.isBfloat16 && dims.outputLayout == LAYOUT_NDC1HWC0_VALUE &&
           (inputData.inputFormat == ge::Format::FORMAT_NCDHW || inputData.inputFormat == ge::Format::FORMAT_NDHWC) &&
           dims.n > 0 && dims.c > 0 && PositiveDim(dims.c) <= block &&
           (dims.inputLayout != LAYOUT_NDC1HWC0_VALUE || (dims.inputC1 == 1 && dims.inputC0Block == 16)) &&
           PositiveDim(dims.outputC1) == 1U && planeElements > 0U && planeElements <= OUTPUT_TILE_NUM &&
           MatchesPoolSpec(inputData, 2, 3, 2, 1, 2, 1, 2, 2, 1, 1, 2, 0);
}

static uint64_t AlignToUbBlock(uint64_t count, int64_t dtypeSize, uint64_t ubBlockBytes)
{
    const uint64_t safeDtypeSize = dtypeSize > 0 ? static_cast<uint64_t>(dtypeSize) : 1U;
    const uint64_t alignNum = std::max<uint64_t>(ubBlockBytes / safeDtypeSize, 1U);
    return (count + alignNum - 1U) / alignNum * alignNum;
}

static uint64_t SelectNdhwcK1DGroup(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims, uint64_t block,
                                    uint64_t validC1, uint64_t ubBlockBytes)
{
    if (inputData.inputFormat != ge::Format::FORMAT_NDHWC || block != NDC1HWC0_LOGICAL_C0 || validC1 <= 1U ||
        dims.outD <= 1 || dims.outH <= 0 || dims.outW <= 0 || dims.c <= 0) {
        return 1U;
    }
    const uint64_t rowElements = SafeMul(PositiveDim(dims.outW), block);
    const uint64_t planeElements = SafeMul(SafeMul(rowElements, PositiveDim(dims.outH)), validC1);
    if (rowElements == 0U || planeElements == 0U || planeElements > OUTPUT_TILE_NUM) {
        return 1U;
    }
    uint64_t maxGroup = std::min<uint64_t>(PositiveDim(dims.outD), OUTPUT_TILE_NUM / planeElements);
    const uint64_t c0Count = std::min<uint64_t>(PositiveDim(dims.c), block);
    const uint64_t alignedC0 = AlignToUbBlock(c0Count, inputData.dtypeSize, ubBlockBytes);
    while (maxGroup > 1U) {
        const uint64_t inputNeed = SafeMul(SafeMul(SafeMul(maxGroup, PositiveDim(dims.outH)), PositiveDim(dims.outW)),
                                           alignedC0);
        if (inputNeed <= INPUT_TILE_NUM) {
            break;
        }
        --maxGroup;
    }
    return std::max<uint64_t>(maxGroup, 1U);
}

static uint64_t SelectNcdhwK1DGroup(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims, uint64_t block,
                                    uint64_t validC1, uint64_t ubBlockBytes)
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCDHW || block != NDC1HWC0_LOGICAL_C0 || validC1 == 0U ||
        dims.outD <= 1 || dims.outH <= 0 || dims.outW <= 0 || dims.c <= 0) {
        return 1U;
    }
    const uint64_t planeValid = SafeMul(PositiveDim(dims.outH), PositiveDim(dims.outW));
    const uint64_t outputBlock = SafeMul(planeValid, block);
    if (planeValid == 0U || outputBlock == 0U || outputBlock > OUTPUT_TILE_NUM) {
        return 1U;
    }
    uint64_t maxGroup = std::min<uint64_t>(PositiveDim(dims.outD), OUTPUT_TILE_NUM / outputBlock);
    if (inputData.dtypeSize == sizeof(uint16_t) && validC1 > 1U) {
        maxGroup = std::min<uint64_t>(maxGroup, NCDHW_K1_HALF_MAX_DEPTH_GROUP);
    }
    const uint64_t channel = PositiveDim(dims.c);
    while (maxGroup > 1U) {
        const uint64_t groupValid = SafeMul(maxGroup, planeValid);
        const uint64_t alignedGroup = AlignToUbBlock(groupValid + 1U, inputData.dtypeSize, ubBlockBytes);
        const uint64_t inputNeed = SafeMul(channel + 1U, alignedGroup);
        const uint64_t transWrite = SafeMul(alignedGroup, block);
        if (inputNeed <= INPUT_TILE_NUM && transWrite <= OUTPUT_TILE_NUM) {
            break;
        }
        --maxGroup;
    }
    return std::max<uint64_t>(maxGroup, 1U);
}

static uint64_t SelectTinyK3Quantum(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                    uint64_t rowElements, uint64_t totalOut, uint64_t outH, uint64_t validC1)
{
    const bool tinyK3 = IsTinyK3PhysicalRoute(inputData, dims) || IsNdhwcTinyK3Feature(inputData, dims) ||
                        IsNcdhwTinyK3Feature(inputData, dims) || IsNdc1hwc0TinyK3ValidSpecial(inputData, dims);
    if (!tinyK3 || rowElements == 0U) {
        return 0U;
    }
    const uint64_t totalRows = SafeMul(SafeMul(PositiveDim(dims.n), PositiveDim(dims.outD)), SafeMul(validC1, outH));
    const bool canGroupWholeN = dims.inputLayout == LAYOUT_NDC1HWC0_VALUE ||
                                (inputData.dtypeSize == 2 && IsNdhwcTinyK3Feature(inputData, dims)) ||
                                IsNcdhwTinyK3Feature(inputData, dims);
    const uint64_t rowsPerCore = canGroupWholeN ?
                                     PositiveDim(dims.outD) :
                                     (inputData.inputFormat == ge::Format::FORMAT_NCDHW ? 1U :
                                                                                          TINY_K3_VALID_ROWS_PER_CORE);
    const uint64_t maxRows = std::min<uint64_t>(std::max<uint64_t>(OUTPUT_TILE_NUM / rowElements, 1U), rowsPerCore);
    const uint64_t groupRows = std::min<uint64_t>(std::max<uint64_t>(totalRows, 1U), maxRows);
    const uint64_t groupedRowElements = SafeMul(rowElements, groupRows);
    return groupedRowElements > 0U && groupedRowElements <= totalOut ? groupedRowElements : 0U;
}

static uint64_t SelectCompactEarlyQuantum(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                          uint64_t rowElements, uint64_t planeElements, uint64_t totalOut,
                                          uint64_t outH, uint64_t validC1)
{
    if (IsHOnlyStride3PhysicalRoute(inputData, dims) && planeElements > 0U && planeElements <= totalOut) {
        return planeElements;
    }
    const uint64_t tinyQuantum = SelectTinyK3Quantum(inputData, dims, rowElements, totalOut, outH, validC1);
    if (tinyQuantum > 0U) {
        return tinyQuantum;
    }
    if (IsNdc1hwc0NcdhwD3W3DilD2Special(inputData, dims) && rowElements > 0U) {
        const uint64_t groupRows = std::min<uint64_t>(PositiveDim(dims.outH), 6U);
        const uint64_t groupedRowElements = SafeMul(rowElements, groupRows);
        if (groupedRowElements > 0U && groupedRowElements <= totalOut) {
            return groupedRowElements;
        }
    }
    return IsK1IdentityPhysicalRoute(inputData, dims) && planeElements > 0U ? planeElements : 0U;
}

static uint64_t SelectCompactK1Quantum(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                       uint64_t block, uint64_t validC1, uint64_t ubBlockBytes, uint64_t planeElements,
                                       uint64_t totalOut)
{
    if (!IsK1Stride1NoPad(inputData)) {
        return 0U;
    }
    uint64_t groupPlanes = 0U;
    if (inputData.inputFormat == ge::Format::FORMAT_NCDHW) {
        groupPlanes = SelectNcdhwK1DGroup(inputData, dims, block, validC1, ubBlockBytes);
    } else if (inputData.inputFormat == ge::Format::FORMAT_NDHWC && validC1 > 1U) {
        groupPlanes = SelectNdhwcK1DGroup(inputData, dims, block, validC1, ubBlockBytes);
    }
    const uint64_t groupedPlaneElements = SafeMul(planeElements, groupPlanes);
    return groupedPlaneElements > 0U && groupedPlaneElements <= totalOut ? groupedPlaneElements : 0U;
}

static bool NeedsCompactWholePlane(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims, uint64_t validC1)
{
    if (IsK1Stride1NoPad(inputData) || IsNdhwcD2H3W2Dil2Special(inputData) ||
        IsNdc1hwc0D2H3W2Dil2Special(inputData, dims) || IsNdc1hwc0HOnlyStride3ReusablePlane(inputData, dims) ||
        IsNdc1hwc0NcdhwD3H3Dil2ReusablePlane(inputData, dims) ||
        IsNdc1hwc0NdhwcD3H3Dil2ReusablePlane(inputData, dims)) {
        return true;
    }
    return IsPool2Stride2NoPad(inputData) && ((validC1 == 1U && inputData.inputFormat == ge::Format::FORMAT_NCDHW) ||
                                              (validC1 > 1U && (inputData.inputFormat == ge::Format::FORMAT_NDHWC ||
                                                                inputData.inputFormat == ge::Format::FORMAT_NCDHW)));
}

static uint64_t SelectCompactThreePlaneQuantum(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                               uint64_t planeElements, uint64_t totalOut)
{
    if (inputData.dtypeSize != sizeof(uint16_t) || !IsNdc1hwc0D2H3W2Dil2Special(inputData, dims)) {
        return 0U;
    }
    const uint64_t effectiveKernelD = SafeMul(PositiveDim(inputData.kernelSize[D_DIM] - 1),
                                              PositiveDim(inputData.dilation[D_DIM])) +
                                      1U;
    const uint64_t groupedPlaneElements = SafeMul(planeElements, effectiveKernelD);
    return groupedPlaneElements > 0U && groupedPlaneElements <= totalOut ? groupedPlaneElements : 0U;
}

static uint64_t SelectCompactLateQuantum(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                         uint64_t block, uint64_t validC1, uint64_t ubBlockBytes, uint64_t rowElements,
                                         uint64_t planeElements, uint64_t totalOut)
{
    const uint64_t k1Quantum = SelectCompactK1Quantum(inputData, dims, block, validC1, ubBlockBytes, planeElements,
                                                      totalOut);
    if (k1Quantum > 0U) {
        return k1Quantum;
    }
    const uint64_t threePlaneQuantum = SelectCompactThreePlaneQuantum(inputData, dims, planeElements, totalOut);
    if (threePlaneQuantum > 0U) {
        return threePlaneQuantum;
    }
    if (NeedsCompactWholePlane(inputData, dims, validC1) && planeElements > 0U && planeElements <= totalOut) {
        return planeElements;
    }
    return rowElements;
}

static uint64_t SelectNdc1hwc0Quantum(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims, uint64_t outW,
                                      uint64_t outH, uint64_t channel, uint64_t totalOut, uint64_t ubBlockBytes)
{
    if (!IsCompactNdc1hwc0Prefix(dims)) {
        const uint64_t storageC0 = std::max<uint64_t>(PositiveDim(dims.outputC0), 1U);
        return SafeMul(std::max<uint64_t>(PositiveDim(dims.outputW), 1U), storageC0);
    }
    const uint64_t block = std::max<uint64_t>(PositiveDim(dims.outputC0Block > 0 ? dims.outputC0Block : dims.outputC0),
                                              1U);
    const uint64_t validC1 = CeilDiv(channel, block);
    const uint64_t rowElements = SafeMul(outW, block);
    const uint64_t planeElements = SafeMul(SafeMul(rowElements, outH), validC1);
    const uint64_t earlyQuantum = SelectCompactEarlyQuantum(inputData, dims, rowElements, planeElements, totalOut, outH,
                                                            validC1);
    if (earlyQuantum > 0U) {
        return earlyQuantum;
    }
    return SelectCompactLateQuantum(inputData, dims, block, validC1, ubBlockBytes, rowElements, planeElements,
                                    totalOut);
}

static uint64_t SelectNdhwcQuantum(const Pool3DInputInfo& inputData, uint64_t outW, uint64_t channel, uint64_t totalOut,
                                   uint64_t coreNum)
{
    const uint64_t rowElements = SafeMul(outW, channel);
    const uint64_t rowCount = rowElements == 0U ? 0U : CeilDiv(totalOut, rowElements);
    if (IsPool2Stride2NoPad(inputData) && rowElements > 0U) {
        const uint64_t twoRowElements = SafeMul(rowElements, NDHWC_STRIDE2_DTHENW_ROWS);
        const uint64_t twoRowInputElements = SafeMul(rowElements, 4U * NDHWC_STRIDE2_DTHENW_ROWS);
        if (rowCount > 0U && rowCount < coreNum && rowElements <= OUTPUT_TILE_NUM) {
            if (inputData.dtypeSize == 2U && channel >= 128U && rowCount >= NDHWC_STRIDE2_DTHENW_ROWS &&
                twoRowElements <= OUTPUT_TILE_NUM && twoRowInputElements <= INPUT_TILE_NUM) {
                return twoRowElements;
            }
            return rowElements;
        }
        if (rowCount >= NDHWC_STRIDE2_DTHENW_ROWS && twoRowElements <= OUTPUT_TILE_NUM &&
            twoRowInputElements <= INPUT_TILE_NUM) {
            return twoRowElements;
        }
        if (rowElements <= OUTPUT_TILE_NUM && rowCount >= std::max<uint64_t>(coreNum / 2U, 1U)) {
            return rowElements;
        }
    }
    return channel;
}

static uint64_t SelectNcdhwD2Quantum(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims, uint64_t outW,
                                     uint64_t planeElements, uint64_t totalOut)
{
    if (!IsPool2Stride2NoPad(inputData) || dims.outD != 2 || planeElements == 0U) {
        return 0U;
    }
    const uint64_t wholeDepthElements = SafeMul(planeElements, 2U);
    if (inputData.inputFormat == ge::Format::FORMAT_NCDHW && inputData.dtypeSize == 2 && dims.inD == 4 && outW <= 32U &&
        wholeDepthElements > 0U && wholeDepthElements <= totalOut) {
        return wholeDepthElements;
    }
    if ((inputData.dtypeSize == 2 || inputData.dtypeSize == 4) && dims.inD == 4 && outW <= 32U) {
        return outW;
    }
    return wholeDepthElements > 0U && wholeDepthElements <= totalOut ? wholeDepthElements : 0U;
}

static uint64_t SelectNcdhwFloatPlaneQuantum(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                             uint64_t planeElements, uint64_t planeCount, uint64_t coreNum)
{
    if (!IsPool2Stride2NoPad(inputData)) {
        return 0U;
    }
    if (inputData.inputFormat == ge::Format::FORMAT_NCDHW && inputData.dtypeSize == 4 && dims.outD > 2 &&
        planeElements > 0U && planeElements <= OUTPUT_TILE_NUM && planeCount >= std::max<uint64_t>(coreNum / 2U, 1U)) {
        return planeElements;
    }
    return 0U;
}

static uint64_t SelectNcdhwCapacityQuantum(const Pool3DInputInfo& inputData, uint64_t outW, uint64_t planeElements)
{
    if (!IsPool2Stride2NoPad(inputData) || planeElements == 0U || planeElements > OUTPUT_TILE_NUM) {
        return 0U;
    }
    const uint64_t largePlaneThreshold = OUTPUT_TILE_NUM - OUTPUT_TILE_NUM / 4U;
    if (inputData.dtypeSize == sizeof(float) && planeElements > largePlaneThreshold) {
        return outW;
    }
    return planeElements;
}

static uint64_t SelectNcdhwFallbackPlaneQuantum(const Pool3DInputInfo& inputData, uint64_t planeElements,
                                                uint64_t planeCount, uint64_t coreNum)
{
    return IsPool2Stride2NoPad(inputData) && planeElements > 0U && planeElements <= OUTPUT_TILE_NUM &&
                   planeCount >= std::max<uint64_t>(coreNum / 2U, 1U) ?
               planeElements :
               0U;
}

static uint64_t SelectNcdhwQuantum(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims, uint64_t outW,
                                   uint64_t outH, uint64_t totalOut, uint64_t coreNum)
{
    const uint64_t planeElements = SafeMul(outW, outH);
    const uint64_t planeCount = planeElements == 0U ? 0U : CeilDiv(totalOut, planeElements);
    if (IsNcdhwSmallDepthStride2Feature(inputData, dims) && planeElements > 0U) {
        return SafeMul(planeElements, 2U);
    }
    const uint64_t d2Quantum = SelectNcdhwD2Quantum(inputData, dims, outW, planeElements, totalOut);
    if (d2Quantum > 0U) {
        return d2Quantum;
    }
    const uint64_t floatPlaneQuantum = SelectNcdhwFloatPlaneQuantum(inputData, dims, planeElements, planeCount,
                                                                    coreNum);
    if (floatPlaneQuantum > 0U) {
        return floatPlaneQuantum;
    }
    const uint64_t capacityQuantum = SelectNcdhwCapacityQuantum(inputData, outW, planeElements);
    if (capacityQuantum > 0U) {
        return capacityQuantum;
    }
    const uint64_t planeQuantum = SelectNcdhwFallbackPlaneQuantum(inputData, planeElements, planeCount, coreNum);
    return planeQuantum > 0U ? planeQuantum : outW;
}

static uint64_t SelectSplitQuantum(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims, uint64_t coreNum,
                                   uint64_t ubBlockBytes)
{
    const uint64_t outW = std::max<uint64_t>(PositiveDim(dims.outW), 1U);
    const uint64_t outH = std::max<uint64_t>(PositiveDim(dims.outH), 1U);
    const uint64_t channel = std::max<uint64_t>(PositiveDim(dims.c), 1U);
    const uint64_t totalOut = CalcTotalOut(dims);
    if (dims.outputLayout == LAYOUT_NDC1HWC0_VALUE) {
        const uint64_t quantum = SelectNdc1hwc0Quantum(inputData, dims, outW, outH, channel, totalOut, ubBlockBytes);
        if (quantum > 0U) {
            return quantum;
        }
    }
    if (inputData.inputFormat == ge::Format::FORMAT_NDHWC) {
        return SelectNdhwcQuantum(inputData, outW, channel, totalOut, coreNum);
    }
    return SelectNcdhwQuantum(inputData, dims, outW, outH, totalOut, coreNum);
}

static void SetFeatureCoreSplit(const MaxPool3DLogicalDims& dims, uint64_t totalOut, uint64_t splitOut,
                                uint64_t featureBlocks, uint64_t coreNum, MaxPool3DTilingData& tiling)
{
    const uint64_t usedCore = std::min<uint64_t>(featureBlocks, coreNum);
    uint64_t splitQuantum = 1U;
    if (IsCompactNdc1hwc0Prefix(dims)) {
        const uint64_t block = std::max<uint64_t>(
            PositiveDim(dims.outputC0Block > 0 ? dims.outputC0Block : dims.outputC0), 1U);
        splitQuantum = std::max<uint64_t>(SafeMul(PositiveDim(dims.outW), block), 1U);
    }
    const uint64_t unitsPerCore = CeilDiv(CeilDiv(totalOut, splitQuantum), usedCore);
    tiling.set_totalOut(totalOut);
    tiling.set_normalCoreOut(SafeMul(unitsPerCore, splitQuantum));
    tiling.set_splitOut(splitOut);
    tiling.set_splitQuantum(splitQuantum);
    tiling.set_blockDim(static_cast<uint32_t>(usedCore));
    tiling.set_balancedSplit(0U);
}

static uint64_t SelectSpatialFeatureBlocks(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    const bool halfNdhwcD3H3 = inputData.dtypeSize == sizeof(uint16_t) &&
                               inputData.inputFormat == ge::Format::FORMAT_NDHWC &&
                               IsD3H3Dil2PhysicalFeature(inputData, dims);
    if (halfNdhwcD3H3) {
        return PositiveDim(dims.n) + PositiveDim(dims.outD) - 1U;
    }
    if (IsNdc1hwc0D3W3DilD2PhysicalFeature(inputData, dims)) {
        return SafeMul(PositiveDim(dims.n), PositiveDim(dims.outH));
    }
    if (IsNdc1hwc0DilatedWPhysicalFeature(inputData, dims) || IsNcdhwDilatedWFloatFeature(inputData, dims)) {
        return PositiveDim(dims.n);
    }
    if (IsNdc1hwc0D2H3W2Dil2PhysicalFeature(inputData, dims)) {
        const uint64_t dilationGroups = SafeMul(PositiveDim(inputData.dilation[D_DIM]),
                                                PositiveDim(inputData.dilation[H_DIM]));
        return SafeMul(PositiveDim(dims.n), dilationGroups);
    }
    if (IsHOnlyStride3PhysicalRoute(inputData, dims)) {
        const bool halfNdhwc = inputData.dtypeSize == sizeof(uint16_t) &&
                               inputData.inputFormat == ge::Format::FORMAT_NDHWC;
        if (inputData.dtypeSize == sizeof(float)) {
            return PositiveDim(dims.inH);
        }
        return halfNdhwc ? SafeMul(PositiveDim(dims.n), PositiveDim(inputData.stride[H_DIM])) :
                           SafeMul(PositiveDim(dims.n), PositiveDim(dims.outD));
    }
    return 0U;
}

static uint64_t SelectK1FeatureBlocks(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    const bool logicalNcdhwCompactK1 = dims.inputLayout != LAYOUT_NDC1HWC0_VALUE &&
                                       IsNcdhwFloatCompactK1Feature(inputData, dims);
    if (logicalNcdhwCompactK1) {
        return PositiveDim(dims.n);
    }
    const bool logicalHalfK1Small = inputData.dtypeSize == sizeof(uint16_t) &&
                                    dims.inputLayout != LAYOUT_NDC1HWC0_VALUE &&
                                    IsNdhwcCompactK1Feature(inputData, dims);
    if (logicalHalfK1Small) {
        return PositiveDim(dims.n);
    }
    const bool physicalCompactK1 = IsK1IdentityPhysicalRoute(inputData, dims) &&
                                   PositiveDim(dims.c) <= PositiveDim(dims.outputC0Block);
    if (physicalCompactK1) {
        return inputData.dtypeSize == sizeof(float) ? SafeMul(PositiveDim(dims.n), PositiveDim(dims.outD)) :
                                                      PositiveDim(dims.n);
    }
    if (IsNdhwcFloatK1DmaFeature(inputData, dims)) {
        const uint64_t depthTasks = SafeMul(PositiveDim(dims.n), PositiveDim(dims.outD));
        const uint64_t channelBlock = PositiveDim(dims.outputC0Block);
        return PositiveDim(dims.c) <= channelBlock ? std::min(depthTasks, channelBlock) : depthTasks;
    }
    if (IsK1IdentityPhysicalRoute(inputData, dims)) {
        return SafeMul(PositiveDim(dims.n), PositiveDim(dims.outD));
    }
    const bool compactHalfPool2 = inputData.dtypeSize == sizeof(uint16_t) && dims.n == 1 && dims.outD <= 2 &&
                                  PositiveDim(dims.inputC1) >= 8U && IsPool2Stride2PhysicalRoute(inputData, dims);
    return compactHalfPool2 ? PositiveDim(dims.inputC1) : 0U;
}

static uint64_t SelectPool2FeatureBlocks(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    uint64_t physicalBlocks = std::max<uint64_t>(SafeMul(PositiveDim(dims.n), PositiveDim(dims.inputC1)), 1U);
    const bool balancedChannelDepth = PositiveDim(dims.inputC1) == SafeMul(PositiveDim(dims.outD), 2U);
    if (inputData.dtypeSize == sizeof(float) && dims.n == 1 && !balancedChannelDepth) {
        return SafeMul(SafeMul(SafeMul(PositiveDim(dims.n), PositiveDim(dims.outD)), PositiveDim(dims.outH)),
                       PositiveDim(dims.inputC1));
    }
    if (dims.n > 1) {
        return physicalBlocks;
    }
    return SafeMul(physicalBlocks, PositiveDim(dims.outD));
}

static void SetGeneralCoreSplit(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims, uint64_t coreNum,
                                uint64_t ubBlockBytes, uint64_t totalOut, uint64_t splitOut,
                                MaxPool3DTilingData& tiling)
{
    uint64_t quantum = SelectSplitQuantum(inputData, dims, coreNum, ubBlockBytes);
    quantum = std::max<uint64_t>(quantum, 1U);
    if (quantum > splitOut) {
        quantum = 1U;
    }

    const uint64_t units = CeilDiv(splitOut, quantum);
    uint64_t coreLimit = coreNum;
    if (IsNdhwcCompactK1Feature(inputData, dims)) {
        coreLimit = std::min<uint64_t>(coreNum, std::max<uint64_t>(PositiveDim(dims.outputC0Block), 1U));
    } else if (IsNcdhwSmallDepthStride2Feature(inputData, dims)) {
        coreLimit = std::min<uint64_t>(coreNum,
                                       std::max<uint64_t>(CeilDiv(PositiveDim(dims.c), PositiveDim(dims.inD)), 1U));
    }
    const uint64_t targetCore = std::max<uint64_t>(std::min<uint64_t>(coreLimit, units), 1U);
    const uint64_t unitsPerCore = std::max<uint64_t>(CeilDiv(units, targetCore), 1U);
    const uint64_t usedCore = std::max<uint64_t>(CeilDiv(units, unitsPerCore), 1U);
    const uint64_t normalCoreOut = std::min<uint64_t>(SafeMul(unitsPerCore, quantum), splitOut);
    const bool balancedFamily = IsBalancedSplitFeature(inputData, dims);
    const bool balancedSplit = balancedFamily && targetCore > usedCore;

    tiling.set_totalOut(totalOut);
    tiling.set_normalCoreOut(normalCoreOut);
    tiling.set_splitOut(splitOut);
    tiling.set_splitQuantum(quantum);
    tiling.set_blockDim(static_cast<uint32_t>(balancedSplit ? targetCore : usedCore));
    tiling.set_balancedSplit(balancedSplit ? 1U : 0U);
}

static void ComputeCoreSplit(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims, uint64_t coreNum,
                             uint64_t ubBlockBytes, MaxPool3DTilingData& tiling)
{
    const uint64_t totalOut = CalcTotalOut(dims);
    const uint64_t splitOut = CalcCoreSplitOut(dims);
    if (totalOut == 0U || splitOut == 0U || coreNum == 0U) {
        tiling.set_totalOut(totalOut);
        tiling.set_normalCoreOut(0U);
        tiling.set_splitOut(splitOut);
        tiling.set_splitQuantum(1U);
        tiling.set_blockDim(1U);
        tiling.set_balancedSplit(0U);
        return;
    }
    uint64_t featureBlocks = SelectSpatialFeatureBlocks(inputData, dims);
    if (featureBlocks == 0U) {
        featureBlocks = SelectK1FeatureBlocks(inputData, dims);
    }
    if (featureBlocks > 0U) {
        SetFeatureCoreSplit(dims, totalOut, splitOut, featureBlocks, coreNum, tiling);
        return;
    }
    if (IsPool2Stride2PhysicalRoute(inputData, dims)) {
        featureBlocks = SelectPool2FeatureBlocks(inputData, dims);
        SetFeatureCoreSplit(dims, totalOut, splitOut, featureBlocks, coreNum, tiling);
        return;
    }
    SetGeneralCoreSplit(inputData, dims, coreNum, ubBlockBytes, totalOut, splitOut, tiling);
}

static bool IsHalfSingleBatchPool2Feature(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return inputData.dtypeSize == sizeof(uint16_t) && dims.inputLayout == LAYOUT_NDC1HWC0_VALUE &&
           dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && dims.n == 1 && IsPool2Stride2PhysicalRoute(inputData, dims);
}

static bool IsFloatSingleBatchDepthHeavyPool2(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return inputData.dtypeSize == sizeof(float) && dims.inputLayout == LAYOUT_NDC1HWC0_VALUE &&
           dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && dims.n == 1 &&
           PositiveDim(dims.inputC1) < PositiveDim(dims.outD) && IsPool2Stride2PhysicalRoute(inputData, dims);
}

static bool IsFloatSingleBatchChannelHeavyPool2(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    return inputData.dtypeSize == sizeof(float) && dims.inputLayout == LAYOUT_NDC1HWC0_VALUE &&
           dims.outputLayout == LAYOUT_NDC1HWC0_VALUE && dims.n == 1 &&
           PositiveDim(dims.inputC1) >= PositiveDim(dims.outD) && IsPool2Stride2PhysicalRoute(inputData, dims);
}

static bool SelectSingleBatchPool2FeatureMode(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                              uint32_t& selectedMode)
{
    if (IsHalfSingleBatchPool2Feature(inputData, dims)) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_POOL2_HALF_WIDE_CHANNEL;
        return true;
    }
    if (IsFloatSingleBatchDepthHeavyPool2(inputData, dims)) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_POOL2_DEPTH_HEAVY;
        return true;
    }
    if (IsFloatSingleBatchChannelHeavyPool2(inputData, dims)) {
        selectedMode = PositiveDim(dims.inputC1) == SafeMul(PositiveDim(dims.outD), 2U) ?
                           MAX_POOL3_D_TPL_SCH_MODE_POOL2_BALANCED_CHANNEL :
                           MAX_POOL3_D_TPL_SCH_MODE_POOL2_CHANNEL_HEAVY;
        return true;
    }
    return false;
}

static bool SelectK1FeatureMode(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                uint32_t& selectedMode)
{
    if (IsNcdhwFloatCompactK1Feature(inputData, dims)) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_K1_COMPACT_LOGICAL;
        return true;
    }
    if (IsNdhwcCompactK1Feature(inputData, dims)) {
        selectedMode = dims.inputLayout != LAYOUT_NDC1HWC0_VALUE ? MAX_POOL3_D_TPL_SCH_MODE_K1_COMPACT_LOGICAL :
                                                                   MAX_POOL3_D_TPL_SCH_MODE_K1_COMPACT_FEATURE;
        return true;
    }
    if (IsK1IdentityPhysicalRoute(inputData, dims) && PositiveDim(dims.c) <= PositiveDim(dims.outputC0Block)) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_K1_COMPACT_FEATURE;
        return true;
    }
    return false;
}

static bool SelectPrimaryFeatureMode(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                     uint32_t& selectedMode)
{
    return SelectSingleBatchPool2FeatureMode(inputData, dims, selectedMode) ||
           SelectK1FeatureMode(inputData, dims, selectedMode);
}

static bool SelectDilatedSpatialFeatureMode(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                            uint32_t& selectedMode)
{
    if (IsNdc1hwc0D3W3DilD2PhysicalFeature(inputData, dims)) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_D3W3_FEATURE;
        return true;
    }
    if (IsD3H3Dil2PhysicalFeature(inputData, dims)) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_D3H3_PHYSICAL_FEATURE;
        return true;
    }
    if (IsNdc1hwc0D2H3W2Dil2PhysicalFeature(inputData, dims)) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_D2H3W2_PHYSICAL_FEATURE;
        return true;
    }
    if (IsHOnlyStride3PhysicalRoute(inputData, dims)) {
        selectedMode = inputData.inputFormat == ge::Format::FORMAT_NDHWC || inputData.dtypeSize == sizeof(float) ?
                           MAX_POOL3_D_TPL_SCH_MODE_H_ONLY_HALF_NDHWC_FEATURE :
                           MAX_POOL3_D_TPL_SCH_MODE_H_ONLY_FEATURE;
        return true;
    }
    return false;
}

static bool SelectCompactSpatialFeatureMode(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                            uint32_t& selectedMode)
{
    const bool halfTinyK3 = inputData.dtypeSize == sizeof(uint16_t) &&
                            (IsTinyK3PhysicalRoute(inputData, dims) || IsNdhwcTinyK3Feature(inputData, dims) ||
                             IsNcdhwTinyK3Feature(inputData, dims));
    if (halfTinyK3) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_TINY_K3_HALF_GROUPED;
        return true;
    }
    const bool multiBatchStride2 = dims.n > 1 && IsPool2Stride2PhysicalRoute(inputData, dims);
    if (multiBatchStride2) {
        selectedMode = inputData.dtypeSize == sizeof(uint16_t) && inputData.inputFormat == ge::Format::FORMAT_NDHWC ?
                           MAX_POOL3_D_TPL_SCH_MODE_POOL2_MULTI_BATCH_HALF_NDHWC_FEATURE :
                           MAX_POOL3_D_TPL_SCH_MODE_POOL2_MULTI_BATCH_FEATURE;
        return true;
    }
    if (IsNcdhwDilatedWFloatFeature(inputData, dims)) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_DILATED_W_FLOAT_NCDHW_FEATURE;
        return true;
    }
    if (IsNdc1hwc0DilatedWPhysicalFeature(inputData, dims)) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_DILATED_W_FEATURE;
        return true;
    }
    if (IsK1IdentityPhysicalRoute(inputData, dims)) {
        selectedMode = inputData.dtypeSize == sizeof(float) ? MAX_POOL3_D_TPL_SCH_MODE_K1_WIDE_FLOAT_PHYSICAL_FEATURE :
                                                              MAX_POOL3_D_TPL_SCH_MODE_K1_WIDE_FEATURE;
        return true;
    }
    if (IsNdhwcFloatK1DmaFeature(inputData, dims) && PositiveDim(dims.c) > PositiveDim(dims.outputC0Block)) {
        selectedMode = MAX_POOL3_D_TPL_SCH_MODE_K1_WIDE_FEATURE;
        return true;
    }
    return false;
}

static bool SelectSpatialFeatureMode(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims,
                                     uint32_t& selectedMode)
{
    return SelectDilatedSpatialFeatureMode(inputData, dims, selectedMode) ||
           SelectCompactSpatialFeatureMode(inputData, dims, selectedMode);
}

static uint32_t SelectFallbackMode(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    if (IsTinyK3PhysicalRoute(inputData, dims) || IsPool2Stride2PhysicalRoute(inputData, dims) ||
        IsNdhwcTinyK3Feature(inputData, dims) || IsNcdhwTinyK3Feature(inputData, dims)) {
        return MAX_POOL3_D_TPL_SCH_MODE_TINY_K3;
    }
    if (inputData.inputFormat == ge::Format::FORMAT_NDHWC && IsNdc1hwc0TinyK3ValidSpecial(inputData, dims)) {
        return MAX_POOL3_D_TPL_SCH_MODE_TINY_K3;
    }
    if (IsNcdhwSmallDepthStride2Feature(inputData, dims)) {
        return MAX_POOL3_D_TPL_SCH_MODE_NCDHW_SMALL_DEPTH_STRIDE2;
    }
    if (IsNcdhwPool2Feature(inputData, dims)) {
        return MAX_POOL3_D_TPL_SCH_MODE_NCDHW_POOL2_FEATURE;
    }
    if (dims.outputLayout == LAYOUT_NDC1HWC0_VALUE) {
        if (inputData.dtypeSize == sizeof(float) && IsCompactNdc1hwc0Prefix(dims) &&
            (IsNdc1hwc0NcdhwD3H3Dil2ReusablePlane(inputData, dims) ||
             IsNdc1hwc0NdhwcD3H3Dil2ReusablePlane(inputData, dims))) {
            return MAX_POOL3_D_TPL_SCH_MODE_GENERAL;
        }
        return MAX_POOL3_D_TPL_SCH_MODE_NDC1HWC0;
    }
    if (IsPool2Stride2NoPad(inputData)) {
        if (inputData.inputFormat == ge::Format::FORMAT_NCDHW) {
            return MAX_POOL3_D_TPL_SCH_MODE_NCDHW_STRIDE2;
        }
        if (inputData.inputFormat == ge::Format::FORMAT_NDHWC) {
            return MAX_POOL3_D_TPL_SCH_MODE_NDHWC_STRIDE2;
        }
    }
    return MAX_POOL3_D_TPL_SCH_MODE_GENERAL;
}

static uint32_t SelectScheduleMode(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims)
{
    uint32_t selectedMode = MAX_POOL3_D_TPL_SCH_MODE_GENERAL;
    if (SelectPrimaryFeatureMode(inputData, dims, selectedMode)) {
        return selectedMode;
    }
    if (SelectSpatialFeatureMode(inputData, dims, selectedMode)) {
        return selectedMode;
    }
    return SelectFallbackMode(inputData, dims);
}

static void FillInputStorageDims(const gert::Shape& inputStorageShape, bool inputStorageNdc1hwc0,
                                 MaxPool3DLogicalDims& dims)
{
    if (inputStorageShape.GetDimNum() == NDC1HWC0_DIMS || inputStorageNdc1hwc0) {
        dims.inputLayout = LAYOUT_NDC1HWC0_VALUE;
        if (inputStorageShape.GetDimNum() == NDC1HWC0_DIMS) {
            dims.inputC1 = inputStorageShape.GetDim(2);
            dims.inputC0 = inputStorageShape.GetDim(5);
            dims.inputC0Block = dims.inputC0;
        }
    }
}

static void FillOutputStorageDims(const gert::Shape& outputShape, MaxPool3DLogicalDims& dims)
{
    const bool outputNdc1hwc0 = outputShape.GetDimNum() == NDC1HWC0_DIMS;
    dims.outputLayout = outputNdc1hwc0 ? LAYOUT_NDC1HWC0_VALUE : LAYOUT_ND_VALUE;
    dims.outputD = outputNdc1hwc0 ? outputShape.GetDim(1) : dims.outD;
    dims.outputH = outputNdc1hwc0 ? outputShape.GetDim(3) : dims.outH;
    dims.outputW = outputNdc1hwc0 ? outputShape.GetDim(4) : dims.outW;
    dims.outputC1 = outputNdc1hwc0 ? outputShape.GetDim(2) : 1;
    dims.outputC0 = outputNdc1hwc0 ? outputShape.GetDim(5) : dims.c;
    dims.outputC0Block = outputNdc1hwc0 ? ResolveNdc1hwc0Block(dims.outputC0) : dims.outputC0;
}

static void FillDefaultInputBlocking(MaxPool3DLogicalDims& dims)
{
    if (dims.inputLayout == LAYOUT_NDC1HWC0_VALUE && dims.inputC0 <= 0) {
        dims.inputC0 = NDC1HWC0_LOGICAL_C0;
        dims.inputC0Block = dims.inputC0;
        dims.inputC1 = static_cast<int64_t>(CeilDiv(PositiveDim(dims.c), PositiveDim(dims.inputC0)));
    }
}

static void FillNdhwcLogicalDims(const Pool3DInputInfo& inputData, const gert::Shape& inputShape,
                                 const gert::Shape& outputShape, MaxPool3DLogicalDims& dims)
{
    dims.n = inputShape.GetDim(0);
    dims.inD = inputShape.GetDim(1);
    dims.inH = inputShape.GetDim(2);
    dims.inW = inputShape.GetDim(3);
    dims.c = inputShape.GetDim(4);
    if (outputShape.GetDimNum() == NDC1HWC0_DIMS) {
        dims.outD = outputShape.GetDim(1);
        dims.outH = outputShape.GetDim(3);
        dims.outW = outputShape.GetDim(4);
    } else {
        dims.outD = outputShape.GetDim(1);
        dims.outH = outputShape.GetDim(2);
        dims.outW = outputShape.GetDim(3);
    }
    FillOutputStorageDims(outputShape, dims);
    FillDefaultInputBlocking(dims);
    RefineLogicalOutDimsForNdc1hwc0(inputData, dims);
}

static void FillNcdhwLogicalDims(const Pool3DInputInfo& inputData, const gert::Shape& inputShape,
                                 const gert::Shape& outputShape, MaxPool3DLogicalDims& dims)
{
    dims.n = inputShape.GetDim(0);
    dims.c = inputShape.GetDim(1);
    dims.inD = inputShape.GetDim(2);
    dims.inH = inputShape.GetDim(3);
    dims.inW = inputShape.GetDim(4);
    if (outputShape.GetDimNum() == NDC1HWC0_DIMS) {
        dims.outD = outputShape.GetDim(1);
        dims.outH = outputShape.GetDim(3);
        dims.outW = outputShape.GetDim(4);
    } else {
        dims.outD = outputShape.GetDim(2);
        dims.outH = outputShape.GetDim(3);
        dims.outW = outputShape.GetDim(4);
    }
    FillOutputStorageDims(outputShape, dims);
    FillDefaultInputBlocking(dims);
    RefineLogicalOutDimsForNdc1hwc0(inputData, dims);
}

static ge::graphStatus FillLogicalDims(gert::TilingContext* context, const Pool3DInputInfo& inputData,
                                       MaxPool3DLogicalDims& dims)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto outputY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputY);

    const auto inputStorageShape = Ops::NN::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    auto inputShape = Ops::NN::OpTiling::EnsureNotScalar(inputX->GetOriginShape());
    if (inputShape.GetDimNum() != NCDHW_DIMS && inputStorageShape.GetDimNum() == NCDHW_DIMS) {
        inputShape = inputStorageShape;
    }
    const auto outputShape = Ops::NN::OpTiling::EnsureNotScalar(outputY->GetStorageShape());
    OP_CHECK_IF(inputShape.GetDimNum() != NCDHW_DIMS ||
                    (outputShape.GetDimNum() != NCDHW_DIMS && outputShape.GetDimNum() != NDC1HWC0_DIMS),
                OP_LOGE(context->GetNodeName(), "MaxPool3D only supports 5D input and 5D/6D output tensors."),
                return ge::GRAPH_FAILED);

    const bool inputStorageNdc1hwc0 = inputDesc->GetFormat().GetStorageFormat() == ge::Format::FORMAT_NDC1HWC0;
    FillInputStorageDims(inputStorageShape, inputStorageNdc1hwc0, dims);
    if (inputData.inputFormat == ge::Format::FORMAT_NDHWC) {
        FillNdhwcLogicalDims(inputData, inputShape, outputShape, dims);
    } else {
        FillNcdhwLogicalDims(inputData, inputShape, outputShape, dims);
    }
    return ge::GRAPH_SUCCESS;
}

static void FillKernelTiling(const Pool3DInputInfo& inputData, const MaxPool3DLogicalDims& dims, uint64_t coreNum,
                             uint64_t ubBlockBytes, MaxPool3DTilingData& tiling)
{
    tiling.set_n(dims.n);
    tiling.set_inD(dims.inD);
    tiling.set_inH(dims.inH);
    tiling.set_inW(dims.inW);
    tiling.set_c(dims.c);
    tiling.set_outD(dims.outD);
    tiling.set_outH(dims.outH);
    tiling.set_outW(dims.outW);
    tiling.set_kD(inputData.kernelSize[D_DIM]);
    tiling.set_kH(inputData.kernelSize[H_DIM]);
    tiling.set_kW(inputData.kernelSize[W_DIM]);
    tiling.set_sD(inputData.stride[D_DIM]);
    tiling.set_sH(inputData.stride[H_DIM]);
    tiling.set_sW(inputData.stride[W_DIM]);
    tiling.set_padFront(inputData.pad[FRONT_PAD_INDEX]);
    tiling.set_padTop(inputData.pad[TOP_PAD_INDEX]);
    tiling.set_padLeft(inputData.pad[LEFT_PAD_INDEX]);
    tiling.set_dilationD(inputData.dilation[D_DIM]);
    tiling.set_dilationH(inputData.dilation[H_DIM]);
    tiling.set_dilationW(inputData.dilation[W_DIM]);
    tiling.set_dataFormat(inputData.inputFormat == ge::Format::FORMAT_NDHWC ? FORMAT_NDHWC_VALUE : FORMAT_NCDHW_VALUE);
    tiling.set_outputLayout(dims.outputLayout);
    tiling.set_outputD(dims.outputD);
    tiling.set_outputH(dims.outputH);
    tiling.set_outputW(dims.outputW);
    tiling.set_outputC1(dims.outputC1);
    tiling.set_outputC0(dims.outputC0);
    tiling.set_outputC0Block(dims.outputC0Block);
    tiling.set_inputLayout(dims.inputLayout);
    tiling.set_inputC1(dims.inputC1);
    tiling.set_inputC0(dims.inputLayout == LAYOUT_NDC1HWC0_VALUE ? dims.inputC0 : dims.c);
    tiling.set_inputC0Block(dims.inputLayout == LAYOUT_NDC1HWC0_VALUE ? dims.inputC0Block : dims.c);
    ComputeCoreSplit(inputData, dims, coreNum, ubBlockBytes, tiling);
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CommitTiling(gert::TilingContext* context, MaxPool3DTilingData& tiling, uint32_t selectedMode)
{
    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    auto tilingBuffer = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingBuffer);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tiling.GetDataSize(),
                OP_LOGE(context->GetNodeName(), "Raw tiling buffer capacity is insufficient."),
                return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(tilingBuffer, rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(std::max<uint32_t>(tiling.get_blockDim(), 1U));
    context->SetTilingKey(GET_TPL_TILING_KEY(selectedMode));
    return SetWorkspace(context);
}

} // namespace

static ge::graphStatus Tiling4MaxPool3D(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MaxPool3D", "Tiling context is null."), return ge::GRAPH_FAILED);
    Pool3DInputInfo inputData;
    OP_CHECK_IF(GetMaxPool3DShapeAttrsInfo(context, inputData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetMaxPool3DShapeAttrsInfo failed."), return ge::GRAPH_FAILED);

    MaxPool3DLogicalDims dims;
    OP_CHECK_IF(FillLogicalDims(context, inputData, dims) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "FillLogicalDims failed."), return ge::GRAPH_FAILED);
    uint64_t coreNum = 0;
    OP_CHECK_IF(GetMaxPool3DPlatformInfo(context, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetMaxPool3DPlatformInfo failed."), return ge::GRAPH_FAILED);
    const int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0, OP_LOGE(context->GetNodeName(), "UB block size must be greater than 0."),
                return ge::GRAPH_FAILED);

    MaxPool3DTilingData tiling;
    FillKernelTiling(inputData, dims, coreNum, static_cast<uint64_t>(ubBlockSize), tiling);
    const uint32_t selectedMode = SelectScheduleMode(inputData, dims);
    return CommitTiling(context, tiling, selectedMode);
}

static ge::graphStatus TilingPrepare4MaxPool3D(gert::TilingParseContext* context)
{
    OP_CHECK_IF(nullptr == context, OP_LOGE("MaxPool3D", "Context is null"), return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<MaxPool3DCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    OP_CHECK_IF(compileInfoPtr->coreNum == 0, OP_LOGE("MaxPool3D", "Vector core count must be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compileInfoPtr->ubSize == 0, OP_LOGE("MaxPool3D", "UB size must be greater than 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPool3D).Tiling(Tiling4MaxPool3D).TilingParse<MaxPool3DCompileInfo>(TilingPrepare4MaxPool3D);

} // namespace optiling
