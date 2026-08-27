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
 * \file bn_inference_tiling_arch35.cpp
 * \brief Checked arch35 tiling for BNInference.
 */
#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <limits>
#include <string>
#include "bn_inference_tiling_arch35.h"
#include "../bn_inference_dtype.h"
#include "../../op_kernel/arch35/bn_inference_tiling_data.h"
#include "../../op_kernel/arch35/bn_inference_tiling_key.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

using namespace optiling;

BEGIN_TILING_DATA_DEF(BNInferenceTilingDataDef)
TILING_DATA_FIELD_DEF(int64_t, baseTilesPerCore);
TILING_DATA_FIELD_DEF(int64_t, extraCoreCount);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, c);
TILING_DATA_FIELD_DEF(int64_t, inner);
TILING_DATA_FIELD_DEF(int64_t, tileElements);
TILING_DATA_FIELD_DEF(int64_t, tileRows);
TILING_DATA_FIELD_DEF(int64_t, paramTileLen);
TILING_DATA_FIELD_DEF(int64_t, paramCacheLen);
TILING_DATA_FIELD_DEF(int64_t, innerTileCount);
TILING_DATA_FIELD_DEF(float, epsilon);
// Explicit ABI padding: keep the serialized host layout identical to the device-side mirror.
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BNInference_1000, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_2000, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_2001, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_2002, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_2003, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_2100, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_2101, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_2102, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_2103, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_3000, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_3001, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_3002, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_3003, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_3100, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_3101, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_3102, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_3103, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_4000, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_4001, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_4003, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_4100, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_4101, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_4103, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_5000, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_5001, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_5003, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_5100, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_5101, BNInferenceTilingDataDef);
REGISTER_TILING_DATA_CLASS(BNInference_5103, BNInferenceTilingDataDef);

namespace {
constexpr int64_t INPUT_X = 0;
constexpr int64_t INPUT_MEAN = 1;
constexpr int64_t INPUT_VARIANCE = 2;
constexpr int64_t INPUT_MOMENTUM = 3;
constexpr int64_t INPUT_SCALE = 4;
constexpr int64_t INPUT_OFFSET = 5;
constexpr int64_t OUTPUT_Y = 0;
constexpr int64_t ATTR_EPSILON = 0;
constexpr int64_t ATTR_MODE = 2;
constexpr int64_t PIPE_META_RESERVE = 16 * 1024;
constexpr int64_t MAX_GENERIC_C_VECTORS = 8;
constexpr int64_t SMALL_CF_Q_LIMIT = 32;
constexpr int64_t SMALL_CL_C_LIMIT = 8;
constexpr int64_t SMALL_CL_MIN_B = 65536;
constexpr int64_t MIN_SMALL_CF_N_PER_CORE = 2;
constexpr int64_t FP32_BYTES = 4;
constexpr size_t SIMPLIFIED_KEY_CAPACITY = 100;

bool TryAddNonNegative(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool TryMulNonNegative(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool TryCeilDivNonNegative(int64_t value, int64_t divisor, int64_t& result)
{
    if (value < 0 || divisor <= 0) {
        return false;
    }
    result = value / divisor + ((value % divisor) != 0 ? 1 : 0);
    return true;
}

bool TryAlignUpNonNegative(int64_t value, int64_t align, int64_t& result)
{
    if (value < 0 || align <= 0) {
        return false;
    }
    const int64_t remainder = value % align;
    return remainder == 0 ? (result = value, true) : TryAddNonNegative(value, align - remainder, result);
}

bool TryAlignedTensorBytes(int64_t elements, int64_t elementBytes, int64_t align, int64_t& result)
{
    int64_t bytes = 0;
    return TryMulNonNegative(elements, elementBytes, bytes) && TryAlignUpNonNegative(bytes, align, result);
}

int64_t GetDtypeBytes(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return 4;
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return 2;
    }
    return 0;
}

bool IsParameterShape(const gert::StorageShape* shape, int64_t c)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() == 1 && shape->GetStorageShape().GetDim(0) == c;
}

ge::Format GetFeatureBinaryFormat(ge::Format format)
{
    const ge::Format primaryFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(format));
    // The kernel implementation is format-macro independent. Public 5-D layouts share the ND binary while
    // tiling keeps the real descriptor format to select the correct channel axis.
    if (primaryFormat == ge::FORMAT_NCDHW || primaryFormat == ge::FORMAT_NDHWC) {
        return ge::FORMAT_ND;
    }
    return primaryFormat;
}
} // namespace

namespace optiling {
ge::graphStatus BNInferenceTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = static_cast<int64_t>(platform.GetCoreNumAiv());
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        OP_CHECK_IF(ubSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    OP_LOGE(context_, "BNInference: UB size exceeds int64"), return ge::GRAPH_FAILED);
        ubSize_ = static_cast<int64_t>(ubSize);
        vectorLength_ = static_cast<int64_t>(Ops::Base::GetVRegSize(context_));
        blockSize_ = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    } else {
        auto compileInfo = reinterpret_cast<const BNInferenceCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
        vectorLength_ = compileInfo->vectorLength;
        blockSize_ = compileInfo->blockSize;
    }
    OP_CHECK_IF(
        coreNum_ <= 0 || ubSize_ <= PIPE_META_RESERVE || vectorLength_ <= 0 || blockSize_ <= 0,
        OP_LOGE(context_,
                "BNInference: invalid platform values core=%" PRId64 " ub=%" PRId64 " vreg=%" PRId64 " block=%" PRId64,
                coreNum_, ubSize_, vectorLength_, blockSize_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        vectorLength_ % FP32_BYTES != 0,
        OP_LOGE(context_, "BNInference: vector length %" PRId64 " is not divisible by fp32 bytes", vectorLength_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferenceTiling::ReadInputInfo(BNInferenceInputInfo& info)
{
    info.xDesc = context_->GetInputDesc(INPUT_X);
    info.meanDesc = context_->GetInputDesc(INPUT_MEAN);
    info.varianceDesc = context_->GetInputDesc(INPUT_VARIANCE);
    info.momentumDesc = context_->GetInputDesc(INPUT_MOMENTUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, info.xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, info.meanDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, info.varianceDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, info.momentumDesc);
    info.xShape = context_->GetRequiredInputShape(INPUT_X);
    info.meanShape = context_->GetRequiredInputShape(INPUT_MEAN);
    info.varianceShape = context_->GetRequiredInputShape(INPUT_VARIANCE);
    info.momentumShape = context_->GetRequiredInputShape(INPUT_MOMENTUM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, info.xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, info.meanShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, info.varianceShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, info.momentumShape);
    info.scaleDesc = context_->GetOptionalInputDesc(INPUT_SCALE);
    info.offsetDesc = context_->GetOptionalInputDesc(INPUT_OFFSET);
    info.scaleShape = context_->GetOptionalInputShape(INPUT_SCALE);
    info.offsetShape = context_->GetOptionalInputShape(INPUT_OFFSET);
    if ((info.scaleDesc == nullptr) != (info.scaleShape == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "scale", "descriptor/shape mismatch",
                                              "descriptor and storage shape must both be present or both be absent");
        return ge::GRAPH_FAILED;
    }
    if ((info.offsetDesc == nullptr) != (info.offsetShape == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "offset", "descriptor/shape mismatch",
                                              "descriptor and storage shape must both be present or both be absent");
        return ge::GRAPH_FAILED;
    }
    hasScale_ = info.scaleDesc != nullptr;
    hasOffset_ = info.offsetDesc != nullptr;
    info.yDesc = context_->GetOutputDesc(OUTPUT_Y);
    info.yShape = context_->GetOutputShape(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, info.yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, info.yShape);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferenceTiling::ValidateFeatureTensor(const BNInferenceInputInfo& info, bool& hasZeroDim)
{
    const gert::Shape& xShape = info.xShape->GetStorageShape();
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());
    const ge::Format xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(info.xDesc->GetStorageFormat()));
    const ge::Format xOriginFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(info.xDesc->GetOriginFormat()));
    const bool validFormat = xFormat == ge::FORMAT_NCHW || xFormat == ge::FORMAT_NHWC || xFormat == ge::FORMAT_NCDHW ||
                             xFormat == ge::FORMAT_NDHWC || xFormat == ge::FORMAT_ND;
    if (!validFormat) {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", ge::TypeUtils::FormatToSerialString(xFormat).c_str(),
                                   "NCHW, NHWC, NCDHW, NDHWC or ND");
        return ge::GRAPH_FAILED;
    }
    bool validRank = (xFormat == ge::FORMAT_ND && (rank == 4 || rank == 5)) ||
                     ((xFormat == ge::FORMAT_NCHW || xFormat == ge::FORMAT_NHWC) && rank == 4) ||
                     ((xFormat == ge::FORMAT_NCDHW || xFormat == ge::FORMAT_NDHWC) && rank == 5);
    if (xFormat == ge::FORMAT_ND) {
        const bool validOriginFormat = xOriginFormat == ge::FORMAT_ND || xOriginFormat == ge::FORMAT_NCHW ||
                                       xOriginFormat == ge::FORMAT_NHWC || xOriginFormat == ge::FORMAT_NCDHW ||
                                       xOriginFormat == ge::FORMAT_NDHWC;
        if (!validOriginFormat) {
            OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x origin",
                                       ge::TypeUtils::FormatToSerialString(xOriginFormat).c_str(),
                                       "ND, NCHW, NHWC, NCDHW or NDHWC");
            return ge::GRAPH_FAILED;
        }
        const bool validOriginRank = xOriginFormat == ge::FORMAT_ND ||
                                     ((xOriginFormat == ge::FORMAT_NCHW || xOriginFormat == ge::FORMAT_NHWC) &&
                                      rank == 4) ||
                                     ((xOriginFormat == ge::FORMAT_NCDHW || xOriginFormat == ge::FORMAT_NDHWC) &&
                                      rank == 5);
        validRank = validRank && validOriginRank;
    }
    if (!validRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(rank).c_str(),
                                                 "NCHW/NHWC origins require rank 4, NCDHW/NDHWC origins require "
                                                 "rank 5, and an ND origin permits rank 4 or 5");
        return ge::GRAPH_FAILED;
    }
    hasZeroDim = false;
    for (int64_t i = 0; i < rank; ++i) {
        if (xShape.GetDim(i) < 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                                  "runtime dimensions must be non-negative");
            return ge::GRAPH_FAILED;
        }
        hasZeroDim = hasZeroDim || xShape.GetDim(i) == 0;
    }
    // canndev preserves the logical origin format when a public layout is materialized as ND storage.
    // A plain ND origin retains the legacy BNInferenceD plane-contiguous fallback (C is dimension 1).
    channelLast_ = xFormat == ge::FORMAT_NHWC || xFormat == ge::FORMAT_NDHWC ||
                   (xFormat == ge::FORMAT_ND &&
                    (xOriginFormat == ge::FORMAT_NHWC || xOriginFormat == ge::FORMAT_NDHWC));
    n_ = xShape.GetDim(0);
    c_ = channelLast_ ? xShape.GetDim(rank - 1) : xShape.GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferenceTiling::ValidateOutputTensor(const BNInferenceInputInfo& info) const
{
    const gert::Shape& xShape = info.xShape->GetStorageShape();
    const gert::Shape& yShape = info.yShape->GetStorageShape();
    bool sameShape = yShape.GetDimNum() == xShape.GetDimNum();
    for (size_t i = 0; sameShape && i < xShape.GetDimNum(); ++i) {
        sameShape = yShape.GetDim(i) == xShape.GetDim(i);
    }
    if (!sameShape) {
        const std::string shapes = Ops::Base::ToString(xShape) + " and " + Ops::Base::ToString(yShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "x and y", shapes.c_str(),
                                               "y shape must equal x shape");
        return ge::GRAPH_FAILED;
    }
    if (info.yDesc->GetStorageFormat() != info.xDesc->GetStorageFormat()) {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "y",
                                   ge::TypeUtils::FormatToSerialString(info.yDesc->GetStorageFormat()).c_str(),
                                   ge::TypeUtils::FormatToSerialString(info.xDesc->GetStorageFormat()).c_str());
        return ge::GRAPH_FAILED;
    }
    const ge::Format xStorageFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(info.xDesc->GetStorageFormat()));
    const ge::Format xOriginFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(info.xDesc->GetOriginFormat()));
    const ge::Format yOriginFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(info.yDesc->GetOriginFormat()));
    if (xStorageFormat == ge::FORMAT_ND && yOriginFormat != xOriginFormat) {
        const std::string formats = ge::TypeUtils::FormatToSerialString(yOriginFormat) + " and " +
                                    ge::TypeUtils::FormatToSerialString(xOriginFormat);
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context_->GetNodeName(), "y and x origin", formats.c_str(),
                                               "y and x must have the same logical origin format");
        return ge::GRAPH_FAILED;
    }
    if (info.yDesc->GetDataType() != info.xDesc->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "y",
                                  ge::TypeUtils::DataTypeToSerialString(info.yDesc->GetDataType()).c_str(),
                                  ge::TypeUtils::DataTypeToSerialString(info.xDesc->GetDataType()).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferenceTiling::ValidateParameterTensors(const BNInferenceInputInfo& info) const
{
    const auto checkNdFormat = [this](const gert::CompileTimeTensorDesc* desc, const char* name) {
        if (desc->GetStorageFormat() == ge::FORMAT_ND) {
            return true;
        }
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), name,
                                   ge::TypeUtils::FormatToSerialString(desc->GetStorageFormat()).c_str(), "ND");
        return false;
    };
    if (!checkNdFormat(info.meanDesc, "mean") || !checkNdFormat(info.varianceDesc, "variance") ||
        !checkNdFormat(info.momentumDesc, "momentum") || (hasScale_ && !checkNdFormat(info.scaleDesc, "scale")) ||
        (hasOffset_ && !checkNdFormat(info.offsetDesc, "offset"))) {
        return ge::GRAPH_FAILED;
    }
    if (!IsParameterShape(info.meanShape, c_) || !IsParameterShape(info.varianceShape, c_)) {
        const std::string shapes = Ops::Base::ToString(info.meanShape->GetStorageShape()) + " and " +
                                   Ops::Base::ToString(info.varianceShape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "mean and variance", shapes.c_str(),
                                               "both shapes must be [C]");
        return ge::GRAPH_FAILED;
    }
    if (hasScale_ && !IsParameterShape(info.scaleShape, c_)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "scale",
                                              Ops::Base::ToString(info.scaleShape->GetStorageShape()).c_str(),
                                              "shape must be [C]");
        return ge::GRAPH_FAILED;
    }
    if (hasOffset_ && !IsParameterShape(info.offsetShape, c_)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "offset",
                                              Ops::Base::ToString(info.offsetShape->GetStorageShape()).c_str(),
                                              "shape must be [C]");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& momentumShape = info.momentumShape->GetStorageShape();
    const bool validMomentum = momentumShape.GetDimNum() == 0 ||
                               (momentumShape.GetDimNum() == 1 &&
                                (momentumShape.GetDim(0) == 1 || momentumShape.GetDim(0) == c_));
    if (!validMomentum) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "momentum",
                                              Ops::Base::ToString(momentumShape).c_str(),
                                              "shape must be scalar [], [1] or [C]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferenceTiling::ResolveDtypes(const BNInferenceInputInfo& info)
{
    const ge::DataType xDtype = info.xDesc->GetDataType();
    const ge::DataType meanDtype = info.meanDesc->GetDataType();
    const ge::DataType varianceDtype = info.varianceDesc->GetDataType();
    const ge::DataType momentumDtype = info.momentumDesc->GetDataType();
    bool matched = false;
    for (const auto& combination : BNInferenceSupport::DTYPE_COMBINATIONS) {
        if (xDtype == combination.x && meanDtype == combination.statistics && varianceDtype == combination.statistics &&
            momentumDtype == combination.momentum &&
            (!hasScale_ || info.scaleDesc->GetDataType() == combination.affine) &&
            (!hasOffset_ || info.offsetDesc->GetDataType() == combination.affine)) {
            matched = true;
            break;
        }
    }
    if (!matched) {
        const std::string dtypes = ge::TypeUtils::DataTypeToSerialString(xDtype) + ", " +
                                   ge::TypeUtils::DataTypeToSerialString(meanDtype) + ", " +
                                   ge::TypeUtils::DataTypeToSerialString(varianceDtype) + ", " +
                                   ge::TypeUtils::DataTypeToSerialString(momentumDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x, mean, variance and momentum",
                                               dtypes.c_str(), "all inputs must match one supported dtype row");
        return ge::GRAPH_FAILED;
    }
    xBytes_ = GetDtypeBytes(xDtype);
    meanBytes_ = GetDtypeBytes(meanDtype);
    varianceBytes_ = GetDtypeBytes(varianceDtype);
    momentumBytes_ = GetDtypeBytes(momentumDtype);
    scaleBytes_ = hasScale_ ? GetDtypeBytes(info.scaleDesc->GetDataType()) : 0;
    offsetBytes_ = hasOffset_ ? GetDtypeBytes(info.offsetDesc->GetDataType()) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferenceTiling::ReadAttributesAndShape(const BNInferenceInputInfo& info, bool hasZeroDim)
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilon = attrs->GetFloat(ATTR_EPSILON);
    epsilon_ = epsilon == nullptr ? 1e-5f : *epsilon;
    const int64_t* mode = attrs->GetInt(ATTR_MODE);
    preFolded_ = mode != nullptr && *mode == 0;
    if (preFolded_ && hasOffset_ && !hasScale_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "offset", "present while scale is absent",
                                              "mode=0 requires scale when offset is present");
        return ge::GRAPH_FAILED;
    }
    empty_ = hasZeroDim;
    if (empty_) {
        totalElements_ = 0;
        inner_ = 0;
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape& xShape = info.xShape->GetStorageShape();
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());
    inner_ = 1;
    const int64_t firstInnerAxis = channelLast_ ? 0 : 2;
    const int64_t lastInnerAxis = channelLast_ ? rank - 1 : rank;
    for (int64_t i = firstInnerAxis; i < lastInnerAxis; ++i) {
        if (!TryMulNonNegative(inner_, xShape.GetDim(i), inner_)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                                  "logical row product must fit in int64");
            return ge::GRAPH_FAILED;
        }
    }
    int64_t outer = channelLast_ ? inner_ : n_;
    if (!channelLast_ && !TryMulNonNegative(outer, c_, outer)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                              "N*C product must fit in int64");
        return ge::GRAPH_FAILED;
    }
    if (!TryMulNonNegative(outer, channelLast_ ? c_ : inner_, totalElements_)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                              "element count must fit in int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferenceTiling::ValidateAndReadInputs()
{
    BNInferenceInputInfo info;
    bool hasZeroDim = false;
    if (ReadInputInfo(info) != ge::GRAPH_SUCCESS || ValidateFeatureTensor(info, hasZeroDim) != ge::GRAPH_SUCCESS ||
        ValidateOutputTensor(info) != ge::GRAPH_SUCCESS || ValidateParameterTensors(info) != ge::GRAPH_SUCCESS ||
        ResolveDtypes(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ReadAttributesAndShape(info, hasZeroDim);
}

bool BNInferenceTiling::GetParamLedger(int64_t paramLen, int64_t cacheLen, bool packed, int64_t& fixedBytes) const
{
    fixedBytes = 0;
    const auto addSource = [this, paramLen, &fixedBytes](int64_t elementBytes) {
        int64_t aligned = 0;
        int64_t doubled = 0;
        int64_t total = 0;
        return TryAlignedTensorBytes(paramLen, elementBytes, blockSize_, aligned) &&
               TryMulNonNegative(aligned, 2, doubled) && TryAddNonNegative(fixedBytes, doubled, total) &&
               (fixedBytes = total, true);
    };
    if (!addSource(meanBytes_) || !addSource(varianceBytes_) || (hasScale_ && !addSource(scaleBytes_)) ||
        (hasOffset_ && !addSource(offsetBytes_))) {
        return false;
    }

    if (!preFolded_ && !hasScale_ && !hasOffset_) {
        int64_t momentumBytes = 0;
        int64_t total = 0;
        if (!TryAlignedTensorBytes(1, momentumBytes_, blockSize_, momentumBytes) ||
            !TryAddNonNegative(fixedBytes, momentumBytes, total)) {
            return false;
        }
        fixedBytes = total;
    }

    int64_t cacheBytes = 0;
    int64_t cacheCount = (packed ? 3 : 2) + (hasScale_ ? 1 : 0) + (hasOffset_ ? 1 : 0);
    int64_t allCacheBytes = 0;
    int64_t total = 0;
    return TryAlignedTensorBytes(cacheLen, FP32_BYTES, blockSize_, cacheBytes) &&
           TryMulNonNegative(cacheBytes, cacheCount, allCacheBytes) &&
           TryAddNonNegative(fixedBytes, allCacheBytes, total) && (fixedBytes = total, true);
}

bool BNInferenceTiling::TryGetPackedRows(int64_t totalRows, int64_t rowElements, int64_t unavailable,
                                         int64_t& rows) const
{
    int64_t bytesPerRow = 0;
    int64_t fourBytesPerRow = 0;
    if (!TryMulNonNegative(rowElements, xBytes_, bytesPerRow) || !TryMulNonNegative(bytesPerRow, 4, fourBytesPerRow) ||
        fourBytesPerRow <= 0) {
        return false;
    }
    rows = std::min(totalRows, (ubSize_ - unavailable) / fourBytesPerRow);
    rows = std::min(rows, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) / bytesPerRow);
    rows = std::min(rows, static_cast<int64_t>(std::numeric_limits<uint16_t>::max()) / rowElements);
    while (rows > 0) {
        int64_t elements = 0;
        int64_t xyOne = 0;
        int64_t xyAll = 0;
        int64_t ledger = 0;
        if (TryMulNonNegative(rows, rowElements, elements) &&
            TryAlignedTensorBytes(elements, xBytes_, blockSize_, xyOne) && TryMulNonNegative(xyOne, 4, xyAll) &&
            TryAddNonNegative(unavailable, xyAll, ledger) && ledger <= ubSize_) {
            return true;
        }
        --rows;
    }
    return false;
}

bool BNInferenceTiling::TrySelectPackedChannelFirst()
{
    int64_t q = 0;
    int64_t twiceCore = 0;
    if (!TryMulNonNegative(c_, inner_, q) || q <= 0 || q > SMALL_CF_Q_LIMIT ||
        !TryMulNonNegative(coreNum_, MIN_SMALL_CF_N_PER_CORE, twiceCore) || n_ <= twiceCore) {
        return false;
    }
    const int64_t vlFp32 = vectorLength_ / FP32_BYTES;
    const int64_t cacheLen = (vlFp32 / q) * q;
    if (cacheLen <= 0) {
        return false;
    }
    int64_t fixedBytes = 0;
    if (!GetParamLedger(c_, cacheLen, true, fixedBytes)) {
        return false;
    }
    int64_t unavailable = 0;
    if (!TryAddNonNegative(PIPE_META_RESERVE, fixedBytes, unavailable) || unavailable >= ubSize_) {
        return false;
    }
    int64_t rows = 0;
    if (!TryGetPackedRows(n_, q, unavailable, rows)) {
        return false;
    }
    int64_t tiles = 0;
    if (!TryCeilDivNonNegative(n_, rows, tiles)) {
        return false;
    }
    tileRows_ = rows;
    tileElements_ = q;
    paramTileLen_ = c_;
    paramCacheLen_ = cacheLen;
    innerTileCount_ = tiles;
    totalTiles_ = tiles;
    const uint64_t base = preFolded_ ? BNInferenceKey::CF_PACKED_PRE_FOLDED_BASE : BNInferenceKey::CF_PACKED_BASE;
    tilingKey_ = base + BNInferenceKey::OptionalMask(hasScale_, hasOffset_);
    return true;
}

bool BNInferenceTiling::TrySelectPackedChannelLast()
{
    if (c_ <= 0 || c_ > SMALL_CL_C_LIMIT || inner_ < SMALL_CL_MIN_B) {
        return false;
    }
    const int64_t vlFp32 = vectorLength_ / FP32_BYTES;
    const int64_t cacheLen = (vlFp32 / c_) * c_;
    if (cacheLen <= 0) {
        return false;
    }
    int64_t fixedBytes = 0;
    if (!GetParamLedger(c_, cacheLen, true, fixedBytes)) {
        return false;
    }
    int64_t unavailable = 0;
    if (!TryAddNonNegative(PIPE_META_RESERVE, fixedBytes, unavailable) || unavailable >= ubSize_) {
        return false;
    }
    int64_t rows = 0;
    if (!TryGetPackedRows(inner_, c_, unavailable, rows)) {
        return false;
    }
    int64_t tiles = 0;
    if (!TryCeilDivNonNegative(inner_, rows, tiles)) {
        return false;
    }
    tileRows_ = rows;
    tileElements_ = c_;
    paramTileLen_ = c_;
    paramCacheLen_ = cacheLen;
    innerTileCount_ = tiles;
    totalTiles_ = tiles;
    const uint64_t base = preFolded_ ? BNInferenceKey::CL_PACKED_PRE_FOLDED_BASE : BNInferenceKey::CL_PACKED_BASE;
    tilingKey_ = base + BNInferenceKey::OptionalMask(hasScale_, hasOffset_);
    return true;
}

bool BNInferenceTiling::SelectGenericChannelFirst()
{
    int64_t fixedBytes = 0;
    if (!GetParamLedger(1, 1, false, fixedBytes)) {
        return false;
    }
    int64_t unavailable = 0;
    if (!TryAddNonNegative(PIPE_META_RESERVE, fixedBytes, unavailable) || unavailable >= ubSize_) {
        return false;
    }
    int64_t perElement = 0;
    if (!TryMulNonNegative(xBytes_, 4, perElement) || perElement <= 0) {
        return false;
    }
    int64_t tile = std::min(inner_, (ubSize_ - unavailable) / perElement);
    tile = std::min(tile, static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) / xBytes_);
    tile = std::min(tile, static_cast<int64_t>(std::numeric_limits<uint16_t>::max()));
    while (tile > 0) {
        int64_t xyOne = 0;
        int64_t xyAll = 0;
        int64_t ledger = 0;
        if (TryAlignedTensorBytes(tile, xBytes_, blockSize_, xyOne) && TryMulNonNegative(xyOne, 4, xyAll) &&
            TryAddNonNegative(unavailable, xyAll, ledger) && ledger <= ubSize_) {
            break;
        }
        --tile;
    }
    if (tile <= 0) {
        return false;
    }
    int64_t innerTiles = 0;
    int64_t nc = 0;
    int64_t total = 0;
    if (!TryCeilDivNonNegative(inner_, tile, innerTiles) || !TryMulNonNegative(n_, c_, nc) ||
        !TryMulNonNegative(nc, innerTiles, total)) {
        return false;
    }
    tileElements_ = tile;
    tileRows_ = 1;
    paramTileLen_ = 1;
    paramCacheLen_ = 1;
    innerTileCount_ = innerTiles;
    totalTiles_ = total;
    const uint64_t base = preFolded_ ? BNInferenceKey::CF_GENERIC_PRE_FOLDED_BASE : BNInferenceKey::CF_GENERIC_BASE;
    tilingKey_ = base + BNInferenceKey::OptionalMask(hasScale_, hasOffset_);
    return true;
}

bool BNInferenceTiling::TryGetGenericChannelLastRows(int64_t cTile, int64_t& rows) const
{
    int64_t fixedBytes = 0;
    int64_t unavailable = 0;
    int64_t perRowElements = 0;
    int64_t perRowBytes = 0;
    rows = 0;
    if (!GetParamLedger(cTile, cTile, false, fixedBytes) ||
        !TryAddNonNegative(PIPE_META_RESERVE, fixedBytes, unavailable) || unavailable >= ubSize_ ||
        !TryMulNonNegative(cTile, xBytes_, perRowElements) || !TryMulNonNegative(perRowElements, 4, perRowBytes) ||
        perRowBytes <= 0 || perRowElements > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    rows = std::min(inner_, (ubSize_ - unavailable) / perRowBytes);
    rows = std::min(rows, static_cast<int64_t>(std::numeric_limits<uint16_t>::max()));
    while (rows > 0) {
        int64_t elements = 0;
        int64_t xyOne = 0;
        int64_t xyAll = 0;
        int64_t ledger = 0;
        if (TryMulNonNegative(rows, cTile, elements) && TryAlignedTensorBytes(elements, xBytes_, blockSize_, xyOne) &&
            TryMulNonNegative(xyOne, 4, xyAll) && TryAddNonNegative(unavailable, xyAll, ledger) && ledger <= ubSize_) {
            return true;
        }
        --rows;
    }
    return false;
}

bool BNInferenceTiling::SelectGenericChannelLast()
{
    const int64_t vlFp32 = vectorLength_ / FP32_BYTES;
    int64_t maxCTile = 0;
    if (!TryMulNonNegative(vlFp32, MAX_GENERIC_C_VECTORS, maxCTile)) {
        return false;
    }
    int64_t cTile = std::min(c_, maxCTile);
    if (cTile > vlFp32) {
        cTile = (cTile / vlFp32) * vlFp32;
    }
    while (cTile > 0) {
        int64_t rows = 0;
        if (TryGetGenericChannelLastRows(cTile, rows)) {
            int64_t cBlocks = 0;
            int64_t bBlocks = 0;
            int64_t total = 0;
            if (!TryCeilDivNonNegative(c_, cTile, cBlocks) || !TryCeilDivNonNegative(inner_, rows, bBlocks) ||
                !TryMulNonNegative(cBlocks, bBlocks, total)) {
                return false;
            }
            tileElements_ = cTile;
            tileRows_ = rows;
            paramTileLen_ = cTile;
            paramCacheLen_ = cTile;
            innerTileCount_ = bBlocks;
            totalTiles_ = total;
            const uint64_t base = preFolded_ ? BNInferenceKey::CL_GENERIC_PRE_FOLDED_BASE :
                                               BNInferenceKey::CL_GENERIC_BASE;
            tilingKey_ = base + BNInferenceKey::OptionalMask(hasScale_, hasOffset_);
            return true;
        }
        if (cTile <= vlFp32) {
            cTile = 0;
        } else {
            cTile -= vlFp32;
        }
    }
    return false;
}

ge::graphStatus BNInferenceTiling::SelectTiling()
{
    if (empty_) {
        tilingKey_ = BNInferenceKey::EMPTY;
        totalTiles_ = 0;
        usedCoreNum_ = 1;
        return ge::GRAPH_SUCCESS;
    }
    const bool selected = channelLast_ ? (TrySelectPackedChannelLast() || SelectGenericChannelLast()) :
                                         (TrySelectPackedChannelFirst() || SelectGenericChannelFirst());
    OP_CHECK_IF(!selected || totalTiles_ <= 0, OP_LOGE(context_, "BNInference: no UB-safe tiling can be selected"),
                return ge::GRAPH_FAILED);
    usedCoreNum_ = std::min(coreNum_, totalTiles_);
    OP_CHECK_IF(usedCoreNum_ <= 0 || static_cast<uint64_t>(usedCoreNum_) > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_, "BNInference: invalid used core count %" PRId64, usedCoreNum_),
                return ge::GRAPH_FAILED);
    baseTilesPerCore_ = totalTiles_ / usedCoreNum_;
    extraCoreCount_ = totalTiles_ % usedCoreNum_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferenceTiling::FillTilingData()
{
    auto raw = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, raw);
    OP_CHECK_IF(raw->GetData() == nullptr || raw->GetCapacity() < sizeof(BNInferenceTilingData),
                OP_LOGE(context_, "BNInference: raw tiling buffer capacity %zu is smaller than %zu", raw->GetCapacity(),
                        sizeof(BNInferenceTilingData)),
                return ge::GRAPH_FAILED);
    auto* data = reinterpret_cast<BNInferenceTilingData*>(raw->GetData());
    data->baseTilesPerCore = baseTilesPerCore_;
    data->extraCoreCount = extraCoreCount_;
    data->n = n_;
    data->c = c_;
    data->inner = inner_;
    data->tileElements = tileElements_;
    data->tileRows = tileRows_;
    data->paramTileLen = paramTileLen_;
    data->paramCacheLen = paramCacheLen_;
    data->innerTileCount = innerTileCount_;
    data->epsilon = epsilon_;
    data->reserved = 0;
    raw->SetDataSize(sizeof(BNInferenceTilingData));
    OP_CHECK_IF(context_->SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "BNInference: failed to set tiling key %" PRIu64, tilingKey_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum_)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "BNInference: failed to set block dim %" PRId64, usedCoreNum_),
                return ge::GRAPH_FAILED);
    size_t* workspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
    workspace[0] = 0;
    OP_LOGI(context_,
            "BNInference tiling key=%" PRIu64 " tiles=%" PRId64 " cores=%" PRId64 " base=%" PRId64 " extra=%" PRId64
            " N=%" PRId64 " C=%" PRId64 " inner=%" PRId64 " tileElements=%" PRId64 " tileRows=%" PRId64
            " param=%" PRId64 " cache=%" PRId64 " epsilon=%f",
            tilingKey_, totalTiles_, usedCoreNum_, baseTilesPerCore_, extraCoreCount_, n_, c_, inner_, tileElements_,
            tileRows_, paramTileLen_, paramCacheLen_, epsilon_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BNInferenceTiling::DoTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS || ValidateAndReadInputs() != ge::GRAPH_SUCCESS ||
        SelectTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return FillTilingData();
}

static ge::graphStatus TilingForBNInference(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("BNInference", "Tiling context is null"), return ge::GRAPH_FAILED);
    BNInferenceTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForBNInference(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("BNInference", "Tiling parse context is null"), return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<BNInferenceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context, "BNInference: UB size exceeds int64"), return ge::GRAPH_FAILED);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    compileInfo->vectorLength = static_cast<int64_t>(Ops::Base::GetVRegSize(context));
    compileInfo->blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GenSimplifiedKeyForBNInference(gert::TilingContext* context, ge::char_t* simplifiedKey)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("BNInference", "Context is null"), return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context, simplifiedKey);
    const auto* xDesc = context->GetInputDesc(INPUT_X);
    const auto* meanDesc = context->GetInputDesc(INPUT_MEAN);
    const auto* varianceDesc = context->GetInputDesc(INPUT_VARIANCE);
    const auto* momentumDesc = context->GetInputDesc(INPUT_MOMENTUM);
    const auto* scaleDesc = context->GetOptionalInputDesc(INPUT_SCALE);
    const auto* offsetDesc = context->GetOptionalInputDesc(INPUT_OFFSET);
    const auto* yDesc = context->GetOutputDesc(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, varianceDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, momentumDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    if (scaleDesc != nullptr && offsetDesc != nullptr && scaleDesc->GetDataType() != offsetDesc->GetDataType()) {
        const std::string dtypes = ge::TypeUtils::DataTypeToSerialString(scaleDesc->GetDataType()) + " and " +
                                   ge::TypeUtils::DataTypeToSerialString(offsetDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "scale and offset", dtypes.c_str(),
                                               "scale and offset must have the same dtype");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType requestedAffine = scaleDesc != nullptr ?
                                             scaleDesc->GetDataType() :
                                             (offsetDesc != nullptr ? offsetDesc->GetDataType() : ge::DT_UNDEFINED);
    ge::DataType affineDtype = ge::DT_UNDEFINED;
    for (const auto& combination : BNInferenceSupport::DTYPE_COMBINATIONS) {
        if (xDesc->GetDataType() == combination.x && meanDesc->GetDataType() == combination.statistics &&
            varianceDesc->GetDataType() == combination.statistics &&
            momentumDesc->GetDataType() == combination.momentum &&
            (requestedAffine == ge::DT_UNDEFINED || requestedAffine == combination.affine)) {
            affineDtype = combination.affine;
            break;
        }
    }
    if (affineDtype == ge::DT_UNDEFINED) {
        const std::string dtypes = ge::TypeUtils::DataTypeToSerialString(xDesc->GetDataType()) + ", " +
                                   ge::TypeUtils::DataTypeToSerialString(meanDesc->GetDataType()) + ", " +
                                   ge::TypeUtils::DataTypeToSerialString(momentumDesc->GetDataType()) + ", " +
                                   ge::TypeUtils::DataTypeToSerialString(requestedAffine);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, statistics, momentum and affine",
                                               dtypes.c_str(), "the dtype tuple is not registered for BNInference");
        return ge::GRAPH_FAILED;
    }

    const ge::Format xBinaryFormat = GetFeatureBinaryFormat(xDesc->GetStorageFormat());
    const ge::Format yBinaryFormat = GetFeatureBinaryFormat(yDesc->GetStorageFormat());
    const std::array<int32_t, 14> fields = {
        static_cast<int32_t>(xBinaryFormat),
        static_cast<int32_t>(meanDesc->GetStorageFormat()),
        static_cast<int32_t>(varianceDesc->GetStorageFormat()),
        static_cast<int32_t>(momentumDesc->GetStorageFormat()),
        static_cast<int32_t>(ge::FORMAT_ND),
        static_cast<int32_t>(ge::FORMAT_ND),
        static_cast<int32_t>(yBinaryFormat),
        static_cast<int32_t>(xDesc->GetDataType()),
        static_cast<int32_t>(meanDesc->GetDataType()),
        static_cast<int32_t>(varianceDesc->GetDataType()),
        static_cast<int32_t>(momentumDesc->GetDataType()),
        static_cast<int32_t>(affineDtype),
        static_cast<int32_t>(affineDtype),
        static_cast<int32_t>(yDesc->GetDataType()),
    };
    std::string key = "diy,";
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i != 0) {
            key.push_back('/');
        }
        key.append(std::to_string(fields[i]));
    }
    OP_CHECK_IF(key.size() + 1 > SIMPLIFIED_KEY_CAPACITY,
                OP_LOGE(context, "BNInference: simplified key length %zu exceeds capacity %zu", key.size() + 1,
                        SIMPLIFIED_KEY_CAPACITY),
                return ge::GRAPH_FAILED);
    std::copy_n(key.c_str(), key.size() + 1, simplifiedKey);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BNInference)
    .Tiling(TilingForBNInference)
    .TilingParse<BNInferenceCompileInfo>(TilingPrepareForBNInference)
    .GenSimplifiedKey(GenSimplifiedKeyForBNInference);
} // namespace optiling
