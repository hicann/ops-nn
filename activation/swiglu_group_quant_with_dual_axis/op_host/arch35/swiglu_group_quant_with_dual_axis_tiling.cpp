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
 * \file swiglu_group_quant_with_dual_axis_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <graph/utils/type_utils.h>
#include "util/shape_util.h"
#include "swiglu_group_quant_mx_tiling.h"
#include "swiglu_group_quant_with_dual_axis_tiling.h"
#include "swiglu_group_quant_with_dual_axis_policy.h"
#include "../../op_kernel/arch35/swiglu_group_quant_with_dual_axis_tiling_key.h"

using namespace ge;
namespace optiling {
namespace {
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MX_SCALE_PAIR = 2;
constexpr int64_t SCALE_PAIR_ELEMENTS = BLOCK_SIZE * MX_SCALE_PAIR;
constexpr int64_t GROUP_CHUNK = 256;
constexpr int64_t UB_GUARD_BYTES = 8192;
constexpr int64_t H_LIMIT = 32;
constexpr size_t X_DIM_NUM = 2;
constexpr size_t MAX_DIM_NUM = 8;
constexpr int64_t DUAL_AXIS_MODE = 1;
constexpr int64_t DST_TYPE_E5M2 = 35;
constexpr int64_t DST_TYPE_E4M3FN = 36;
constexpr float DEFAULT_CLAMP_LIMIT = -1.0f;
constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr size_t ATTR_INDEX_QUANT_MODE = 1;
constexpr size_t ATTR_INDEX_CLAMP_LIMIT = 2;
constexpr size_t ATTR_INDEX_OUTPUT_ORIGIN = 3;
constexpr size_t ATTR_INDEX_ALPHA = 4;
constexpr size_t ATTR_INDEX_BIAS = 5;
constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_WEIGHT = 1;
constexpr size_t INPUT_INDEX_GROUP_INDEX = 2;
constexpr size_t OUTPUT_INDEX_Y1 = 0;
constexpr size_t OUTPUT_INDEX_MX_SCALE1 = 1;
constexpr size_t OUTPUT_INDEX_Y2 = 2;
constexpr size_t OUTPUT_INDEX_MX_SCALE2 = 3;
constexpr size_t OUTPUT_INDEX_Y_ORIGIN = 4;

int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return x / y + (x % y != 0);
    }
    return x;
}

bool CheckedMul(int64_t lhs, int64_t rhs, int64_t& out)
{
    if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool ShapeIs(const gert::Shape& shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t i = 0;
    for (int64_t dim : dims) {
        if (shape.GetDim(i++) != dim) {
            return false;
        }
    }
    return true;
}
} // namespace

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::GetPlatformInfoCommon(gert::TilingContext* context,
                                                                          uint64_t& coreNum, uint64_t& ubSize)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context->GetCompileInfo<SwigluGroupQuantWithDualAxisCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compile info should not be null."),
                    return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize = ubSizePlatForm;
    }
    OP_CHECK_IF(
        (coreNum == 0 || ubSize == 0),
        OP_LOGE(context->GetNodeName(),
                "platform coreNum and ubSize should be greater than 0, got coreNum %lu, ubSize %lu.", coreNum, ubSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::GetPlatformInfo()
{
    if (GetPlatformInfoCommon(context_, coreNum_, ubSize_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::GetWorkspaceSize()
{
    if (context_->GetPlatformInfo() != nullptr) {
        auto platform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        workspaceSize_ = platform.GetLibApiWorkSpaceSize();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::PostTiling()
{
    SetTilingData();
    if (context_->SetBlockDim(usedCoreNums_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::GetClampLimitAttr(const gert::RuntimeAttrs* attrs)
{
    auto clampLimitAttr = attrs->GetAttrPointer<float>(ATTR_INDEX_CLAMP_LIMIT);
    clampLimit_ = clampLimitAttr == nullptr ? DEFAULT_CLAMP_LIMIT : *clampLimitAttr;
    OP_CHECK_IF((!std::isfinite(clampLimit_) || (clampLimit_ != DEFAULT_CLAMP_LIMIT && clampLimit_ <= 0.0)),
                OP_LOGE(context_->GetNodeName(), "attr clamp_limit should be %f or greater than 0, got %f.",
                        DEFAULT_CLAMP_LIMIT, clampLimit_),
                return ge::GRAPH_FAILED);
    if (clampLimit_ != DEFAULT_CLAMP_LIMIT) {
        hasClampLimit_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::GetAttr()
{
    const auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto dstTypeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DST_TYPE);
    const int64_t dstTypeValue = dstTypeAttr == nullptr ? DST_TYPE_E4M3FN : *dstTypeAttr;
    OP_CHECK_IF((dstTypeValue != DST_TYPE_E4M3FN && dstTypeValue != DST_TYPE_E5M2),
                OP_LOGE(context_->GetNodeName(),
                        "attr dst_type only supports 35(FLOAT8_E5M2) or 36(FLOAT8_E4M3FN), got %ld.", dstTypeValue),
                return ge::GRAPH_FAILED);
    dstType_ = dstTypeValue == DST_TYPE_E5M2 ? ge::DT_FLOAT8_E5M2 : ge::DT_FLOAT8_E4M3FN;

    auto quantModeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    quantMode_ = quantModeAttr == nullptr ? DUAL_AXIS_MODE : *quantModeAttr;
    OP_CHECK_IF((quantMode_ != DUAL_AXIS_MODE),
                OP_LOGE(context_->GetNodeName(), "attr quant_mode only supports 1(dual axis), got %ld.", quantMode_),
                return ge::GRAPH_FAILED);

    auto outputOriginAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_OUTPUT_ORIGIN);
    if (outputOriginAttr != nullptr && *outputOriginAttr) {
        outputOrigin_ = 1;
    }

    auto alphaAttr = attrs->GetAttrPointer<float>(ATTR_INDEX_ALPHA);
    auto biasAttr = attrs->GetAttrPointer<float>(ATTR_INDEX_BIAS);
    alpha_ = alphaAttr == nullptr ? 1.0f : *alphaAttr;
    bias_ = biasAttr == nullptr ? 0.0f : *biasAttr;
    OP_CHECK_IF((!std::isfinite(alpha_) || alpha_ <= 0.0f || !std::isfinite(bias_)),
                OP_LOGE(context_->GetNodeName(),
                        "attr alpha should be finite and greater than 0, bias should be finite, got alpha %f, bias %f.",
                        alpha_, bias_),
                return ge::GRAPH_FAILED);

    if (GetClampLimitAttr(attrs) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::CheckWeightInfo()
{
    auto weightDesc = context_->GetOptionalInputDesc(INPUT_INDEX_WEIGHT);
    if (weightDesc != nullptr) {
        weightDtype_ = weightDesc->GetDataType();
        OP_CHECK_IF(
            (weightDtype_ != ge::DT_FLOAT16 && weightDtype_ != ge::DT_BF16 && weightDtype_ != ge::DT_FLOAT),
            OP_LOGE(context_->GetNodeName(), "input weight dtype should be FLOAT16, BFLOAT16 or FLOAT32, got %s.",
                    ge::TypeUtils::DataTypeToSerialString(weightDtype_).c_str()),
            return ge::GRAPH_FAILED);
        auto weightShape = context_->GetOptionalInputShape(INPUT_INDEX_WEIGHT);
        OP_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
        const auto weightStorageShape = weightShape->GetStorageShape();
        const size_t weightDimNum = weightStorageShape.GetDimNum();
        OP_CHECK_IF((weightDimNum < 1 || weightDimNum > MAX_DIM_NUM),
                    OP_LOGE(context_->GetNodeName(), "input weight dim num should be in [1, %zu], got %zu.",
                            MAX_DIM_NUM, weightDimNum),
                    return ge::GRAPH_FAILED);
        int64_t weightElementNum = 1;
        for (size_t i = 0; i < weightDimNum; ++i) {
            const int64_t extent = weightStorageShape.GetDim(i);
            OP_CHECK_IF(!CheckedMul(weightElementNum, extent, weightElementNum),
                        OP_LOGE(context_->GetNodeName(),
                                "input weight dimensions must be positive and their product must fit int64, "
                                "got dim[%zu] %ld.",
                                i, extent),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF((weightElementNum != bs_),
                    OP_LOGE(context_->GetNodeName(),
                            "input weight element num should be equal to the product of input x dims except the last "
                            "one, got %ld, expected %ld.",
                            weightElementNum, bs_),
                    return ge::GRAPH_FAILED);
        hasWeight_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::CheckGroupIndexInfo()
{
    auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_INDEX_GROUP_INDEX);
    if (groupIndexDesc != nullptr) {
        auto groupIndexDtype = groupIndexDesc->GetDataType();
        OP_CHECK_IF((groupIndexDtype != ge::DT_INT64),
                    OP_LOGE(context_->GetNodeName(), "input group_index dtype should be INT64, got %s.",
                            ge::TypeUtils::DataTypeToSerialString(groupIndexDtype).c_str()),
                    return ge::GRAPH_FAILED);
        auto groupIndexShape = context_->GetOptionalInputShape(INPUT_INDEX_GROUP_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, groupIndexShape);
        auto groupIndexStorageShape = groupIndexShape->GetStorageShape();
        OP_CHECK_IF((groupIndexStorageShape.GetDimNum() != 1),
                    OP_LOGE(context_->GetNodeName(), "input group_index dim num should be 1, got %zu.",
                            groupIndexStorageShape.GetDimNum()),
                    return ge::GRAPH_FAILED);
        g_ = groupIndexStorageShape.GetDim(0);
        OP_CHECK_IF(
            (g_ <= 0),
            OP_LOGE(context_->GetNodeName(), "input group_index element num should be greater than 0, got %ld.", g_),
            return ge::GRAPH_FAILED);
        hasGroupIndex_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::CheckOutputInfo(ge::DataType xDtype,
                                                                    const gert::Shape& xStorageShape)
{
    const ge::DataType expectedScale = ge::DT_FLOAT8_E8M0;
    auto y1Desc = context_->GetOutputDesc(OUTPUT_INDEX_Y1);
    auto scale1Desc = context_->GetOutputDesc(OUTPUT_INDEX_MX_SCALE1);
    auto y2Desc = context_->GetOutputDesc(OUTPUT_INDEX_Y2);
    auto scale2Desc = context_->GetOutputDesc(OUTPUT_INDEX_MX_SCALE2);
    auto originDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y_ORIGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y1Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale1Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y2Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale2Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, originDesc);

    OP_CHECK_IF((y1Desc->GetDataType() != dstType_ || y2Desc->GetDataType() != dstType_),
                OP_LOGE(context_->GetNodeName(),
                        "output y1/y2 dtype should be same as dst_type %s, got y1 dtype %s, y2 dtype %s.",
                        ge::TypeUtils::DataTypeToSerialString(dstType_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(y1Desc->GetDataType()).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(y2Desc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((scale1Desc->GetDataType() != expectedScale || scale2Desc->GetDataType() != expectedScale),
                OP_LOGE(context_->GetNodeName(), "output mxscale1/mxscale2 dtype should be %s, got %s and %s.",
                        ge::TypeUtils::DataTypeToSerialString(expectedScale).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(scale1Desc->GetDataType()).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(scale2Desc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((originDesc->GetDataType() != xDtype),
                OP_LOGE(context_->GetNodeName(), "output y_origin dtype should be same as input x %s, got %s.",
                        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(originDesc->GetDataType()).c_str()),
                return ge::GRAPH_FAILED);

    const auto& y1 = context_->GetOutputShape(OUTPUT_INDEX_Y1)->GetStorageShape();
    const auto& y2 = context_->GetOutputShape(OUTPUT_INDEX_Y2)->GetStorageShape();
    const auto& s1 = context_->GetOutputShape(OUTPUT_INDEX_MX_SCALE1)->GetStorageShape();
    const auto& s2 = context_->GetOutputShape(OUTPUT_INDEX_MX_SCALE2)->GetStorageShape();
    const auto& origin = context_->GetOutputShape(OUTPUT_INDEX_Y_ORIGIN)->GetStorageShape();

    if (!hasGroupIndex_) {
        const auto rank = xStorageShape.GetDimNum();
        auto expectedY = xStorageShape;
        expectedY.SetDim(rank - 1, splitD_);
        auto expectedS1 = expectedY;
        expectedS1.SetDim(rank - 1, CeilDiv(splitD_, SCALE_PAIR_ELEMENTS));
        expectedS1.AppendDim(MX_SCALE_PAIR);
        auto expectedS2 = expectedY;
        expectedS2.SetDim(rank - 2, CeilDiv(batchRows_, SCALE_PAIR_ELEMENTS));
        expectedS2.AppendDim(MX_SCALE_PAIR);
        OP_CHECK_IF(
            (y1 != expectedY || y2 != expectedY || s1 != expectedS1 || s2 != expectedS2 ||
             (outputOrigin_ != 0 ? origin != expectedY : !ShapeIs(origin, {0}))),
            OP_LOGE(context_->GetNodeName(),
                    "non-group output shapes should preserve batch dimensions: expected y %s, scale1 %s, "
                    "scale2 %s, origin %s; got y1 %s, y2 %s, scale1 %s, scale2 %s, origin %s.",
                    Ops::Base::ToString(expectedY).c_str(), Ops::Base::ToString(expectedS1).c_str(),
                    Ops::Base::ToString(expectedS2).c_str(),
                    outputOrigin_ != 0 ? Ops::Base::ToString(expectedY).c_str() : "[0]",
                    Ops::Base::ToString(y1).c_str(), Ops::Base::ToString(y2).c_str(), Ops::Base::ToString(s1).c_str(),
                    Ops::Base::ToString(s2).c_str(), Ops::Base::ToString(origin).c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF((!ShapeIs(y1, {bs_, splitD_})),
                OP_LOGE(context_->GetNodeName(), "output y1 shape should be [%ld, %ld], got [%ld, %ld].", bs_, splitD_,
                        y1.GetDimNum() > 0 ? y1.GetDim(0) : -1, y1.GetDimNum() > 1 ? y1.GetDim(1) : -1),
                return ge::GRAPH_FAILED);
    const int64_t scale1Rows = CeilDiv(splitD_, SCALE_PAIR_ELEMENTS);
    OP_CHECK_IF((!ShapeIs(s1, {bs_, scale1Rows, MX_SCALE_PAIR})),
                OP_LOGE(context_->GetNodeName(), "output mxscale1 shape should be [%ld, %ld, %ld], got %s.", bs_,
                        scale1Rows, MX_SCALE_PAIR, Ops::Base::ToString(s1).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((!ShapeIs(y2, {bs_, splitD_})),
                OP_LOGE(context_->GetNodeName(), "output y2 shape should be [%ld, %ld], got [%ld, %ld].", bs_, splitD_,
                        y2.GetDimNum() > 0 ? y2.GetDim(0) : -1, y2.GetDimNum() > 1 ? y2.GetDim(1) : -1),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(g_ > std::numeric_limits<int64_t>::max() - bs_ / SCALE_PAIR_ELEMENTS,
                OP_LOGE(context_->GetNodeName(), "grouped mxscale2 row count exceeds int64 range."),
                return ge::GRAPH_FAILED);
    const int64_t scale2Rows = bs_ / SCALE_PAIR_ELEMENTS + g_;
    OP_CHECK_IF((!ShapeIs(s2, {scale2Rows, splitD_, MX_SCALE_PAIR})),
                OP_LOGE(context_->GetNodeName(), "output mxscale2 shape should be [%ld, %ld, %ld], got %s.", scale2Rows,
                        splitD_, MX_SCALE_PAIR, Ops::Base::ToString(s2).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((!ShapeIs(origin, outputOrigin_ != 0 ? std::initializer_list<int64_t>{bs_, splitD_} :
                                                       std::initializer_list<int64_t>{0})),
                OP_LOGE(context_->GetNodeName(),
                        "output y_origin shape should be [%ld, %ld] when output_origin is true and [0] otherwise, got "
                        "%s.",
                        bs_, splitD_, Ops::Base::ToString(origin).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::GetShapeAttrsInfoInner()
{
    auto shapeX = context_->GetInputShape(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    auto xDesc = context_->GetInputDesc(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);

    xDtype_ = xDesc->GetDataType();
    OP_CHECK_IF((xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_BF16),
                OP_LOGE(context_->GetNodeName(), "input x dtype only supports FLOAT16 or BFLOAT16, got %s.",
                        ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
                return ge::GRAPH_FAILED);

    const auto& xStorageShape = shapeX->GetStorageShape();
    const int64_t xDimNum = static_cast<int64_t>(xStorageShape.GetDimNum());
    OP_CHECK_IF((xDimNum != static_cast<int64_t>(X_DIM_NUM)),
                OP_LOGE(context_->GetNodeName(), "input x dim num should be %zu, got %ld.", X_DIM_NUM, xDimNum),
                return ge::GRAPH_FAILED);

    bs_ = 1;
    for (int64_t i = 0; i + 1 < xDimNum; ++i) {
        const int64_t extent = xStorageShape.GetDim(i);
        OP_CHECK_IF((extent <= 0),
                    OP_LOGE(context_->GetNodeName(), "input x dim[%ld] should be greater than 0, got %ld.", i, extent),
                    return ge::GRAPH_FAILED);
        int64_t next = 0;
        OP_CHECK_IF(!CheckedMul(bs_, extent, next),
                    OP_LOGE(context_->GetNodeName(),
                            "input x shape exceeds int64 range, got dim[%ld] %ld with accumulated product %ld.", i,
                            extent, bs_),
                    return ge::GRAPH_FAILED);
        bs_ = next;
    }

    batchRows_ = xStorageShape.GetDim(xDimNum - 2);
    d_ = xStorageShape.GetDim(xDimNum - 1);
    OP_CHECK_IF((d_ <= 0), OP_LOGE(context_->GetNodeName(), "input x last dim should be greater than 0, got %ld.", d_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((d_ % SCALE_PAIR_ELEMENTS != 0),
                OP_LOGE(context_->GetNodeName(), "input x last dim should be divisible by 64, got %ld.", d_),
                return ge::GRAPH_FAILED);
    splitD_ = d_ / 2;
    OP_CHECK_IF((splitD_ < H_LIMIT),
                OP_LOGE(context_->GetNodeName(), "input x last dim should be greater than or equal to %ld, got %ld.",
                        H_LIMIT * 2, d_),
                return ge::GRAPH_FAILED);

    if (CheckWeightInfo() == ge::GRAPH_FAILED || CheckGroupIndexInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((hasWeight_ && !hasGroupIndex_),
                OP_LOGE(context_->GetNodeName(),
                        "input group_index should be provided when weight is provided, got group_index absent."),
                return ge::GRAPH_FAILED);

    if (GetAttr() == ge::GRAPH_FAILED) {
        OP_LOGE(context_->GetNodeName(), "Get attr failed.");
        return ge::GRAPH_FAILED;
    }

    if (CheckOutputInfo(xDtype_, xStorageShape) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void SwigluGroupQuantWithDualAxisTiling::CalcCoreTiling()
{
    // group_index partitions route 2 only; the activation and route 1 always consume all T rows,
    // so every core participates in the global 64-row task space.
    const auto geometry = SwigluGroupQuantMxTiling::MakeGeometry(splitD_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = geometry.usedCoreNum;
    rowOfFormerBlock_ = bs_;
    rowOfTailBlock_ = bs_;
    rowFactor_ = SwigluGroupQuantMxTiling::TILE_ROWS;
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;
}

void SwigluGroupQuantWithDualAxisTiling::CalcBlockTiling()
{
    // The vector helpers intentionally run a full tile; tail lanes are zero padded and
    // only the valid columns are copied out.
    const auto geometry = SwigluGroupQuantMxTiling::MakeGeometry(splitD_, static_cast<int64_t>(coreNum_));
    dFactor_ = SwigluGroupQuantMxTiling::TILE_COLS;
    dLoop_ = geometry.dimNBlockNum;
    tailDFactor_ = geometry.dimNTail;
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::CalcOpTiling()
{
    CalcCoreTiling();
    CalcBlockTiling();

    // Conservative legacy-layout budget. The common unweighted H=384 path uses
    // 205952 bytes (single-buffered 64x384), below the 238336-byte legacy maximum.
    constexpr int64_t b16Bytes = 2;
    constexpr int64_t fp8Bytes = 1;
    constexpr int64_t doubleBuffer = 2;
    const int64_t tileElements = rowFactor_ * dFactor_;
    const int64_t inputBytes = doubleBuffer * tileElements * b16Bytes * 2;
    const int64_t activationBytes = tileElements * b16Bytes;
    const bool shareOrigin = batchRows_ == bs_ && SwigluDualAxisPolicy::ShareOrigin(hasWeight_, outputOrigin_ != 0);
    const int64_t outputDepth = (hasWeight_ || outputOrigin_ != 0) && !shareOrigin ? 1 : doubleBuffer;
    const int64_t y1Bytes = outputDepth * tileElements * fp8Bytes;
    const int64_t y2Bytes = outputDepth * tileElements * fp8Bytes;
    const int64_t scale1Bytes = outputDepth * rowFactor_ * BLOCK_SIZE;
    const int64_t scale2Bytes = outputDepth * SwigluGroupQuantMxTiling::MX_SCALE2_BUFFER_BYTES;
    const int64_t reciprocal1Bytes = rowFactor_ * BLOCK_SIZE;
    const int64_t reciprocal2Bytes = dFactor_ * 2 * b16Bytes;
    const int64_t weightBytes = hasWeight_ ? rowFactor_ * static_cast<int64_t>(sizeof(float)) : 0;
    const int64_t originBytes = outputOrigin_ != 0 && !shareOrigin ? activationBytes : 0;
    plannedUbBytes_ = inputBytes + activationBytes + y1Bytes + y2Bytes + scale1Bytes + scale2Bytes + reciprocal1Bytes +
                      reciprocal2Bytes + weightBytes + originBytes + UB_GUARD_BYTES;
    OP_CHECK_IF(plannedUbBytes_ > static_cast<int64_t>(ubSize_),
                OP_LOGE(context_->GetNodeName(), "planned UB bytes should not exceed available UB size, got %ld, %lu.",
                        plannedUbBytes_, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void SwigluGroupQuantWithDualAxisTiling::SetTilingData()
{
    uint32_t flags = 0U;
    flags |= hasWeight_ ? MX_HAS_WEIGHT : 0U;
    flags |= hasGroupIndex_ ? MX_HAS_GROUP : 0U;
    flags |= hasClampLimit_ != 0 ? MX_HAS_CLAMP : 0U;
    flags |= outputOrigin_ != 0 ? MX_OUTPUT_ORIGIN : 0U;
    if (batchRows_ == bs_) {
        flags |= SwigluDualAxisPolicy::ShareOrigin(hasWeight_, outputOrigin_ != 0) ? MX_SHARE_ORIGIN : 0U;
        flags |= SwigluDualAxisPolicy::OwnWholeGroups(splitD_, hasWeight_, outputOrigin_ != 0, hasGroupIndex_, g_,
                                                      usedCoreNums_) ?
                     MX_OWN_WHOLE_GROUPS :
                     0U;
    }

    tilingData_.set_version(2);
    tilingData_.set_quantMode(quantMode_);
    tilingData_.set_inputType(xDtype_ == ge::DT_FLOAT16 ? 0 : 1);
    tilingData_.set_weightType(weightDtype_ == ge::DT_FLOAT16 ? 0 : (weightDtype_ == ge::DT_BF16 ? 1 : 2));
    tilingData_.set_flags(flags);
    tilingData_.set_t(bs_);
    tilingData_.set_h(splitD_);
    tilingData_.set_keep(bs_);
    tilingData_.set_groupCount(g_);
    tilingData_.set_batchRows(batchRows_);
    tilingData_.set_alpha(alpha_);
    tilingData_.set_bias(bias_);
    tilingData_.set_clampLimit(static_cast<float>(clampLimit_));
    tilingData_.set_scale1RowBytes(MX_SCALE_PAIR * CeilDiv(splitD_, SCALE_PAIR_ELEMENTS));
    tilingData_.set_scale2PairRows(hasGroupIndex_ ? bs_ / SCALE_PAIR_ELEMENTS + g_ :
                                                    (bs_ / batchRows_) * CeilDiv(batchRows_, SCALE_PAIR_ELEMENTS));
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_usedCoreCount(usedCoreNums_);
    tilingData_.set_tileRows(SwigluGroupQuantMxTiling::TILE_ROWS);
    tilingData_.set_tileCols(dFactor_);
    tilingData_.set_groupChunkSize(GROUP_CHUNK);
    tilingData_.set_ubUserBytes(ubSize_);
    tilingData_.set_ubPlannedBytes(static_cast<uint64_t>(plannedUbBytes_));
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::SetTilingKey()
{
    const uint64_t mode = dLoop_ < usedCoreNums_ ? TPL_MODE_ROTATE : TPL_MODE_BLOCK;
    const uint64_t isGroup = hasGroupIndex_ ? TPL_GROUP_INDEX : TPL_NO_GROUP_INDEX;
    const uint64_t hasClamp = hasClampLimit_ != 0 ? 1U : 0U;
    const uint64_t hasAttrs = hasClamp != 0 || alpha_ != 1.0F || bias_ != 0.0F ? 1U : 0U;
    tilingKey_ = GET_TPL_TILING_KEY(mode, isGroup, hasAttrs, hasClamp);
    return context_->SetTilingKey(tilingKey_);
}

ge::graphStatus SwigluGroupQuantWithDualAxisTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (SetTilingKey() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSwigluGroupQuantWithDualAxis(gert::TilingContext* context)
{
    SwigluGroupQuantWithDualAxisTiling tiling(context);
    return tiling.DoOpTiling();
}

ge::graphStatus TilingPrepareForSwigluGroupQuantWithDualAxis(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwigluGroupQuantWithDualAxis)
    .Tiling(TilingForSwigluGroupQuantWithDualAxis)
    .TilingParse<SwigluGroupQuantWithDualAxisCompileInfo>(TilingPrepareForSwigluGroupQuantWithDualAxis);
} // namespace optiling
