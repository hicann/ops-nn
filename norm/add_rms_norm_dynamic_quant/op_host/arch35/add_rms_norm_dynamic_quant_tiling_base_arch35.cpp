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
 * \file add_rms_norm_dynamic_quant_tiling_base_arch35.cpp
 * \brief Common validation and utility methods for RegBase tiling classes
 */

#include "add_rms_norm_dynamic_quant_tiling_arch35.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

#include <set>

namespace optiling {
using namespace NormCheck;

ge::graphStatus AddRmsNormDynamicQuantRegbaseTilingBase::CheckDtypeVaild(ge::DataType& srcDtype,
                                                                         std::vector<ge::DataType>& supportDtypeList,
                                                                         string srcName)
{
    for (const auto& supportedDtype : supportDtypeList) {
        if (supportedDtype == srcDtype) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE(nodeName_.c_str(), "Dtype check invalid, %s dtype is %s, not in supportDtypeList.", srcName.c_str(),
            Ops::Base::ToString(srcDtype).c_str());
    return ge::GRAPH_FAILED;
}

bool AddRmsNormDynamicQuantRegbaseTilingBase::CheckShapeNull()
{
    OP_LOGD(nodeName_.c_str(), "Enter CheckShapeNull.");

    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* y1Shape = context_->GetOutputShape(Y1_INDEX);
    const gert::StorageShape* y2Shape = context_->GetOutputShape(Y2_INDEX);

    const gert::StorageShape* y3Shape = isV2_ ? context_->GetOutputShape(Y3_INDEX) : nullptr;
    const gert::StorageShape* y4Shape = isV2_ ? context_->GetOutputShape(Y4_INDEX) : nullptr;
    const gert::StorageShape* xShape = isV2_ ? context_->GetOutputShape(X_INDEX_V2) : context_->GetOutputShape(X_INDEX);
    const gert::StorageShape* scale1Shape = isV2_ ? context_->GetOutputShape(SCALE1_INDEX_V2) :
                                                    context_->GetOutputShape(SCALE1_INDEX);
    const gert::StorageShape* scale2Shape = isV2_ ? context_->GetOutputShape(SCALE2_INDEX_V2) :
                                                    context_->GetOutputShape(SCALE2_INDEX);

    OP_CHECK_IF((nullptr == x1Shape) || (nullptr == x2Shape) || (nullptr == gammaShape) || (nullptr == xShape) ||
                    (nullptr == y1Shape) || (nullptr == y2Shape) || (nullptr == scale1Shape) ||
                    (nullptr == scale2Shape),
                OP_LOGE(nodeName_.c_str(), "CheckShapeNull return false"), return false);

    OP_CHECK_IF(isV2_ && ((nullptr == y3Shape) || (nullptr == y4Shape)),
                OP_LOGE(nodeName_.c_str(), "CheckY3Y4ShapeNull return false"), return false);

    if (hasSmoothScale1_) {
        const gert::StorageShape* smoothScale1Shape = context_->GetOptionalInputShape(SMOOTH_SCALE1_INDEX);
        OP_CHECK_IF(nullptr == smoothScale1Shape,
                    OP_LOGE(nodeName_.c_str(), "smoothScale1 is null but hasSmoothScale1=true."), return false);
    }
    if (hasSmoothScale2_) {
        const gert::StorageShape* smoothScale2Shape = context_->GetOptionalInputShape(SMOOTH_SCALE2_INDEX);
        OP_CHECK_IF(nullptr == smoothScale2Shape,
                    OP_LOGE(nodeName_.c_str(), "smoothScale2 is null but hasSmoothScale2=true."), return false);
    }
    if (hasBeta_) {
        const gert::StorageShape* betaShape = context_->GetOptionalInputShape(BETA_INDEX);
        OP_CHECK_IF(nullptr == betaShape, OP_LOGE(nodeName_.c_str(), "beta is null but hasBeta=true."), return false);
    }
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTilingBase::ParseOutputFlags()
{
    OP_LOGD(nodeName_.c_str(), "Enter ParseOutputFlags.");
    std::string opType(context_->GetNodeType());
    isV2_ = opType == "AddRmsNormDynamicQuantV2";
    const gert::StorageShape* smoothScale1Shape = context_->GetOptionalInputShape(SMOOTH_SCALE1_INDEX);
    const gert::StorageShape* smoothScale2Shape = context_->GetOptionalInputShape(SMOOTH_SCALE2_INDEX);
    const gert::StorageShape* betaShape = context_->GetOptionalInputShape(BETA_INDEX);
    if (smoothScale1Shape != nullptr) {
        hasSmoothScale1_ = true;
    }
    if (smoothScale2Shape != nullptr) {
        hasSmoothScale2_ = true;
    }
    if (betaShape != nullptr) {
        hasBeta_ = true;
    }

    // Parse output_mask attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const gert::ContinuousVector* outputMaskAttr = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_MASK_ATTR_IDX);
    if (outputMaskAttr != nullptr && outputMaskAttr->GetSize() != OUTPUT_MASK_NULLPTR_LEN) {
        OP_CHECK_IF((!isV2_ && outputMaskAttr->GetSize() != OUTPUT_MASK_LEN_V1) ||
                        (isV2_ && outputMaskAttr->GetSize() != OUTPUT_MASK_LEN_V2),
                    OP_LOGE(nodeName_.c_str(), "outputMask size invalid, expected %u or %u, got %zu.",
                            static_cast<unsigned int>(OUTPUT_MASK_LEN_V1),
                            static_cast<unsigned int>(OUTPUT_MASK_LEN_V2), outputMaskAttr->GetSize()),
                    return false);
        const bool* mask = static_cast<const bool*>(outputMaskAttr->GetData());
        outQuant1Flag_ = mask[0] ? 1 : 0;
        outQuant2Flag_ = mask[1] ? 1 : 0;
        hasY3_ = isV2_ && mask[2];
        hasY4_ = isV2_ && mask[3];
    } else {
        outQuant1Flag_ = 1;
        outQuant2Flag_ = (hasSmoothScale1_ && hasSmoothScale2_) ? 1 : 0;
        hasY3_ = isV2_;
        hasY4_ = isV2_;
    }
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTilingBase::CheckInputAttr()
{
    OP_LOGD(nodeName_.c_str(), "Enter CheckInputAttr.");

    // Validate output_mask constraints (flags already parsed by ParseOutputFlags)
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const gert::ContinuousVector* outputMaskAttr = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_MASK_ATTR_IDX);
    if (outputMaskAttr != nullptr && outputMaskAttr->GetSize() != OUTPUT_MASK_NULLPTR_LEN) {
        OP_CHECK_IF(hasSmoothScale1_ && outQuant1Flag_ == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName_.c_str(), "output_mask[0]", "false",
                                                          "should be true when smoothScale1 exists"),
                    return false);
        OP_CHECK_IF(hasSmoothScale2_ && outQuant2Flag_ == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName_.c_str(), "output_mask[1]", "false",
                                                          "should be true when smoothScale2 exists"),
                    return false);
        OP_CHECK_IF(outQuant1Flag_ == 0 && outQuant2Flag_ == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName_.c_str(), "output_mask", "[false, false]",
                                                          "outputMask[0] and outputMask[1] cannot both be false"),
                    return false);
    } else {
        OP_CHECK_IF(
            !hasSmoothScale1_ && hasSmoothScale2_,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName_.c_str(), "smoothScale2", "available without smoothScale1",
                                                  "smoothScale2 requires smoothScale1 when outputMask is unavailable"),
            return false);
    }

    // Validate dst_type attr: must match the active y dtype
    ge::DataType yDtype = outQuant1Flag_ == 1 ? context_->GetOutputDesc(Y1_INDEX)->GetDataType() :
                                                context_->GetOutputDesc(Y2_INDEX)->GetDataType();
    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(DST_TYPE_ATTR_INDEX);
    if (dstTypePtr != nullptr) {
        int64_t dstType = *dstTypePtr;
        OP_CHECK_IF(static_cast<ge::DataType>(dstType) != yDtype,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName_.c_str(), "dst_type", std::to_string(dstType).c_str(),
                        "2(int8), 29(int4), 34(hifloat8), 35(float8_e5m2), 36(float8_e4m3fn)"),
                    return false);
    }
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTilingBase::CheckInputShapeDim()
{
    OP_LOGD(nodeName_.c_str(), "Enter CheckInputShapeDim.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (x1DimNum < 2) || (x1DimNum > MAX_DIM_CNT) || (x2DimNum < 2) || (x2DimNum > MAX_DIM_CNT),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName_.c_str(), "x1 and x2", (std::to_string(x1DimNum) + " and " + std::to_string(x2DimNum)).c_str(),
            "The shape dims of x1 and x2 should be in range [2, 8]"),
        return false);

    // gamma, smoothScale and beta must be 1D
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(gammaDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_.c_str(), "gamma", "1", "gamma must be 1D"),
                return false);
    OP_CHECK_IF(!CheckDimBiggerZero(gammaShape, 1, nodeName_, "gamma"), , return false);

    if (hasSmoothScale1_) {
        const gert::StorageShape* s1Shape = context_->GetOptionalInputShape(SMOOTH_SCALE1_INDEX);
        size_t s1DimNum = s1Shape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            s1DimNum != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_.c_str(), "smoothScale1", "1", "smoothScale1 must be 1D"),
            return false);
        OP_CHECK_IF(!CheckDimBiggerZero(s1Shape, 1, nodeName_, "smoothScale1"), , return false);
    }
    if (hasSmoothScale2_) {
        const gert::StorageShape* s2Shape = context_->GetOptionalInputShape(SMOOTH_SCALE2_INDEX);
        size_t s2DimNum = s2Shape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(
            s2DimNum != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_.c_str(), "smoothScale2", "1", "smoothScale2 must be 1D"),
            return false);
        OP_CHECK_IF(!CheckDimBiggerZero(s2Shape, 1, nodeName_, "smoothScale2"), , return false);
    }
    if (hasBeta_) {
        const gert::StorageShape* betaShape = context_->GetOptionalInputShape(BETA_INDEX);
        size_t betaDimNum = betaShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(betaDimNum != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName_.c_str(), "beta", "1", "beta must be 1D"),
                    return false);
        OP_CHECK_IF(!CheckDimBiggerZero(betaShape, 1, nodeName_, "beta"), , return false);
    }
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTilingBase::CheckInputShapeValue()
{
    OP_LOGD(nodeName_.c_str(), "Enter CheckInputShapeValue.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* smoothScale1Shape = context_->GetOptionalInputShape(SMOOTH_SCALE1_INDEX);
    const gert::StorageShape* smoothScale2Shape = context_->GetOptionalInputShape(SMOOTH_SCALE2_INDEX);
    const gert::StorageShape* betaShape = context_->GetOptionalInputShape(BETA_INDEX);

    OP_CHECK_IF(!NormCheck::CheckShapeSame(x1Shape, x2Shape, nodeName_, "x1", "x2"),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    nodeName_.c_str(), "x2", Ops::Base::ToString(x2Shape->GetStorageShape()).c_str(), "same as x1"),
                return false);

    OP_CHECK_IF(!NormCheck::CheckShapeBC(x1Shape, gammaShape, nodeName_, "x1", "gamma", true),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_.c_str(), "gamma",
                                                      Ops::Base::ToString(gammaShape->GetStorageShape()).c_str(),
                                                      "should match last dim of x1"),
                return false);

    if (hasSmoothScale1_) {
        OP_CHECK_IF(!NormCheck::CheckShapeSame(gammaShape, smoothScale1Shape, nodeName_, "gamma", "smoothScale1"),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_.c_str(), "smoothScale1",
                        Ops::Base::ToString(smoothScale1Shape->GetStorageShape()).c_str(), "same as gamma"),
                    return false);
    }
    if (hasSmoothScale2_) {
        OP_CHECK_IF(!NormCheck::CheckShapeSame(gammaShape, smoothScale2Shape, nodeName_, "gamma", "smoothScale2"),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_.c_str(), "smoothScale2",
                        Ops::Base::ToString(smoothScale2Shape->GetStorageShape()).c_str(), "same as gamma"),
                    return false);
    }
    if (hasBeta_) {
        OP_CHECK_IF(
            !NormCheck::CheckShapeSame(gammaShape, betaShape, nodeName_, "gamma", "beta"),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName_.c_str(), "beta", Ops::Base::ToString(betaShape->GetStorageShape()).c_str(), "same as gamma"),
            return false);
    }
    // INT4 requires the last dim of x1 to be even (可被2整除)
    auto attrs = context_->GetAttrs();
    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(DST_TYPE_ATTR_INDEX);
    if (dstTypePtr != nullptr && static_cast<ge::DataType>(*dstTypePtr) == ge::DataType::DT_INT4) {
        int64_t lastDim = x1Shape->GetStorageShape().GetDim(x1Shape->GetStorageShape().GetDimNum() - 1);
        OP_CHECK_IF(lastDim % INT4_PACK_RATIO != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName_.c_str(), "x1 last dim", std::to_string(lastDim).c_str(),
                        "when dst_type is int4, the last dimension of x1 must be even (divisible by 2)"),
                    return false);
    }
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTilingBase::CheckOutputShapeValue()
{
    OP_LOGD(nodeName_.c_str(), "Enter CheckOutputShapeValue.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* y1Shape = context_->GetOutputShape(Y1_INDEX);
    const gert::StorageShape* y2Shape = context_->GetOutputShape(Y2_INDEX);
    const gert::StorageShape* y3Shape = isV2_ ? context_->GetOutputShape(Y3_INDEX) : nullptr;
    const gert::StorageShape* y4Shape = isV2_ ? context_->GetOutputShape(Y4_INDEX) : nullptr;
    const gert::StorageShape* xShape = isV2_ ? context_->GetOutputShape(X_INDEX_V2) : context_->GetOutputShape(X_INDEX);
    const gert::StorageShape* scale1Shape = isV2_ ? context_->GetOutputShape(SCALE1_INDEX_V2) :
                                                    context_->GetOutputShape(SCALE1_INDEX);
    const gert::StorageShape* scale2Shape = isV2_ ? context_->GetOutputShape(SCALE2_INDEX_V2) :
                                                    context_->GetOutputShape(SCALE2_INDEX);
    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    std::string scaleReasonStr = "x1 dimNum=" + std::to_string(x1DimNum) + "(without last dim)";
    std::string scalePrefixReasonStr = "same as x1 prefix dims";

    // output x must equal x1 (always output)
    OP_CHECK_IF(!NormCheck::CheckShapeSame(x1Shape, xShape, nodeName_, "x1", "x"),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    nodeName_.c_str(), "x", Ops::Base::ToString(xShape->GetStorageShape()).c_str(), "same as x1"),
                return false);

    // product of x1 prefix dims (without last dim), used for 1D scale1/scale2 validation
    int64_t x1PrefixProduct = 1;
    for (size_t i = 0; i < x1DimNum - 1; i++) {
        x1PrefixProduct *= x1Shape->GetStorageShape().GetDim(i);
    }

    if (outQuant1Flag_ == 1) {
        OP_CHECK_IF(!NormCheck::CheckShapeSame(x1Shape, y1Shape, nodeName_, "x1", "y1"),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_.c_str(), "y1", Ops::Base::ToString(y1Shape->GetStorageShape()).c_str(), "same as x1"),
                    return false);

        // scale1 = x1 without last dim (multi-dim prefix match, or 1D flattened prefix)
        size_t s1DimNum = scale1Shape->GetStorageShape().GetDimNum();
        if (s1DimNum == 1) {
            OP_CHECK_IF(
                scale1Shape->GetStorageShape().GetDim(0) != x1PrefixProduct,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    nodeName_.c_str(), "scale1", Ops::Base::ToString(scale1Shape->GetStorageShape()).c_str(),
                    ("1D scale1 must equal product of x1 prefix dims, expected " + std::to_string(x1PrefixProduct))
                        .c_str()),
                return false);
        } else {
            OP_CHECK_IF(s1DimNum + 1 != x1DimNum && s1DimNum != x1DimNum,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            nodeName_.c_str(), "scale1", Ops::Base::ToString(scale1Shape->GetStorageShape()).c_str(),
                            scaleReasonStr.c_str()),
                        return false);
            for (size_t i = 0; i < x1DimNum - 1; i++) {
                OP_CHECK_IF(
                    scale1Shape->GetStorageShape().GetDim(i) != x1Shape->GetStorageShape().GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_.c_str(), "scale1",
                                                          Ops::Base::ToString(scale1Shape->GetStorageShape()).c_str(),
                                                          scalePrefixReasonStr.c_str()),
                    return false);
            }
            if (s1DimNum == x1DimNum) {
                OP_CHECK_IF(
                    scale1Shape->GetStorageShape().GetDim(s1DimNum - 1) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_.c_str(), "scale1",
                                                          Ops::Base::ToString(scale1Shape->GetStorageShape()).c_str(),
                                                          "scale1 last dim does not equal 1"),
                    return false);
            }
        }
    }

    if (outQuant2Flag_ == 1) {
        OP_CHECK_IF(!NormCheck::CheckShapeSame(x1Shape, y2Shape, nodeName_, "x1", "y2"),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_.c_str(), "y2", Ops::Base::ToString(y2Shape->GetStorageShape()).c_str(), "same as x1"),
                    return false);

        // scale2 = x1 without last dim (multi-dim prefix match, or 1D flattened prefix)
        size_t s2DimNum = scale2Shape->GetStorageShape().GetDimNum();
        if (s2DimNum == 1) {
            OP_CHECK_IF(
                scale2Shape->GetStorageShape().GetDim(0) != x1PrefixProduct,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    nodeName_.c_str(), "scale2", Ops::Base::ToString(scale2Shape->GetStorageShape()).c_str(),
                    ("1D scale2 must equal product of x1 prefix dims, expected " + std::to_string(x1PrefixProduct))
                        .c_str()),
                return false);
        } else {
            OP_CHECK_IF(s2DimNum + 1 != x1DimNum && s2DimNum != x1DimNum,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            nodeName_.c_str(), "scale2", Ops::Base::ToString(scale2Shape->GetStorageShape()).c_str(),
                            scaleReasonStr.c_str()),
                        return false);
            for (size_t i = 0; i < x1DimNum - 1; i++) {
                OP_CHECK_IF(
                    scale2Shape->GetStorageShape().GetDim(i) != x1Shape->GetStorageShape().GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_.c_str(), "scale2",
                                                          Ops::Base::ToString(scale2Shape->GetStorageShape()).c_str(),
                                                          scalePrefixReasonStr.c_str()),
                    return false);
            }
            if (s2DimNum == x1DimNum) {
                OP_CHECK_IF(
                    scale2Shape->GetStorageShape().GetDim(s2DimNum - 1) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName_.c_str(), "scale2",
                                                          Ops::Base::ToString(scale2Shape->GetStorageShape()).c_str(),
                                                          "scale2 last dim does not equal 1"),
                    return false);
            }
        }
    }

    if (hasY3_) {
        OP_CHECK_IF(!NormCheck::CheckShapeSame(x1Shape, y3Shape, nodeName_, "x1", "y3"),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_.c_str(), "y3", Ops::Base::ToString(y3Shape->GetStorageShape()).c_str(), "same as x1"),
                    return false);
    }

    if (hasY4_) {
        OP_CHECK_IF(!NormCheck::CheckShapeSame(x1Shape, y4Shape, nodeName_, "x1", "y4"),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName_.c_str(), "y4", Ops::Base::ToString(y4Shape->GetStorageShape()).c_str(), "same as x1"),
                    return false);
    }
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTilingBase::CheckInputDtype()
{
    OP_LOGD(nodeName_.c_str(), "Enter CheckInputDtype.");
    std::vector<ge::DataType> supportedXGammaDtypes = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};

    ge::DataType x1Dtype = context_->GetInputTensor(X1_INDEX)->GetDataType();
    ge::DataType x2Dtype = context_->GetInputTensor(X2_INDEX)->GetDataType();
    ge::DataType gammaDtype = context_->GetInputTensor(GAMMA_INDEX)->GetDataType();

    // x1, x2 and gamma must share the same dtype, and must be float16 or bf16
    OP_CHECK_IF(
        (ge::GRAPH_SUCCESS != CheckDtypeVaild(x1Dtype, supportedXGammaDtypes, "x1")),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName_.c_str(), "x1",
                                  Ops::Base::ToString(static_cast<ge::DataType>(x1Dtype)).c_str(), "float16 or bf16"),
        return false);

    std::string x2ReasonStr = "same as x1 (" + Ops::Base::ToString(static_cast<ge::DataType>(x1Dtype)) + ")";
    OP_CHECK_IF(x1Dtype != x2Dtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName_.c_str(), "x2",
                                                      Ops::Base::ToString(static_cast<ge::DataType>(x2Dtype)).c_str(),
                                                      x2ReasonStr.c_str()),
                return false);

    std::string gammaReasonStr = "same as x1 (" + Ops::Base::ToString(static_cast<ge::DataType>(x1Dtype)) + ")";
    OP_CHECK_IF(x1Dtype != gammaDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    nodeName_.c_str(), "gamma", Ops::Base::ToString(static_cast<ge::DataType>(gammaDtype)).c_str(),
                    gammaReasonStr.c_str()),
                return false);

    if (hasSmoothScale1_) {
        ge::DataType smoothScale1Dtype = context_->GetOptionalInputTensor(SMOOTH_SCALE1_INDEX)->GetDataType();
        std::string s1ReasonStr = "same as x1 (" + Ops::Base::ToString(static_cast<ge::DataType>(x1Dtype)) + ")";
        OP_CHECK_IF(x1Dtype != smoothScale1Dtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName_.c_str(), "smoothScale1",
                        Ops::Base::ToString(static_cast<ge::DataType>(smoothScale1Dtype)).c_str(), s1ReasonStr.c_str()),
                    return false);
    }
    if (hasSmoothScale2_) {
        ge::DataType smoothScale2Dtype = context_->GetOptionalInputTensor(SMOOTH_SCALE2_INDEX)->GetDataType();
        std::string s2ReasonStr = "same as x1 (" + Ops::Base::ToString(static_cast<ge::DataType>(x1Dtype)) + ")";
        OP_CHECK_IF(x1Dtype != smoothScale2Dtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName_.c_str(), "smoothScale2",
                        Ops::Base::ToString(static_cast<ge::DataType>(smoothScale2Dtype)).c_str(), s2ReasonStr.c_str()),
                    return false);
    }
    if (hasBeta_) {
        ge::DataType betaDtype = context_->GetOptionalInputTensor(BETA_INDEX)->GetDataType();
        std::string betaReasonStr = "same as gamma (" + Ops::Base::ToString(static_cast<ge::DataType>(gammaDtype)) +
                                    ")";
        OP_CHECK_IF(gammaDtype != betaDtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName_.c_str(), "beta", Ops::Base::ToString(static_cast<ge::DataType>(betaDtype)).c_str(),
                        betaReasonStr.c_str()),
                    return false);
    }
    return true;
}

bool AddRmsNormDynamicQuantRegbaseTilingBase::CheckOutputDtype()
{
    OP_LOGD(nodeName_.c_str(), "Enter CheckOutputDtype.");
    static const std::set<ge::DataType> supportedYDtypeSet = {ge::DataType::DT_INT8, ge::DataType::DT_INT4,
                                                              ge::DataType::DT_HIFLOAT8, ge::DataType::DT_FLOAT8_E4M3FN,
                                                              ge::DataType::DT_FLOAT8_E5M2};
    const char* ySupportedStr = "int8, int4, fp8e4m3, fp8e5m2 or hifp8";

    if (outQuant1Flag_ == 1) {
        auto y1DataType = context_->GetOutputDesc(Y1_INDEX)->GetDataType();
        OP_CHECK_IF(supportedYDtypeSet.count(y1DataType) == 0,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName_.c_str(), "y1",
                                              Ops::Base::ToString(static_cast<ge::DataType>(y1DataType)).c_str(),
                                              ySupportedStr),
                    return false);

        auto scale1Dtype = isV2_ ? context_->GetOutputDesc(SCALE1_INDEX_V2)->GetDataType() :
                                   context_->GetOutputDesc(SCALE1_INDEX)->GetDataType();
        OP_CHECK_IF(
            scale1Dtype != ge::DataType::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(nodeName_.c_str(), "scale1",
                                      Ops::Base::ToString(static_cast<ge::DataType>(scale1Dtype)).c_str(), "float32"),
            return false);
    }

    if (outQuant2Flag_ == 1) {
        auto y2DataType = context_->GetOutputDesc(Y2_INDEX)->GetDataType();
        OP_CHECK_IF(supportedYDtypeSet.count(y2DataType) == 0,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName_.c_str(), "y2",
                                              Ops::Base::ToString(static_cast<ge::DataType>(y2DataType)).c_str(),
                                              ySupportedStr),
                    return false);

        auto scale2Dtype = isV2_ ? context_->GetOutputDesc(SCALE2_INDEX_V2)->GetDataType() :
                                   context_->GetOutputDesc(SCALE2_INDEX)->GetDataType();
        OP_CHECK_IF(
            scale2Dtype != ge::DataType::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(nodeName_.c_str(), "scale2",
                                      Ops::Base::ToString(static_cast<ge::DataType>(scale2Dtype)).c_str(), "float32"),
            return false);
    }

    if (outQuant1Flag_ == 1 && outQuant2Flag_ == 1) {
        auto y1DataType = context_->GetOutputDesc(Y1_INDEX)->GetDataType();
        auto y2DataType = context_->GetOutputDesc(Y2_INDEX)->GetDataType();
        std::string y2ReasonStr = "same as y1 (" + Ops::Base::ToString(static_cast<ge::DataType>(y1DataType)) + ")";
        OP_CHECK_IF(y1DataType != y2DataType,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName_.c_str(), "y2", Ops::Base::ToString(static_cast<ge::DataType>(y2DataType)).c_str(),
                        y2ReasonStr.c_str()),
                    return false);
    }

    // x dtype only needs to match x1 (x1 dtype already validated in CheckInputDtype)
    auto xDtype = isV2_ ? context_->GetOutputDesc(X_INDEX_V2)->GetDataType() :
                          context_->GetOutputDesc(X_INDEX)->GetDataType();
    auto x1Dtype = context_->GetInputTensor(X1_INDEX)->GetDataType();
    std::string xReasonStr = "same as x1 (" + Ops::Base::ToString(static_cast<ge::DataType>(x1Dtype)) + ")";
    OP_CHECK_IF(
        xDtype != x1Dtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            nodeName_.c_str(), "x", Ops::Base::ToString(static_cast<ge::DataType>(xDtype)).c_str(), xReasonStr.c_str()),
        return false);

    if (isV2_) {
        auto y3Dtype = context_->GetOutputDesc(Y3_INDEX)->GetDataType();
        auto y4Dtype = context_->GetOutputDesc(Y4_INDEX)->GetDataType();
        OP_CHECK_IF(
            y3Dtype != ge::DataType::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(nodeName_.c_str(), "y3",
                                      Ops::Base::ToString(static_cast<ge::DataType>(y3Dtype)).c_str(), "float32"),
            return false);
        std::string y4ReasonStr = "same as x1 (" + Ops::Base::ToString(static_cast<ge::DataType>(x1Dtype)) + ")";
        OP_CHECK_IF(y4Dtype != x1Dtype,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName_.c_str(), "y4", Ops::Base::ToString(static_cast<ge::DataType>(y4Dtype)).c_str(),
                        y4ReasonStr.c_str()),
                    return false);
    }
    return true;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(nodeName_.c_str(), "Enter GetShapeAttrsInfo.");
    OP_CHECK_IF(!ParseOutputFlags(), OP_LOGE(nodeName_.c_str(), "Parse output flags failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckShapeNull(), OP_LOGE(nodeName_.c_str(), "The not optional input is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputAttr(), OP_LOGE(nodeName_.c_str(), "The input attr is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputShapeDim(), OP_LOGE(nodeName_.c_str(), "The input shape dim is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputShapeValue(), OP_LOGE(nodeName_.c_str(), "The input shape relationship is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputDtype(), OP_LOGE(nodeName_.c_str(), "The input dtype is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckOutputShapeValue(), OP_LOGE(nodeName_.c_str(), "The output shape is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckOutputDtype(), OP_LOGE(nodeName_.c_str(), "The output dtype is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != SetInputParams(), OP_LOGE(nodeName_.c_str(), "Set input shape failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
