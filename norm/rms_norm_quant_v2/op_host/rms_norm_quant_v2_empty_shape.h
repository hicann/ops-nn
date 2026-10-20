/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_QUANT_V2_EMPTY_SHAPE_H_
#define RMS_NORM_QUANT_V2_EMPTY_SHAPE_H_

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

namespace ops::rms_norm_quant {
inline bool HasEmptyDim(const gert::Shape& shape)
{
    for (size_t axis = 0; axis < shape.GetDimNum(); ++axis) {
        if (shape.GetDim(axis) == 0) {
            return true;
        }
    }
    return false;
}

inline const gert::Shape& NormalizeScalar(const gert::Shape& shape)
{
    static const gert::Shape scalarShape({1});
    return shape.IsScalar() ? scalarShape : shape;
}

inline bool CompatibleShape(const gert::Shape& first, const gert::Shape& second, bool suffix = false)
{
    if (Ops::Base::IsUnknownRank(first) || Ops::Base::IsUnknownRank(second)) {
        return true;
    }
    const auto& lhs = NormalizeScalar(first);
    const auto& rhs = NormalizeScalar(second);
    if (lhs.GetDimNum() < rhs.GetDimNum() || (!suffix && lhs.GetDimNum() != rhs.GetDimNum())) {
        return false;
    }
    const size_t offset = lhs.GetDimNum() - rhs.GetDimNum();
    for (size_t axis = 0; axis < rhs.GetDimNum(); ++axis) {
        const int64_t lhsDim = lhs.GetDim(offset + axis);
        const int64_t rhsDim = rhs.GetDim(axis);
        if (lhsDim >= 0 && rhsDim >= 0 && lhsDim != rhsDim) {
            return false;
        }
    }
    return true;
}

inline bool CheckEmptyInputDtypes(const gert::InferShapeContext* context)
{
    const auto* xDesc = context->GetInputDesc(0);
    const auto* gammaDesc = context->GetInputDesc(1);
    const auto* scaleDesc = context->GetInputDesc(2);
    if (xDesc == nullptr || gammaDesc == nullptr || scaleDesc == nullptr) {
        OP_LOGE(context, "Required input descriptor is null.");
        return false;
    }
    const auto xType = xDesc->GetDataType();
    const auto scaleType = scaleDesc->GetDataType();
    if (xType == ge::DT_UNDEFINED || scaleType == ge::DT_UNDEFINED) {
        // Inference may run before input dtypes are resolved.
        return true;
    }
    bool valid = (xType == ge::DT_FLOAT || xType == ge::DT_FLOAT16 || xType == ge::DT_BF16) &&
                 gammaDesc->GetDataType() == xType && (scaleType == ge::DT_FLOAT || scaleType == xType);
    ge::DataType firstZeroPointType = ge::DT_UNDEFINED;
    for (size_t index : {3U, 4U, 5U, 6U}) {
        if (context->GetOptionalInputShape(index) == nullptr) {
            continue;
        }
        const auto* desc = context->GetOptionalInputDesc(index);
        if (desc == nullptr) {
            OP_LOGE(context, "Optional input descriptor is null.");
            return false;
        }
        const auto type = desc->GetDataType();
        if (type == ge::DT_UNDEFINED) {
            continue;
        }
        if (index == 3U) {
            valid = valid && type == scaleType;
        } else if (index == 6U) {
            valid = valid && type == xType;
        } else {
            const bool allowedZeroPoint = xType == ge::DT_FLOAT ? type == ge::DT_FLOAT :
                                                                  (scaleType == ge::DT_FLOAT ?
                                                                       (type == ge::DT_FLOAT || type == ge::DT_INT32) :
                                                                       (type == scaleType || type == ge::DT_INT8));
            valid = valid && allowedZeroPoint && (firstZeroPointType == ge::DT_UNDEFINED || firstZeroPointType == type);
            firstZeroPointType = type;
        }
    }
    if (!valid) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "input dtypes",
            "x, gamma and beta must have the same floating dtype; scales1/scales2 and zero_points1/zero_points2 "
            "must satisfy the quantization dtype relationships, including when x is empty");
    }
    return valid;
}

inline bool CheckEmptyInputShapes(gert::InferShapeContext* context)
{
    // GE may skip tiling when all outputs are empty. Validate the input relationships before that happens.
    const auto* x = context->GetInputShape(0);
    const auto* gamma = context->GetInputShape(1);
    const auto* scale = context->GetInputShape(2);
    if (gamma == nullptr || scale == nullptr) {
        OP_LOGE(context, "Required gamma or scales1 shape is null.");
        return false;
    }
    const auto* scale2 = context->GetOptionalInputShape(3);
    const auto* zp1 = context->GetOptionalInputShape(4);
    const auto* zp2 = context->GetOptionalInputShape(5);
    const auto* beta = context->GetOptionalInputShape(6);
    for (const auto* shape : {x, gamma, scale, scale2, zp1, zp2, beta}) {
        if (shape != nullptr && !Ops::Base::IsUnknownRank(*shape) && shape->GetDimNum() > 8) {
            OP_LOGE(context, "Input rank must not exceed 8.");
            return false;
        }
    }
    bool scalarScale = true;
    for (size_t axis = 0; axis < scale->GetDimNum(); ++axis) {
        scalarScale = scalarScale && (scale->GetDim(axis) == 1 || scale->GetDim(axis) < 0);
    }
    if (!CompatibleShape(*x, *gamma, true) || (!scalarScale && !CompatibleShape(*x, *scale, true)) ||
        (scale2 != nullptr && !CompatibleShape(*scale, *scale2)) ||
        (zp1 != nullptr && !CompatibleShape(*scale, *zp1)) ||
        (zp2 != nullptr && (scale2 == nullptr || !CompatibleShape(*scale2, *zp2))) ||
        (beta != nullptr && !CompatibleShape(*gamma, *beta))) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "input shapes",
            "gamma must match the trailing axes of x; scales and optional parameters must satisfy their shape "
            "relationships, including when x is empty");
        return false;
    }
    // An all-empty GE graph can skip tiling, but INT4 still requires an even last axis.
    constexpr size_t DST_TYPE_ATTR_INDEX = 2;
    const auto* attrs = context->GetAttrs();
    const auto* dstType = attrs != nullptr && attrs->GetAttrNum() > DST_TYPE_ATTR_INDEX ?
                              attrs->GetInt(DST_TYPE_ATTR_INDEX) :
                              nullptr;
    if (dstType != nullptr && *dstType == static_cast<int64_t>(ge::DT_INT4) && x->GetDimNum() > 0) {
        const int64_t lastDim = x->GetDim(x->GetDimNum() - 1);
        if (lastDim >= 0 && lastDim % 2 != 0) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                context->GetNodeName(), "x shape",
                "The last dimension of x must be even when dst_type is INT4, including when x is empty");
            return false;
        }
    }
    return CheckEmptyInputDtypes(context);
}
} // namespace ops::rms_norm_quant
#endif
