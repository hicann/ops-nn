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
 * \file swiglu_group_quant_grad_infershape.cpp
 * \brief SwiGLU Group Dynamic Quant Backward shape inference
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
constexpr size_t INPUT_GRAD_Y = 0;
constexpr size_t INPUT_X = 1;
constexpr size_t INPUT_WEIGHT = 2;
constexpr size_t INPUT_Y_ORIGIN = 3;
constexpr size_t INPUT_GROUP_INDEX = 4;
constexpr size_t OUTPUT_GRAD_X = 0;
constexpr size_t OUTPUT_GRAD_WEIGHT = 1;
constexpr int64_t SPLIT_NUM = 2;

inline bool IsUnknownDim(const int64_t dim) { return dim < 0; }

inline bool IsCompatibleDim(const int64_t lhs, const int64_t rhs)
{
    return IsUnknownDim(lhs) || IsUnknownDim(rhs) || lhs == rhs;
}

inline bool IsCompatibleMultipleDim(const int64_t lhs, const int64_t rhs, const int64_t multiple)
{
    if (IsUnknownDim(lhs) || IsUnknownDim(rhs)) {
        return true;
    }
    return lhs == multiple * rhs;
}

// 计算除最后一维外的前导维乘积；若含 unknown dim 则返回 false（结果不可用）
inline bool CalcOuterNumel(const gert::Shape* shape, int64_t& outerNumel)
{
    outerNumel = 1;
    const size_t rank = shape->GetDimNum();
    for (size_t i = 0; i + 1U < rank; ++i) {
        const int64_t dim = shape->GetDim(i);
        if (IsUnknownDim(dim)) {
            return false;
        }
        outerNumel *= dim;
    }
    return true;
}

// 计算全部维乘积；若含 unknown dim 则返回 false（结果不可用）
inline bool CalcTotalNumel(const gert::Shape* shape, int64_t& totalNumel)
{
    totalNumel = 1;
    const size_t rank = shape->GetDimNum();
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = shape->GetDim(i);
        if (IsUnknownDim(dim)) {
            return false;
        }
        totalNumel *= dim;
    }
    return true;
}
} // namespace

namespace ops {

static ge::graphStatus CheckGradYAndXShape(gert::InferShapeContext* context, const gert::Shape* gradYShape,
                                           const gert::Shape* xShape)
{
    const size_t gradYRank = gradYShape->GetDimNum();
    const size_t xRank = xShape->GetDimNum();

    if (gradYRank < 1U) {
        OP_LOGE(context->GetNodeName(), "Invalid grad_y rank: expected rank >= 1, but got rank %zu.", gradYRank);
        return ge::GRAPH_FAILED;
    }
    if (xRank < 1U) {
        OP_LOGE(context->GetNodeName(), "Invalid x rank: expected rank >= 1, but got rank %zu.", xRank);
        return ge::GRAPH_FAILED;
    }
    if (gradYRank != xRank) {
        OP_LOGE(context->GetNodeName(), "Rank mismatch: grad_y rank(%zu) must be equal to x rank(%zu).", gradYRank,
                xRank);
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < xRank; ++i) {
        const int64_t gradYDim = gradYShape->GetDim(i);
        const int64_t xDim = xShape->GetDim(i);
        if (i + 1U < xRank) {
            if (!IsCompatibleDim(gradYDim, xDim)) {
                OP_LOGE(context->GetNodeName(),
                        "Invalid shape: grad_y.shape[%zu](%lld) must be equal to x.shape[%zu]"
                        "(%lld) except the last dimension.",
                        i, static_cast<long long>(gradYDim), i, static_cast<long long>(xDim));
                return ge::GRAPH_FAILED;
            }
            continue;
        }
        if (!IsUnknownDim(gradYDim) && gradYDim <= 0) {
            OP_LOGE(context->GetNodeName(), "Invalid last dimension: grad_y.shape[-1](%lld) must be greater than 0.",
                    static_cast<long long>(gradYDim));
            return ge::GRAPH_FAILED;
        }
        if (!IsUnknownDim(xDim) && xDim % SPLIT_NUM != 0) {
            OP_LOGE(context->GetNodeName(), "Invalid last dimension: x.shape[-1](%lld) must be divisible by %lld.",
                    static_cast<long long>(xDim), static_cast<long long>(SPLIT_NUM));
            return ge::GRAPH_FAILED;
        }
        if (!IsCompatibleMultipleDim(xDim, gradYDim, SPLIT_NUM)) {
            OP_LOGE(context->GetNodeName(),
                    "Invalid last dimension: x.shape[-1](%lld) must be equal to %lld * "
                    "grad_y.shape[-1](%lld).",
                    static_cast<long long>(xDim), static_cast<long long>(SPLIT_NUM), static_cast<long long>(gradYDim));
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightShape(gert::InferShapeContext* context, const gert::Shape* weightShape,
                                        const gert::Shape* gradYShape, const gert::Shape* xShape)
{
    int64_t gradYOuterNumel = 1;
    int64_t xOuterNumel = 1;
    int64_t weightElementCount = 1;
    const bool gradYKnown = CalcOuterNumel(gradYShape, gradYOuterNumel);
    const bool xKnown = CalcOuterNumel(xShape, xOuterNumel);
    const bool weightKnown = CalcTotalNumel(weightShape, weightElementCount);

    // 含 unknown dim 时无法可靠比较，跳过（交由 tiling 阶段以具体 shape 兜底校验）
    if (!weightKnown || (!gradYKnown && !xKnown)) {
        return ge::GRAPH_SUCCESS;
    }
    if ((gradYKnown && weightElementCount == gradYOuterNumel) || (xKnown && weightElementCount == xOuterNumel)) {
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(context->GetNodeName(),
            "Invalid weight: weight element count(%lld) must be equal to the product of "
            "grad_y or x dimensions excluding the last dimension.",
            static_cast<long long>(weightElementCount));
    return ge::GRAPH_FAILED;
}

static ge::graphStatus CheckYOriginShape(gert::InferShapeContext* context, const gert::Shape* yOriginShape,
                                         const gert::Shape* gradYShape)
{
    const size_t yOriginRank = yOriginShape->GetDimNum();
    const size_t gradYRank = gradYShape->GetDimNum();
    if (yOriginRank != gradYRank) {
        OP_LOGE(context->GetNodeName(), "Rank mismatch: y_origin rank(%zu) must be equal to grad_y rank(%zu).",
                yOriginRank, gradYRank);
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < gradYRank; ++i) {
        const int64_t yOriginDim = yOriginShape->GetDim(i);
        const int64_t gradYDim = gradYShape->GetDim(i);
        if (!IsCompatibleDim(yOriginDim, gradYDim)) {
            OP_LOGE(context->GetNodeName(),
                    "Invalid shape: y_origin.shape[%zu](%lld) must be equal to grad_y.shape[%zu]"
                    "(%lld).",
                    i, static_cast<long long>(yOriginDim), i, static_cast<long long>(gradYDim));
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupIndexShape(gert::InferShapeContext* context, const gert::Shape* groupIndexShape)
{
    const size_t groupIndexRank = groupIndexShape->GetDimNum();
    if (groupIndexRank != 1U) {
        OP_LOGE(context->GetNodeName(), "Invalid group_index rank: expected rank 1, but got rank %zu.", groupIndexRank);
        return ge::GRAPH_FAILED;
    }

    const int64_t groupIndexSize = groupIndexShape->GetDim(0);
    if (!IsUnknownDim(groupIndexSize) && groupIndexSize < 1) {
        OP_LOGE(context->GetNodeName(), "Invalid group_index shape: expected a non-empty tensor, but got size %lld.",
                static_cast<long long>(groupIndexSize));
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForSwigluGroupQuantGrad(gert::InferShapeContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context);

    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForSwigluGroupQuantGrad.");

    const gert::Shape* gradYShape = context->GetInputShape(INPUT_GRAD_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gradYShape);

    const gert::Shape* xShape = context->GetInputShape(INPUT_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape* weightShape = context->GetOptionalInputShape(INPUT_WEIGHT);
    const gert::Shape* yOriginShape = context->GetOptionalInputShape(INPUT_Y_ORIGIN);
    const gert::Shape* groupIndexShape = context->GetOptionalInputShape(INPUT_GROUP_INDEX);

    const bool hasWeight = (weightShape != nullptr);
    const bool hasYOrigin = (yOriginShape != nullptr);

    if (hasWeight != hasYOrigin) {
        OP_LOGE(context->GetNodeName(),
                "Invalid optional inputs: weight and y_origin must both be present or both be "
                "absent. has_weight=%d, has_y_origin=%d.",
                static_cast<int>(hasWeight), static_cast<int>(hasYOrigin));
        return ge::GRAPH_FAILED;
    }

    gert::Shape* gradXShape = context->GetOutputShape(OUTPUT_GRAD_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gradXShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*gradXShape);
    } else {
        *gradXShape = *xShape;
    }

    if (groupIndexShape != nullptr && !Ops::Base::IsUnknownRank(*groupIndexShape)) {
        if (CheckGroupIndexShape(context, groupIndexShape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    const bool gradYUnknownRank = Ops::Base::IsUnknownRank(*gradYShape);
    const bool xUnknownRank = Ops::Base::IsUnknownRank(*xShape);
    if (gradYUnknownRank || xUnknownRank) {
        if (hasWeight) {
            gert::Shape* gradWeightShape = context->GetOutputShape(OUTPUT_GRAD_WEIGHT);
            if (gradWeightShape != nullptr) {
                if (Ops::Base::IsUnknownRank(*weightShape)) {
                    Ops::Base::SetUnknownRank(*gradWeightShape);
                } else {
                    *gradWeightShape = *weightShape;
                }
            }
        }
        OP_LOGD(context->GetNodeName(), "Finish InferShapeForSwigluGroupQuantGrad with unknown required-input rank.");
        return ge::GRAPH_SUCCESS;
    }

    if (CheckGradYAndXShape(context, gradYShape, xShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (hasWeight) {
        if (!Ops::Base::IsUnknownRank(*weightShape) &&
            CheckWeightShape(context, weightShape, gradYShape, xShape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        gert::Shape* gradWeightShape = context->GetOutputShape(OUTPUT_GRAD_WEIGHT);
        if (gradWeightShape != nullptr) {
            if (Ops::Base::IsUnknownRank(*weightShape)) {
                Ops::Base::SetUnknownRank(*gradWeightShape);
            } else {
                *gradWeightShape = *weightShape;
            }
        }
    }

    if (hasYOrigin && !Ops::Base::IsUnknownRank(*yOriginShape)) {
        if (CheckYOriginShape(context, yOriginShape, gradYShape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForSwigluGroupQuantGrad.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwigluGroupQuantGrad(gert::InferDataTypeContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context);

    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForSwigluGroupQuantGrad.");

    const ge::DataType gradYDtype = context->GetInputDataType(INPUT_GRAD_Y);
    const ge::DataType xDtype = context->GetInputDataType(INPUT_X);

    if (gradYDtype != ge::DT_FLOAT16 && gradYDtype != ge::DT_FLOAT && gradYDtype != ge::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "Invalid grad_y dtype(%d): expected FP16, FP32 or BF16.",
                static_cast<int>(gradYDtype));
        return ge::GRAPH_FAILED;
    }
    if (xDtype != gradYDtype) {
        OP_LOGE(context->GetNodeName(), "Dtype mismatch: x dtype(%d) must be equal to grad_y dtype(%d).",
                static_cast<int>(xDtype), static_cast<int>(gradYDtype));
        return ge::GRAPH_FAILED;
    }

    const ge::DataType weightDtype = context->GetOptionalInputDataType(INPUT_WEIGHT);
    const ge::DataType yOriginDtype = context->GetOptionalInputDataType(INPUT_Y_ORIGIN);
    const ge::DataType groupIndexDtype = context->GetOptionalInputDataType(INPUT_GROUP_INDEX);

    const bool hasWeight = (weightDtype != ge::DT_UNDEFINED);
    const bool hasYOrigin = (yOriginDtype != ge::DT_UNDEFINED);
    const bool hasGroupIndex = (groupIndexDtype != ge::DT_UNDEFINED);

    if (hasWeight != hasYOrigin) {
        OP_LOGE(context->GetNodeName(),
                "Invalid optional inputs: weight and y_origin must both be present or both be "
                "absent. has_weight=%d, has_y_origin=%d.",
                static_cast<int>(hasWeight), static_cast<int>(hasYOrigin));
        return ge::GRAPH_FAILED;
    }

    if (hasWeight) {
        if (weightDtype != ge::DT_FLOAT) {
            OP_LOGE(context->GetNodeName(), "Invalid weight dtype(%d): weight must be FP32.",
                    static_cast<int>(weightDtype));
            return ge::GRAPH_FAILED;
        }
        if (yOriginDtype != gradYDtype) {
            OP_LOGE(context->GetNodeName(), "Dtype mismatch: y_origin dtype(%d) must be equal to grad_y dtype(%d).",
                    static_cast<int>(yOriginDtype), static_cast<int>(gradYDtype));
            return ge::GRAPH_FAILED;
        }
    }

    if (hasGroupIndex && groupIndexDtype != ge::DT_INT64) {
        OP_LOGE(context->GetNodeName(), "Invalid group_index dtype(%d): group_index must be INT64.",
                static_cast<int>(groupIndexDtype));
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = context->SetOutputDataType(OUTPUT_GRAD_X, gradYDtype);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Failed to set grad_x output dtype, ret=%d.", static_cast<int>(ret));
        return ret;
    }

    if (hasWeight) {
        ret = context->SetOutputDataType(OUTPUT_GRAD_WEIGHT, ge::DT_FLOAT);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(), "Failed to set grad_weight output dtype, ret=%d.", static_cast<int>(ret));
            return ret;
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForSwigluGroupQuantGrad.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluGroupQuantGrad)
    .InferShape(InferShapeForSwigluGroupQuantGrad)
    .InferDataType(InferDataTypeForSwigluGroupQuantGrad);

} // namespace ops
