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
 * \file swiglu_group_grad_infershape.cpp
 * \brief SwigluGroupGrad InferShape implementation
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/shape_util.h"

using namespace ge;

namespace {

constexpr size_t GRAD_Y_INDEX = 0;
constexpr size_t X_INDEX = 1;
constexpr size_t WEIGHT_INDEX = 2;
constexpr size_t Y_ORIGIN_INDEX = 3;
constexpr size_t GROUP_INDEX_INDEX = 4;

constexpr size_t GRAD_X_INDEX = 0;
constexpr size_t GRAD_WEIGHT_INDEX = 1;

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

} // namespace

namespace ops {

static ge::graphStatus InferShapeForSwigluGroupGrad(gert::InferShapeContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context);

    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForSwigluGroupGrad.");

    const gert::Shape* grad_y_shape = context->GetInputShape(GRAD_Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, grad_y_shape);

    const gert::Shape* x_shape = context->GetInputShape(X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    const gert::Shape* weight_shape = context->GetOptionalInputShape(WEIGHT_INDEX);
    const gert::Shape* y_origin_shape = context->GetOptionalInputShape(Y_ORIGIN_INDEX);
    const gert::Shape* group_index_shape = context->GetOptionalInputShape(GROUP_INDEX_INDEX);

    const bool has_weight = (weight_shape != nullptr);
    const bool has_y_origin = (y_origin_shape != nullptr);
    const bool has_group_index = (group_index_shape != nullptr);

    if (has_weight != has_y_origin) {
        OP_LOGE(context->GetNodeName(),
                "Invalid optional inputs: weight and y_origin must both be present or both be absent. "
                "has_weight=%d, has_y_origin=%d.",
                static_cast<int>(has_weight), static_cast<int>(has_y_origin));
        return ge::GRAPH_FAILED;
    }

    gert::Shape* grad_x_shape = context->GetOutputShape(GRAD_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, grad_x_shape);

    if (Ops::Base::IsUnknownRank(*x_shape)) {
        Ops::Base::SetUnknownRank(*grad_x_shape);
    } else {
        *grad_x_shape = *x_shape;
    }

    if (has_group_index && !Ops::Base::IsUnknownRank(*group_index_shape)) {
        const size_t group_index_rank = group_index_shape->GetDimNum();
        if (group_index_rank != 1U) {
            OP_LOGE(context->GetNodeName(), "Invalid group_index rank: expected rank 1, but got rank %zu.",
                    group_index_rank);
            return ge::GRAPH_FAILED;
        }
        const int64_t group_index_size = group_index_shape->GetDim(0);
        if (!IsUnknownDim(group_index_size) && group_index_size < 1) {
            OP_LOGE(context->GetNodeName(), "Invalid group_index shape: expected a non-empty tensor.");
            return ge::GRAPH_FAILED;
        }
    }

    const bool grad_y_unknown_rank = Ops::Base::IsUnknownRank(*grad_y_shape);
    const bool x_unknown_rank = Ops::Base::IsUnknownRank(*x_shape);

    if (grad_y_unknown_rank || x_unknown_rank) {
        if (has_weight) {
            gert::Shape* grad_weight_shape = context->GetOutputShape(GRAD_WEIGHT_INDEX);
            if (grad_weight_shape != nullptr) {
                if (Ops::Base::IsUnknownRank(*weight_shape)) {
                    Ops::Base::SetUnknownRank(*grad_weight_shape);
                } else {
                    *grad_weight_shape = *weight_shape;
                }
            }
        }
        OP_LOGD(context->GetNodeName(), "Finish InferShapeForSwigluGroupGrad with unknown required-input rank.");
        return ge::GRAPH_SUCCESS;
    }

    const size_t grad_y_rank = grad_y_shape->GetDimNum();
    const size_t x_rank = x_shape->GetDimNum();

    if (grad_y_rank < 1U) {
        OP_LOGE(context->GetNodeName(), "Invalid grad_y rank: expected rank >= 1, but got rank %zu.", grad_y_rank);
        return ge::GRAPH_FAILED;
    }
    if (x_rank < 1U) {
        OP_LOGE(context->GetNodeName(), "Invalid x rank: expected rank >= 1, but got rank %zu.", x_rank);
        return ge::GRAPH_FAILED;
    }

    const size_t last_dim_index = grad_y_rank - 1U;
    const int64_t grad_y_last_dim = grad_y_shape->GetDim(last_dim_index);
    const int64_t x_last_dim = x_shape->GetDim(x_shape->GetDimNum() - 1U);

    if (!IsUnknownDim(grad_y_last_dim) && grad_y_last_dim <= 0) {
        OP_LOGE(context->GetNodeName(), "Invalid last dimension: grad_y.shape[-1](%lld) must be greater than 0.",
                static_cast<long long>(grad_y_last_dim));
        return ge::GRAPH_FAILED;
    }

    if (!IsCompatibleMultipleDim(x_last_dim, grad_y_last_dim, 2)) {
        OP_LOGE(context->GetNodeName(),
                "Invalid last dimension: x.shape[-1](%lld) must be equal to "
                "2 * grad_y.shape[-1](%lld).",
                static_cast<long long>(x_last_dim), static_cast<long long>(grad_y_last_dim));
        return ge::GRAPH_FAILED;
    }

    int64_t grad_y_total_rows = 1;
    bool grad_y_rows_known = true;
    if (!Ops::Base::IsUnknownRank(*grad_y_shape) && !Ops::Base::IsUnknownRank(*x_shape)) {
        for (size_t i = 0; i + 1U < grad_y_rank; ++i) {
            const int64_t dim = grad_y_shape->GetDim(i);
            if (IsUnknownDim(dim)) {
                grad_y_rows_known = false;
                break;
            }
            grad_y_total_rows *= dim;
        }
        int64_t x_total_rows = 1;
        bool x_rows_known = true;
        for (size_t i = 0; i + 1U < x_rank; ++i) {
            const int64_t dim = x_shape->GetDim(i);
            if (IsUnknownDim(dim)) {
                x_rows_known = false;
                break;
            }
            x_total_rows *= dim;
        }
        if (grad_y_rows_known && x_rows_known && grad_y_total_rows != x_total_rows) {
            OP_LOGE(context->GetNodeName(), "x outer numel(%lld) must equal grad_y outer numel(%lld).",
                    static_cast<long long>(x_total_rows), static_cast<long long>(grad_y_total_rows));
            return ge::GRAPH_FAILED;
        }
    }

    if (has_weight) {
        const bool weight_unknown_rank = Ops::Base::IsUnknownRank(*weight_shape);
        const bool y_origin_unknown_rank = Ops::Base::IsUnknownRank(*y_origin_shape);

        if (!weight_unknown_rank) {
            const int64_t weightElementNum = weight_shape->GetShapeSize();
            int64_t totalRows = 1;
            for (size_t i = 0; i + 1U < grad_y_rank; ++i) {
                totalRows *= grad_y_shape->GetDim(i);
            }
            if (weightElementNum != totalRows) {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context->GetNodeName(), "weight", std::to_string(weightElementNum).c_str(),
                    "The element num of weight must be equal to the product of grad_y leading dims.");
                return ge::GRAPH_FAILED;
            }
        }

        if (!y_origin_unknown_rank) {
            if (y_origin_shape->GetDimNum() < 1U) {
                OP_LOGE(context->GetNodeName(), "Invalid y_origin rank: expected rank >= 1.");
                return ge::GRAPH_FAILED;
            }
            const int64_t y_origin_last_dim = y_origin_shape->GetDim(y_origin_shape->GetDimNum() - 1U);
            if (!IsUnknownDim(y_origin_last_dim) && y_origin_last_dim != grad_y_last_dim) {
                OP_LOGE(context->GetNodeName(),
                        "Invalid y_origin last dim: y_origin.shape[-1](%lld) must equal grad_y.shape[-1](%lld).",
                        static_cast<long long>(y_origin_last_dim), static_cast<long long>(grad_y_last_dim));
                return ge::GRAPH_FAILED;
            }
            int64_t y_origin_total_rows = 1;
            bool y_origin_rows_known = true;
            for (size_t i = 0; i + 1U < y_origin_shape->GetDimNum(); ++i) {
                const int64_t dim = y_origin_shape->GetDim(i);
                if (IsUnknownDim(dim)) {
                    y_origin_rows_known = false;
                    break;
                }
                y_origin_total_rows *= dim;
            }
            if (y_origin_rows_known && grad_y_rows_known && y_origin_total_rows != grad_y_total_rows) {
                OP_LOGE(context->GetNodeName(), "y_origin outer numel(%lld) must equal grad_y outer numel(%lld).",
                        static_cast<long long>(y_origin_total_rows), static_cast<long long>(grad_y_total_rows));
                return ge::GRAPH_FAILED;
            }
        }

        gert::Shape* grad_weight_shape = context->GetOutputShape(GRAD_WEIGHT_INDEX);
        if (grad_weight_shape != nullptr) {
            if (weight_unknown_rank) {
                Ops::Base::SetUnknownRank(*grad_weight_shape);
            } else {
                *grad_weight_shape = *weight_shape;
            }
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForSwigluGroupGrad.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwigluGroupGrad(gert::InferDataTypeContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context);

    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForSwigluGroupGrad.");

    const ge::DataType grad_y_dtype = context->GetInputDataType(GRAD_Y_INDEX);
    const ge::DataType x_dtype = context->GetInputDataType(X_INDEX);

    if (grad_y_dtype != ge::DT_FLOAT16 && grad_y_dtype != ge::DT_FLOAT && grad_y_dtype != ge::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "Invalid grad_y dtype(%d): expected FP16, FP32 or BF16.",
                static_cast<int>(grad_y_dtype));
        return ge::GRAPH_FAILED;
    }

    if (x_dtype != grad_y_dtype) {
        OP_LOGE(context->GetNodeName(), "Dtype mismatch: x dtype(%d) must be equal to grad_y dtype(%d).",
                static_cast<int>(x_dtype), static_cast<int>(grad_y_dtype));
        return ge::GRAPH_FAILED;
    }

    const ge::DataType weight_dtype = context->GetOptionalInputDataType(WEIGHT_INDEX);
    const ge::DataType y_origin_dtype = context->GetOptionalInputDataType(Y_ORIGIN_INDEX);
    const ge::DataType group_index_dtype = context->GetOptionalInputDataType(GROUP_INDEX_INDEX);

    const bool has_weight = (weight_dtype != ge::DT_UNDEFINED);
    const bool has_y_origin = (y_origin_dtype != ge::DT_UNDEFINED);
    const bool has_group_index = (group_index_dtype != ge::DT_UNDEFINED);

    if (has_weight != has_y_origin) {
        OP_LOGE(context->GetNodeName(),
                "Invalid optional inputs: weight and y_origin must both be present "
                "or both be absent. has_weight=%d, has_y_origin=%d.",
                static_cast<int>(has_weight), static_cast<int>(has_y_origin));
        return ge::GRAPH_FAILED;
    }

    if (has_weight) {
        if (weight_dtype != ge::DT_FLOAT) {
            OP_LOGE(context->GetNodeName(), "Invalid weight dtype(%d): weight must be FP32.",
                    static_cast<int>(weight_dtype));
            return ge::GRAPH_FAILED;
        }

        if (y_origin_dtype != grad_y_dtype) {
            OP_LOGE(context->GetNodeName(),
                    "Dtype mismatch: y_origin dtype(%d) must be equal to "
                    "grad_y dtype(%d).",
                    static_cast<int>(y_origin_dtype), static_cast<int>(grad_y_dtype));
            return ge::GRAPH_FAILED;
        }
    }

    if (has_group_index && group_index_dtype != ge::DT_INT64) {
        OP_LOGE(context->GetNodeName(), "Invalid group_index dtype(%d): group_index must be INT64.",
                static_cast<int>(group_index_dtype));
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = context->SetOutputDataType(GRAD_X_INDEX, grad_y_dtype);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Failed to set grad_x output dtype, ret=%d.", static_cast<int>(ret));
        return ret;
    }

    if (has_weight) {
        ret = context->SetOutputDataType(GRAD_WEIGHT_INDEX, ge::DT_FLOAT);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(), "Failed to set grad_weight output dtype, ret=%d.", static_cast<int>(ret));
            return ret;
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForSwigluGroupGrad.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluGroupGrad)
    .InferShape(InferShapeForSwigluGroupGrad)
    .InferDataType(InferDataTypeForSwigluGroupGrad);

} // namespace ops
