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
 * \file sparse_apply_adagrad_infershape.cpp
 * \brief shape and dtype inference for sparse_apply_adagrad
 */

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {

static constexpr int64_t IDX_VAR = 0;
static constexpr int64_t IDX_ACCUM = 1;
static constexpr int64_t IDX_LR = 2;
static constexpr int64_t IDX_GRAD = 3;
static constexpr int64_t IDX_INDICES = 4;
static constexpr int64_t UNKNOWN_DIM = -1;
static constexpr int64_t UNKNOWN_RANK = -2;

static bool IsUnknownRank(const gert::Shape* shape)
{
    return (shape->GetDimNum() == 1U) && (shape->GetDim(0) == UNKNOWN_RANK);
}

static ge::graphStatus InferShapeSparseApplyAdagrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter InferShapeSparseApplyAdagrad");

    const gert::Shape* varShape = context->GetInputShape(IDX_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    const gert::Shape* accumShape = context->GetInputShape(IDX_ACCUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, accumShape);
    const gert::Shape* lrShape = context->GetInputShape(IDX_LR);
    OP_CHECK_NULL_WITH_CONTEXT(context, lrShape);
    const gert::Shape* gradShape = context->GetInputShape(IDX_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* indicesShape = context->GetInputShape(IDX_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);

    gert::Shape* varOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varOutShape);
    gert::Shape* accumOutShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, accumOutShape);

    if (IsUnknownRank(varShape) || IsUnknownRank(accumShape) || IsUnknownRank(lrShape) || IsUnknownRank(gradShape) ||
        IsUnknownRank(indicesShape)) {
        varOutShape->SetDimNum(1U);
        varOutShape->SetDim(0, UNKNOWN_RANK);
        accumOutShape->SetDimNum(1U);
        accumOutShape->SetDim(0, UNKNOWN_RANK);
        return ge::GRAPH_SUCCESS;
    }

    auto varDimNum = varShape->GetDimNum();
    varOutShape->SetDimNum(varDimNum);
    for (size_t i = 0; i < varDimNum; ++i) {
        varOutShape->SetDim(i, varShape->GetDim(i));
    }

    auto accumDimNum = accumShape->GetDimNum();
    accumOutShape->SetDimNum(accumDimNum);
    for (size_t i = 0; i < accumDimNum; ++i) {
        accumOutShape->SetDim(i, accumShape->GetDim(i));
    }

    OP_CHECK_IF(varDimNum < 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "var", std::to_string(varDimNum), "at least 2D"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(accumDimNum != varDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "accum and var",
                    (std::to_string(accumDimNum) + "D, " + std::to_string(varDimNum) + "D").c_str(),
                    "accum dim num must equal var dim num"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < varDimNum; ++i) {
        auto varDim = varShape->GetDim(i);
        auto accumDim = accumShape->GetDim(i);
        OP_CHECK_IF(varDim != UNKNOWN_DIM && accumDim != UNKNOWN_DIM && varDim != accumDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "accum and var",
                        (Ops::Base::ToString(*accumShape) + ", " + Ops::Base::ToString(*varShape)).c_str(),
                        "accum shape must equal var shape"),
                    return ge::GRAPH_FAILED);
    }

    auto lrDimNum = lrShape->GetDimNum();
    OP_CHECK_IF(lrDimNum != 0 && !(lrDimNum == 1 && (lrShape->GetDim(0) == 1 || lrShape->GetDim(0) == UNKNOWN_DIM)),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "lr", std::to_string(lrDimNum).c_str(),
                                                         "lr must be a scalar (shape empty or [1])"),
                return ge::GRAPH_FAILED);

    auto indicesDimNum = indicesShape->GetDimNum();
    OP_CHECK_IF(indicesDimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "indices", std::to_string(indicesDimNum), "1D"),
                return ge::GRAPH_FAILED);
    int64_t indicesLen = indicesShape->GetDim(0);

    auto gradDimNum = gradShape->GetDimNum();
    OP_CHECK_IF(gradDimNum != varDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "grad and var",
                    (std::to_string(gradDimNum) + "D, " + std::to_string(varDimNum) + "D").c_str(),
                    "grad dim num must equal var dim num"),
                return ge::GRAPH_FAILED);

    auto gradDim0 = gradShape->GetDim(0);
    OP_CHECK_IF(gradDim0 != UNKNOWN_DIM && indicesLen != UNKNOWN_DIM && gradDim0 != indicesLen,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "grad", Ops::Base::ToString(*gradShape).c_str(),
                    ("grad first dim must equal indices length " + std::to_string(indicesLen)).c_str()),
                return ge::GRAPH_FAILED);

    for (size_t i = 1; i < varDimNum; ++i) {
        auto varDim = varShape->GetDim(i);
        auto gradDim = gradShape->GetDim(i);
        OP_CHECK_IF(varDim != UNKNOWN_DIM && gradDim != UNKNOWN_DIM && varDim != gradDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "grad and var",
                        (Ops::Base::ToString(*gradShape) + ", " + Ops::Base::ToString(*varShape)).c_str(),
                        "grad shape[1:] must equal var shape[1:]"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSparseApplyAdagrad(gert::InferDataTypeContext* context)
{
    auto varDtype = context->GetInputDataType(IDX_VAR);
    auto accumDtype = context->GetInputDataType(IDX_ACCUM);
    auto lrDtype = context->GetInputDataType(IDX_LR);
    auto gradDtype = context->GetInputDataType(IDX_GRAD);
    auto indicesDtype = context->GetInputDataType(IDX_INDICES);

    OP_CHECK_IF(varDtype != accumDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "var and accum",
                                                       (ge::TypeUtils::DataTypeToSerialString(varDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(accumDtype))
                                                           .c_str(),
                                                       "var dtype must equal accum dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(varDtype != lrDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "var and lr",
                                                       (ge::TypeUtils::DataTypeToSerialString(varDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(lrDtype))
                                                           .c_str(),
                                                       "var dtype must equal lr dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(varDtype != gradDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "var and grad",
                                                       (ge::TypeUtils::DataTypeToSerialString(varDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(gradDtype))
                                                           .c_str(),
                                                       "var dtype must equal grad dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(varDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "var",
                                          ge::TypeUtils::DataTypeToSerialString(varDtype).c_str(), "FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesDtype != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "indices",
                                          ge::TypeUtils::DataTypeToSerialString(indicesDtype).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    context->SetOutputDataType(0, varDtype);
    context->SetOutputDataType(1, accumDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseApplyAdagrad)
    .InferShape(InferShapeSparseApplyAdagrad)
    .InferDataType(InferDataTypeSparseApplyAdagrad);

} // namespace ops
