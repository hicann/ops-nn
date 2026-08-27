/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_to_dense_infershape.cpp
 * \brief
 */

#include <vector>

#include "error_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "util/const_util.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
namespace {
constexpr size_t kInputIndex0 = 0U;
constexpr size_t kInputIndex1 = 1U;
constexpr size_t kInputIndex2 = 2U;
constexpr size_t kOutputIndex0 = 0U;
constexpr int64_t kRank1 = 1;
constexpr int64_t kUnknownDim = -1;
} // namespace

static ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* outputShape)
{
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outputShape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        outputShape->SetDim(i, kUnknownDim);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForSparseToDense(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForSparseToDense");
    const gert::Shape* outputShapeShape = context->GetInputShape(kInputIndex1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outputShapeShape);
    std::vector<int64_t> outputShapeDims = {};
    size_t outputShapeDimNum = outputShapeShape->GetDimNum();
    OP_LOGE_IF(outputShapeDimNum != kRank1, ge::GRAPH_FAILED, context->GetNodeName(), "output_shape must be rank [%ld]",
               kRank1);
    for (size_t i = 0U; i < outputShapeDimNum; ++i) {
        outputShapeDims.emplace_back(outputShapeShape->GetDim(i));
    }
    gert::Shape* yShape = context->GetOutputShape(kOutputIndex0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    if (IsUnknownRank(*outputShapeShape)) {
        SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }
    gert::Shape outputShape;
    bool canGetOutputShape = Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context, kInputIndex1, outputShape);
    if (!canGetOutputShape) {
        const int64_t outputRank = outputShapeDims[kInputIndex0];
        if (outputRank == ge::UNKNOWN_DIM) {
            SetUnknownRank(*yShape);
            return ge::GRAPH_SUCCESS;
        }
        OP_LOGE_IF(outputRank < 0, ge::GRAPH_FAILED, context->GetNodeName(),
                   "output rank must be non-negative, but got [%ld]", outputRank);
        return SetAllUnknownDim(outputRank, yShape);
    }
    *yShape = outputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseToDense).InputsDataDependency({kInputIndex1}).InferShape(InferShapeForSparseToDense);
} // namespace ops
