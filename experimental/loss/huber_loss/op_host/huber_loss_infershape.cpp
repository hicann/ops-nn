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
 * \file huber_loss_infershape.cpp
 * \brief HuberLoss shape and dtype inference
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../op_kernel/huber_loss_tiling_data.h"

using namespace ge;

namespace ops {

static constexpr size_t IDX_INPUT = 0;
static constexpr size_t IDX_TARGET = 1;
static constexpr size_t IDX_OUT = 0;
static constexpr size_t ATTR_REDUCTION = 0; // attribute order is fixed in the OpDef: reduction 0, delta 1

static ge::graphStatus InferShapeHuberLoss(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeHuberLoss");

    const gert::Shape* inputShape = context->GetInputShape(IDX_INPUT);
    const gert::Shape* targetShape = context->GetInputShape(IDX_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);

    OP_CHECK_IF(inputShape->GetDimNum() != targetShape->GetDimNum(),
                OP_LOGE(context, "input and target rank mismatch: %zu vs %zu", inputShape->GetDimNum(),
                        targetShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    for (size_t i = 0; i < inputShape->GetDimNum(); ++i) {
        const int64_t a = inputShape->GetDim(i);
        const int64_t b = targetShape->GetDim(i);
        // An unknown dim (< 0) carries no information yet, so refusing it here
        // would reject legal dynamic-shape graphs. It must not become a
        // broadcast entry point either: tiling re-checks the concrete shapes,
        // where they are known.
        if (a < 0 || b < 0) {
            continue;
        }
        OP_CHECK_IF(a != b, OP_LOGE(context, "shape mismatch at dim %zu: %ld vs %ld", i, a, b),
                    return ge::GRAPH_FAILED);
    }

    gert::Shape* outShape = context->GetOutputShape(IDX_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* reductionPtr = attrs->GetInt(ATTR_REDUCTION);
    OP_CHECK_NULL_WITH_CONTEXT(context, reductionPtr);
    const int64_t reduction = *reductionPtr;

    if (reduction == HUBER_LOSS_REDUCE_NONE) {
        *outShape = *inputShape;
    } else if (reduction == HUBER_LOSS_REDUCE_MEAN || reduction == HUBER_LOSS_REDUCE_SUM) {
        // A true rank-0 scalar. Not a shape of {1}: that is rank 1, which
        // breaks the scalar contract.
        outShape->SetDimNum(0);
    } else {
        OP_LOGE(context, "reduction must be 0 (none), 1 (mean) or 2 (sum), got %ld", reduction);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeHuberLoss");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeHuberLoss(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeHuberLoss");

    const ge::DataType inputDtype = context->GetInputDataType(IDX_INPUT);
    const ge::DataType targetDtype = context->GetInputDataType(IDX_TARGET);
    OP_CHECK_IF(inputDtype != targetDtype,
                OP_LOGE(context, "input and target dtype mismatch: %d vs %d", static_cast<int>(inputDtype),
                        static_cast<int>(targetDtype)),
                return ge::GRAPH_FAILED);

    // Output dtype follows the input for every reduction mode; the fp32
    // accumulation is internal and narrows once before the store.
    context->SetOutputDataType(IDX_OUT, inputDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeHuberLoss");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HuberLoss).InferShape(InferShapeHuberLoss).InferDataType(InferDataTypeHuberLoss);

} // namespace ops
