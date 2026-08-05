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
 * \file poisson_nll_loss_infershape.cpp
 * \brief
 */
#include <string>
#include "util/shape_util.h"
#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
// Attribute index in def.cpp: log_input(0), full(1), eps(2), reduction(3)
static constexpr size_t ATTR_REDUCTION_IDX = 3;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

static ge::graphStatus InferShapePoissonNllLoss(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapePoissonNllLoss");

    // Get input shapes
    const gert::Shape* inputShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape* targetShape = context->GetInputShape(IDX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);

    // input_x and target must have the exact same shape (non-broadcast operator).
    // This mirrors the ascend910b TBE entry gate `if not operator.eq(shape_input, shape_target): raise`
    // (canndev/ops/built-in/tbe/impl/poisson_nll_loss.py L134-135). ascend910b's compute-layer
    // tbe.broadcast is dead code because the entry already forces equal shapes, so ascend910b does NOT
    // truly support broadcasting; ascend950 keeps the same contract and rejects mismatched shapes here
    // rather than silently proceeding on input_x's shape. Unknown-rank/dim are skipped (dynamic).
    if (!Ops::Base::IsUnknownRank(*inputShape) && !Ops::Base::IsUnknownRank(*targetShape)) {
        if (inputShape->GetDimNum() != targetShape->GetDimNum()) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context->GetNodeName(), "input_x, target",
                (std::to_string(inputShape->GetDimNum()) + ", " + std::to_string(targetShape->GetDimNum())).c_str(),
                "the rank of input_x must be equal to target");
            return GRAPH_FAILED;
        }
        for (size_t i = 0; i < inputShape->GetDimNum(); i++) {
            if (inputShape->GetDim(i) != UNKNOWN_DIM_VALUE_ && targetShape->GetDim(i) != UNKNOWN_DIM_VALUE_ &&
                inputShape->GetDim(i) != targetShape->GetDim(i)) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "input_x, target",
                    (Ops::Base::ToString(*inputShape) + ", " + Ops::Base::ToString(*targetShape)).c_str(),
                    "the shape of input_x must be equal to target");
                return GRAPH_FAILED;
            }
        }
    }

    // Get output shape
    gert::Shape* outputShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // Get reduction attribute (index 3)
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* reduction = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reduction);

    // Set output shape based on reduction mode
    if (strcmp(reduction, "none") == 0) {
        // Output has the same shape as input (input_x and target are equal-shape, checked above).
        *outputShape = *inputShape;
    } else {
        // "mean" or "sum": output is scalar
        outputShape->SetDimNum(0);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapePoissonNllLoss");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PoissonNllLoss).InferShape(InferShapePoissonNllLoss);
} // namespace ops
