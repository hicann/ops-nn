/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "cosine_embedding_loss_common.h"

#include <cstring>
#include <string>

using namespace ge;
namespace ops {
namespace {
constexpr size_t INPUT_X1 = 0;
constexpr size_t INPUT_X2 = 1;
constexpr size_t INPUT_TARGET = 2;
constexpr size_t OUTPUT_Y = 0;
constexpr size_t ATTR_REDUCTION = 1;
namespace cel = cosine_embedding_loss;
} // namespace

// y = broadcast(remove_axis_1(broadcast(x1, x2)), target) for reduction "none";
// y = [1] for "sum"/"mean".
static ge::graphStatus InferShapeCosineEmbeddingLoss(gert::InferShapeContext* context)
{
    auto x1_shape = context->GetInputShape(INPUT_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    auto x2_shape = context->GetInputShape(INPUT_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    auto target_shape = context->GetInputShape(INPUT_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context, target_shape);
    auto out_shape = context->GetOutputShape(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char* reduction = attrs->GetAttrPointer<char>(ATTR_REDUCTION); // margin(0), reduction(1)
    const char* reductionValue = cel::ReductionOrDefault(reduction);
    uint32_t reductionKey = cel::kReductionMean;
    if (!cel::ParseReduction(reduction, reductionKey)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "reduction", reductionValue,
                                              "reduction should be none, sum or mean");
        return GRAPH_FAILED;
    }

    if (Ops::Base::IsUnknownRank(*x1_shape) || Ops::Base::IsUnknownRank(*x2_shape) ||
        Ops::Base::IsUnknownRank(*target_shape)) {
        if (reductionKey == cel::kReductionNone) {
            Ops::Base::SetUnknownRank(*out_shape);
        } else {
            *out_shape = gert::Shape({1});
        }
        return GRAPH_SUCCESS;
    }

    cel::Dims x1Dims;
    cel::Dims x2Dims;
    cel::Dims targetDims;
    if (!cel::ShapeToDims(*x1_shape, x1Dims) || !cel::ShapeToDims(*x2_shape, x2Dims) ||
        !cel::ShapeToDims(*target_shape, targetDims)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "input", "invalid shape",
                                              "rank should be in [1, 8], and dimensions should be -1 or positive");
        return GRAPH_FAILED;
    }

    cel::Dims xBroadcastDims;
    if (!cel::BroadcastShapes(x1Dims, x2Dims, xBroadcastDims)) {
        std::string shapeMsg = Ops::Base::ToString(*x1_shape) + " and " + Ops::Base::ToString(*x2_shape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1 and x2", shapeMsg.c_str(),
                                               "x1 and x2 should be broadcastable");
        return GRAPH_FAILED;
    }
    if (xBroadcastDims.size() < 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x1 and x2",
                                                 std::to_string(xBroadcastDims.size()).c_str(),
                                                 "broadcast rank should be at least 2 for axis=1 reduction");
        return GRAPH_FAILED;
    }

    cel::Dims xReducedDims;
    if (!cel::RemoveAxis(xBroadcastDims, cel::kFeatureAxis, xReducedDims)) {
        return GRAPH_FAILED;
    }
    cel::Dims outputDims;
    if (!cel::BroadcastShapes(xReducedDims, targetDims, outputDims)) {
        std::string shapeMsg = Ops::Base::ToString(*target_shape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "target", shapeMsg.c_str(),
                                              "target should be broadcastable with x1/x2 shape after reducing axis 1");
        return GRAPH_FAILED;
    }
    if (reductionKey != cel::kReductionNone) {
        *out_shape = gert::Shape({1});
        return GRAPH_SUCCESS;
    }
    cel::SetShape(*out_shape, outputDims);
    return GRAPH_SUCCESS;
}

// y is always fp32 (IR: .OUTPUT(y, DT_FLOAT)), regardless of input dtype.
static ge::graphStatus InferDataTypeCosineEmbeddingLoss(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CosineEmbeddingLoss)
    .InferShape(InferShapeCosineEmbeddingLoss)
    .InferDataType(InferDataTypeCosineEmbeddingLoss);

} // namespace ops
