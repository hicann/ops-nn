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
 * \file softmax_focal_loss_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
constexpr uint64_t INPUT_PRED_IDX = 0;
constexpr uint64_t INPUT_TARGET_IDX = 1;
constexpr uint64_t OUTPUT_Y_IDX = 0;
} // namespace

namespace ops {
static graphStatus InferShape4SoftmaxFocalLoss(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4SoftmaxFocalLoss.");
    const gert::Shape* predShape = context->GetInputShape(INPUT_PRED_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, predShape);
    const gert::Shape* targetShape = context->GetInputShape(INPUT_TARGET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (!Ops::Base::IsUnknownRank(*predShape) && !Ops::Base::IsUnknownRank(*targetShape)) {
        OP_CHECK_IF(
            predShape->GetDimNum() != targetShape->GetDimNum(),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "target", std::to_string(targetShape->GetDimNum()), "must equal pred.dimNum"),
            return ge::GRAPH_FAILED);
    }

    // 计算侧不做 reduce, 输出恒与 pred 同形(逐行的 loss 广播回整行), 与 A2 计算语义一致
    *yShape = *predShape;
    OP_LOGD(context, "InferShape4SoftmaxFocalLoss End.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4SoftmaxFocalLoss(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDataType4SoftmaxFocalLoss Begin.");
    context->SetOutputDataType(OUTPUT_Y_IDX, context->GetInputDataType(INPUT_PRED_IDX));
    OP_LOGD(context, "InferDataType4SoftmaxFocalLoss End.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftmaxFocalLoss)
    .InferShape(InferShape4SoftmaxFocalLoss)
    .InferDataType(InferDataType4SoftmaxFocalLoss);
} // namespace ops
