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
 * \file bn_training_reduce_grad_infershape.cpp
 * \brief BNTrainingReduceGrad inferShape/inferDataType：
 *        y shape/dtype 同 grads（与 canndev built-in op_proto/runtime/bn_training_reduce_grad.cc 的
 *        InferShape4Elewise 功能一致），gert infershape 2.0 写法。
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {

static constexpr size_t INPUT_GRADS_INDEX = 0;
static constexpr size_t OUTPUT_Y_INDEX = 0;

static ge::graphStatus BNTrainingReduceGradInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do BNTrainingReduceGradInferShape");

    const gert::Shape* gradsShape = context->GetRequiredInputShape(INPUT_GRADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradsShape);

    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // -2 UNKNOWN_RANK：输出同样未知，显式置 unknown rank 后早退（动态 rank 契约）
    if (IsUnknownRank(*gradsShape)) {
        SetUnknownRank(*yShape);
        OP_LOGD(context->GetNodeName(), "End to do BNTrainingReduceGradInferShape with unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    *yShape = *gradsShape;

    OP_LOGD(context->GetNodeName(), "End to do BNTrainingReduceGradInferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BNTrainingReduceGradInferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do BNTrainingReduceGradInferDataType");
    const ge::DataType gradsDataType = context->GetRequiredInputDataType(INPUT_GRADS_INDEX);
    context->SetOutputDataType(OUTPUT_Y_INDEX, gradsDataType); // y 同 grads
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BNTrainingReduceGrad)
    .InferShape(BNTrainingReduceGradInferShape)
    .InferDataType(BNTrainingReduceGradInferDataType);
} // namespace ops
