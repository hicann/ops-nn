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
 * \file bn_training_update_grad_infershape.cpp
 * \brief BNTrainingUpdateGrad inferShape/inferDataType：
 *        diff_scale/diff_offset shape/dtype 同 batch_mean（fp32 [C]）。
 *        与 canndev built-in op_proto/runtime/bn_training_update_grad.cc 的
 *        InferShapeForBNTrainingUpdateGrad 功能一致，gert infershape 2.0 写法。
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {

static constexpr size_t INPUT_BATCH_MEAN_INDEX = 2;
static constexpr size_t OUTPUT_DIFF_SCALE_INDEX = 0;
static constexpr size_t OUTPUT_DIFF_OFFSET_INDEX = 1;

static ge::graphStatus BNTrainingUpdateGradInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do BNTrainingUpdateGradInferShape");

    const gert::Shape* batchMeanShape = context->GetRequiredInputShape(INPUT_BATCH_MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, batchMeanShape);

    gert::Shape* diffScaleShape = context->GetOutputShape(OUTPUT_DIFF_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, diffScaleShape);
    gert::Shape* diffOffsetShape = context->GetOutputShape(OUTPUT_DIFF_OFFSET_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, diffOffsetShape);

    // -2 UNKNOWN_RANK：两路输出同样未知，显式置 unknown rank 后早退（动态 rank 契约）
    if (IsUnknownRank(*batchMeanShape)) {
        SetUnknownRank(*diffScaleShape);
        SetUnknownRank(*diffOffsetShape);
        OP_LOGD(context->GetNodeName(), "End to do BNTrainingUpdateGradInferShape with unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    *diffScaleShape = *batchMeanShape;
    *diffOffsetShape = *batchMeanShape;

    OP_LOGD(context->GetNodeName(), "End to do BNTrainingUpdateGradInferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BNTrainingUpdateGradInferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do BNTrainingUpdateGradInferDataType");
    const ge::DataType batchMeanDataType = context->GetRequiredInputDataType(INPUT_BATCH_MEAN_INDEX);
    context->SetOutputDataType(OUTPUT_DIFF_SCALE_INDEX, batchMeanDataType);
    context->SetOutputDataType(OUTPUT_DIFF_OFFSET_INDEX, batchMeanDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BNTrainingUpdateGrad)
    .InferShape(BNTrainingUpdateGradInferShape)
    .InferDataType(BNTrainingUpdateGradInferDataType);
} // namespace ops
