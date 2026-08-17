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
 * \file bn_training_update_v2_infershape.cpp
 * \brief BNTrainingUpdateV2 inferShape/inferDataType：
 *        y shape/dtype 同 x；batch_mean/batch_variance shape/dtype 同 scale（fp32 [C]）。
 *        与 canndev built-in op_proto/runtime/bn_training_update_v2.cc 的
 *        InferShapeForBNTrainingUpdateV2 功能一致（y=x、batch_mean=batch_variance=scale），
 *        gert infershape 2.0 写法；inferDataType 对齐 canndev reduce_ops.cc 的
 *        BNTrainingUpdateV2InferShape 中 SetDataType 推导语义。
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {

static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_SCALE_INDEX = 3;
static constexpr size_t OUTPUT_Y_INDEX = 0;
static constexpr size_t OUTPUT_BATCH_MEAN_INDEX = 1;
static constexpr size_t OUTPUT_BATCH_VAR_INDEX = 2;

static ge::graphStatus BNTrainingUpdateV2InferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do BNTrainingUpdateV2InferShape");

    const gert::Shape* xShape = context->GetRequiredInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* scaleShape = context->GetRequiredInputShape(INPUT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* batchMeanShape = context->GetOutputShape(OUTPUT_BATCH_MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, batchMeanShape);
    gert::Shape* batchVarShape = context->GetOutputShape(OUTPUT_BATCH_VAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, batchVarShape);

    // -2 UNKNOWN_RANK：三路输出同样未知，显式置 unknown rank 后早退（动态 rank 契约）
    if (IsUnknownRank(*xShape)) {
        SetUnknownRank(*yShape);
        SetUnknownRank(*batchMeanShape);
        SetUnknownRank(*batchVarShape);
        OP_LOGD(context->GetNodeName(), "End to do BNTrainingUpdateV2InferShape with unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    *yShape = *xShape;
    *batchMeanShape = *scaleShape;
    *batchVarShape = *scaleShape;

    OP_LOGD(context->GetNodeName(), "End to do BNTrainingUpdateV2InferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BNTrainingUpdateV2InferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do BNTrainingUpdateV2InferDataType");
    const ge::DataType xDataType = context->GetRequiredInputDataType(INPUT_X_INDEX);
    const ge::DataType scaleDataType = context->GetRequiredInputDataType(INPUT_SCALE_INDEX);
    context->SetOutputDataType(OUTPUT_Y_INDEX, xDataType); // y 同 x
    context->SetOutputDataType(OUTPUT_BATCH_MEAN_INDEX, scaleDataType);
    context->SetOutputDataType(OUTPUT_BATCH_VAR_INDEX, scaleDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BNTrainingUpdateV2)
    .InferShape(BNTrainingUpdateV2InferShape)
    .InferDataType(BNTrainingUpdateV2InferDataType);
} // namespace ops
