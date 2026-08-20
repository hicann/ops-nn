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
 * \file bn_training_update_v3_infershape.cpp
 * \brief BNTrainingUpdateV3 inferShape/inferDataType：
 *        y shape/dtype 同 x；batch_mean/batch_variance/reserve_1/reserve_2 shape/dtype 同 scale（fp32 [C]）。
 *        与 canndev built-in op_proto/reduce_ops.cc 的 BNTrainingUpdateV3InferShape 功能一致
 *        （y=x、其余四路统计量输出=scale），gert infershape 2.0 写法。
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
static constexpr size_t OUTPUT_RESERVE_1_INDEX = 3;
static constexpr size_t OUTPUT_RESERVE_2_INDEX = 4;

static ge::graphStatus BNTrainingUpdateV3InferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do BNTrainingUpdateV3InferShape");

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
    gert::Shape* reserve1Shape = context->GetOutputShape(OUTPUT_RESERVE_1_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reserve1Shape);
    gert::Shape* reserve2Shape = context->GetOutputShape(OUTPUT_RESERVE_2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reserve2Shape);

    // -2 UNKNOWN_RANK：五路输出同样未知，显式置 unknown rank 后早退（动态 rank 契约）
    if (IsUnknownRank(*xShape)) {
        SetUnknownRank(*yShape);
        SetUnknownRank(*batchMeanShape);
        SetUnknownRank(*batchVarShape);
        SetUnknownRank(*reserve1Shape);
        SetUnknownRank(*reserve2Shape);
        OP_LOGD(context->GetNodeName(), "End to do BNTrainingUpdateV3InferShape with unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    *yShape = *xShape;
    *batchMeanShape = *scaleShape;
    *batchVarShape = *scaleShape;
    *reserve1Shape = *scaleShape;
    *reserve2Shape = *scaleShape;

    OP_LOGD(context->GetNodeName(), "End to do BNTrainingUpdateV3InferShape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BNTrainingUpdateV3InferDataType(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do BNTrainingUpdateV3InferDataType");
    const ge::DataType xDataType = context->GetRequiredInputDataType(INPUT_X_INDEX);
    const ge::DataType scaleDataType = context->GetRequiredInputDataType(INPUT_SCALE_INDEX);
    context->SetOutputDataType(OUTPUT_Y_INDEX, xDataType); // y 同 x
    context->SetOutputDataType(OUTPUT_BATCH_MEAN_INDEX, scaleDataType);
    context->SetOutputDataType(OUTPUT_BATCH_VAR_INDEX, scaleDataType);
    context->SetOutputDataType(OUTPUT_RESERVE_1_INDEX, scaleDataType);
    context->SetOutputDataType(OUTPUT_RESERVE_2_INDEX, scaleDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BNTrainingUpdateV3)
    .InferShape(BNTrainingUpdateV3InferShape)
    .InferDataType(BNTrainingUpdateV3InferDataType);
} // namespace ops
