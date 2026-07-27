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
 * \file sync_batch_norm_backward_reduce_infershape.cpp
 * \brief SyncBatchNormBackwardReduce shape / dtype 推导。
 *   两个输出 sum_dy_xmu、y 的 shape 与 dtype 均与输入 sum_dy 一致。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_SUMDY_IDX = 0;
static constexpr size_t OUTPUT_SUMDYXMU_IDX = 0;
static constexpr size_t OUTPUT_Y_IDX = 1;

static ge::graphStatus InferShapeSyncBatchNormBackwardReduce(gert::InferShapeContext* context)
{
    const gert::Shape* inShape = context->GetInputShape(INPUT_SUMDY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    gert::Shape* sumDyXmuShape = context->GetOutputShape(OUTPUT_SUMDYXMU_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumDyXmuShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *sumDyXmuShape = *inShape;
    *yShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSyncBatchNormBackwardReduce(gert::InferDataTypeContext* context)
{
    ge::DataType dtype = context->GetInputDataType(INPUT_SUMDY_IDX);
    context->SetOutputDataType(OUTPUT_SUMDYXMU_IDX, dtype);
    context->SetOutputDataType(OUTPUT_Y_IDX, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SyncBatchNormBackwardReduce)
    .InferShape(InferShapeSyncBatchNormBackwardReduce)
    .InferDataType(InferDataTypeSyncBatchNormBackwardReduce);
} // namespace ops
