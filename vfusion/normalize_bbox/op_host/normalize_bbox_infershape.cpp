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
 * \file normalize_bbox_infershape.cpp
 * \brief y shares shape and dtype of boxes
 */
#include "log/log.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_BOXES_IDX = 0;
static constexpr size_t OUTPUT_Y_IDX = 0;

static ge::graphStatus NormalizeBBoxInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do NormalizeBBoxInferShape");

    const gert::Shape* boxesShape = context->GetInputShape(INPUT_BOXES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *boxesShape;

    OP_LOGD(context, "End to do NormalizeBBoxInferShape");
    return ge::GRAPH_SUCCESS;
}

// InferDataType is graph-only and now lives in op_graph/normalize_bbox_graph_infer.cpp;
// only InferShape (shared by graph and single-op paths) is registered here.
IMPL_OP_INFERSHAPE(NormalizeBBox).InferShape(NormalizeBBoxInferShape);
} // namespace ops
