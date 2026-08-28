/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShape4BNTrainingReduce(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(0);
    gert::Shape* sumShape = context->GetOutputShape(0);
    gert::Shape* squareSumShape = context->GetOutputShape(1);
    if (xShape == nullptr || xDesc == nullptr || sumShape == nullptr || squareSumShape == nullptr) {
        OP_LOGE(context, "BNTrainingReduce requires input x and two outputs.");
        return GRAPH_FAILED;
    }

    const bool isUnknownRank = xShape->GetDimNum() == 1 && xShape->GetDim(0) == UNKNOWN_DIM_NUM;
    if (isUnknownRank) {
        *sumShape = *xShape;
        *squareSumShape = *xShape;
        return GRAPH_SUCCESS;
    }

    const size_t rank = xShape->GetDimNum();
    const ge::Format format = xDesc->GetOriginFormat();
    size_t channelIndex = 0;
    if (format == FORMAT_NCHW && rank >= 2U && rank <= 4U) {
        channelIndex = 1U;
    } else if (format == FORMAT_NHWC && rank == 4U) {
        channelIndex = 3U;
    } else if (format == FORMAT_NCDHW && rank == 5U) {
        channelIndex = 1U;
    } else {
        OP_LOGE(context,
                "BNTrainingReduce on Ascend 950 only supports NCHW rank 2-4, NHWC rank 4 and NCDHW rank 5, "
                "but got format %d and rank %zu.",
                static_cast<int32_t>(format), rank);
        return GRAPH_FAILED;
    }

    const int64_t channel = xShape->GetDim(channelIndex);
    sumShape->SetDimNum(1);
    sumShape->SetDim(0, channel);
    squareSumShape->SetDimNum(1);
    squareSumShape->SetDim(0, channel);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BNTrainingReduce).InferShape(InferShape4BNTrainingReduce);
} // namespace ops
