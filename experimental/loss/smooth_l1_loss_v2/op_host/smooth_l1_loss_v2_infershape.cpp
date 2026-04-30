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
 * \file smoothl1lossv2_infershape.cpp
 * \brief SmoothL1LossV2 infer shape
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../op_kernel/smooth_l1_loss_v2_tiling_data.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_PREDICT = 0;
static constexpr int64_t IDX_LABEL = 1;
static constexpr int64_t IDX_OUTPUT = 0;

static int32_t GetReductionType(gert::InferShapeContext* context)
{
    return smooth_l1_loss_v2_host::GetReductionTypeFromContext(context, false);
}

static ge::graphStatus InferShapeSmoothL1LossV2(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeSmoothL1LossV2");

    // get input shapes
    const gert::Shape* predictShape = context->GetInputShape(IDX_PREDICT);
    const gert::Shape* labelShape = context->GetInputShape(IDX_LABEL);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelShape);

    // check rank
    OP_CHECK_IF(
        predictShape->GetDimNum() != labelShape->GetDimNum(),
        OP_LOGE(
            context, "Predict and label rank mismatch: %zu vs %zu", predictShape->GetDimNum(), labelShape->GetDimNum()),
        return ge::GRAPH_FAILED);

    // check each dimension
    for (size_t i = 0; i < predictShape->GetDimNum(); ++i) {
        OP_CHECK_IF(
            predictShape->GetDim(i) != labelShape->GetDim(i),
            OP_LOGE(
                context, "Shape mismatch at dim %zu: %ld vs %ld", i, predictShape->GetDim(i), labelShape->GetDim(i)),
            return ge::GRAPH_FAILED);
    }

    // get output shape
    gert::Shape* outShape = context->GetOutputShape(IDX_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    int32_t reductionType = GetReductionType(context);
    if (reductionType == 0) {
        *outShape = *predictShape;
    } else if (reductionType == 1 || reductionType == 2) {
        // Reduction output is scalar for sum/mean.
        outShape->SetDimNum(0);
    } else {
        OP_LOGE(context, "Invalid reduction attr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeSmoothL1LossV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SmoothL1LossV2).InferShape(InferShapeSmoothL1LossV2);

} // namespace ops
