/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_infershape.cpp
 * \brief
 */
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace ge;
namespace {
const size_t INDEX_PREDICT = 0;
const size_t INDEX_LABEL = 1;
const size_t INDEX_LOSS = 0;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
} // namespace

namespace ops {
static ge::graphStatus SmoothL1LossInferShapeFunc(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "SmoothL1LossInferShapeFunc begin.");

    auto predict = context->GetInputShape(INDEX_PREDICT);
    OP_CHECK_NULL_WITH_CONTEXT(context, predict);

    auto label = context->GetInputShape(INDEX_LABEL);
    OP_CHECK_NULL_WITH_CONTEXT(context, label);

    if (!Ops::Base::IsUnknownRank(*predict) && !Ops::Base::IsUnknownRank(*label)) {
        auto predictDimNum = predict->GetDimNum();
        auto labelDimNum = label->GetDimNum();
        if (predictDimNum != labelDimNum) {
            OP_LOGE(context->GetNodeName(), "Input[predict]'s shape is not equal to Input[label]'s.");
            return GRAPH_FAILED;
        }
        for (size_t i = 0; i < predictDimNum; i++) {
            if (predict->GetDim(i) != UNKNOWN_DIM_VALUE_ && label->GetDim(i) != UNKNOWN_DIM_VALUE_ &&
                predict->GetDim(i) != label->GetDim(i)) {
                OP_LOGE(context->GetNodeName(), "Input[predict]'s shape is not equal to Input[label]'s.");
                return GRAPH_FAILED;
            }
        }
    }

    auto loss = context->GetOutputShape(INDEX_LOSS);
    *loss = *predict;

    OP_LOGD(context->GetNodeName(), "SmoothL1LossInferShapeFunc end.");
    return ge::GRAPH_SUCCESS;
}

graphStatus SmoothL1LossInferDtypeFunc(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "SmoothL1LossInferDtypeFunc begin.");

    auto predictDtype = context->GetInputDataType(INDEX_PREDICT);
    OP_CHECK_IF((predictDtype != ge::DT_FLOAT) && (predictDtype != ge::DT_FLOAT16) && (predictDtype != ge::DT_BF16),
                OP_LOGE(context->GetNodeName(), "predict only support float, float16 and bfloat16"),
                return GRAPH_FAILED);

    context->SetOutputDataType(INDEX_LOSS, predictDtype);

    OP_LOGD(context->GetNodeName(), "SmoothL1LossInferDtypeFunc end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SmoothL1Loss).InferShape(SmoothL1LossInferShapeFunc).InferDataType(SmoothL1LossInferDtypeFunc);
} // namespace ops
