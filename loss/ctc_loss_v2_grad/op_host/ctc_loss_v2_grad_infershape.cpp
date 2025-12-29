/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file ctc_loss_v2_grad_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"

#define CHECK(cond, log_func, return_expr) \
    do {                                   \
        if (cond) {                        \
            log_func;                      \
            return_expr;                   \
        }                                  \
    } while (0)

using namespace ge;
namespace ops {
template <typename T>
static ge::graphStatus GetMaxValueAndSumFromTensor(
    const T* value, const int64_t batch_size, int64_t& max_length, int64_t& target_lengths_sum)
{
    for (int64_t i = 0; i < batch_size; i++) {
        T current_value = value[i];
        max_length = max_length > current_value ? max_length : current_value;
        target_lengths_sum = target_lengths_sum + current_value;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CTCLossV2GradInferShapeFunc(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "CTCLossV2GradInferShape run.");
    auto const log_probs_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, log_probs_shape);
    auto grad_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_shape);

    *grad_shape = *log_probs_shape;

    OP_LOGD(context->GetNodeName(), "grad_shape: %s", Shape2String(*grad_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

static graphStatus CTCLossV2GradInferDtypeFunc(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do CTCLossV2GradInferDtype rt2.0");

    auto gradDtype = context->GetInputDataType(0);

    context->SetOutputDataType(0, gradDtype);
    OP_LOGD(context->GetNodeName(), "Datatype of grad is:%s", Ops::Base::ToString(gradDtype).c_str());
    OP_LOGD(context->GetNodeName(), "End to do CTCLossV2GradInferDtype");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CTCLossV2Grad).InferShape(CTCLossV2GradInferShapeFunc).InferDataType(CTCLossV2GradInferDtypeFunc);
} // namespace ops
