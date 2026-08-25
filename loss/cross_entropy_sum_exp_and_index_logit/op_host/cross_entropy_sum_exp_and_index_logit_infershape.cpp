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
 * \file cross_entropy_sum_exp_and_index_logit_infershape.cpp
 * \brief CrossEntropySumExpAndIndexLogit shape inference.
 *
 * - predicted_logits / sum_exp_logits / target_offset / target_mask 的 shape 与 target 相同
 *   （即 vocab_parallel_logits.shape[:-1]）。
 * - exp_logits 的 shape 与 vocab_parallel_logits 相同。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

namespace {
// 输入索引
constexpr size_t INPUT_LOGITS = 0;
constexpr size_t INPUT_TARGET = 1;

// 输出索引
constexpr size_t OUTPUT_EXP_LOGITS = 2;
constexpr size_t OUTPUT_NUM = 5;
} // namespace

static ge::graphStatus InferShape4CrossEntropySumExpAndIndexLogit(gert::InferShapeContext* context)
{
    // vocab_parallel_logits 与 target 的 shape 分别作为两类输出的 shape 来源。
    const gert::Shape* logitsShape = context->GetInputShape(INPUT_LOGITS);
    const gert::Shape* targetShape = context->GetInputShape(INPUT_TARGET);
    if (logitsShape == nullptr || targetShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // exp_logits 与 vocab_parallel_logits 同 shape；其余 4 个输出与 target 同 shape。
    for (size_t i = 0; i < OUTPUT_NUM; ++i) {
        gert::Shape* outShape = context->GetOutputShape(i);
        if (outShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (i == OUTPUT_EXP_LOGITS) {
            *outShape = *logitsShape;
        } else {
            *outShape = *targetShape;
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CrossEntropySumExpAndIndexLogit).InferShape(InferShape4CrossEntropySumExpAndIndexLogit);

} // namespace ops
