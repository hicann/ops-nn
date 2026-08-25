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
 * \file cross_entropy_sum_exp_and_index_logit_graph_infer.cpp
 * \brief CrossEntropySumExpAndIndexLogit dtype inference（由 op_host infershape 移入 op_graph）
 *
 * - predicted_logits / sum_exp_logits / exp_logits 固定 float32；
 * - target_offset / target_mask 固定 int32。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

namespace {
// 输出索引
constexpr size_t OUTPUT_PREDICTED_LOGITS = 0;
constexpr size_t OUTPUT_SUM_EXP_LOGITS = 1;
constexpr size_t OUTPUT_EXP_LOGITS = 2;
constexpr size_t OUTPUT_TARGET_OFFSET = 3;
constexpr size_t OUTPUT_TARGET_MASK = 4;
} // namespace

static ge::graphStatus InferDataType4CrossEntropySumExpAndIndexLogit(gert::InferDataTypeContext* context)
{
    // predicted_logits / sum_exp_logits / exp_logits 固定 float32，target_offset / target_mask 固定 int32。
    context->SetOutputDataType(OUTPUT_PREDICTED_LOGITS, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_SUM_EXP_LOGITS, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_EXP_LOGITS, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_TARGET_OFFSET, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_TARGET_MASK, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(CrossEntropySumExpAndIndexLogit).InferDataType(InferDataType4CrossEntropySumExpAndIndexLogit);

} // namespace ops
