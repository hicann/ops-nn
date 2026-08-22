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
 * \file bn3d_training_reduce_graph_infer.cpp
 * \brief BN3DTrainingReduce InferDataType：sum / square_sum 恒 fp32（不随输入 dtype）。
 *        InferDataType 仅图场景使用，故落在 op_graph；InferShape 图与单算子共用，留在 op_host。
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
// 与 op_host/bn3d_training_reduce_tiling.h 里的同名常量取值一致；两处分属 op_graph 与
// op_host，不跨层 include，故各留一份，靠 REG_OP 的输出序（sum, square_sum）约束。
constexpr size_t OUTPUT_SUM_INDEX = 0;
constexpr size_t OUTPUT_SQUARE_SUM_INDEX = 1;
} // namespace

static ge::graphStatus InferDataType4BN3DTrainingReduce(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4BN3DTrainingReduce");
    // 输出恒 fp32（不随输入 dtype，fp16 / bf16 在内核内提升为 fp32 累加）。
    context->SetOutputDataType(OUTPUT_SUM_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_SQUARE_SUM_INDEX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4BN3DTrainingReduce");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(BN3DTrainingReduce).InferDataType(InferDataType4BN3DTrainingReduce);
} // namespace ops
