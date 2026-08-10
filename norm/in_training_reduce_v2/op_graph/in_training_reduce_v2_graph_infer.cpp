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
 * \file in_training_reduce_v2_graph_infer.cpp
 * \brief INTrainingReduceV2 InferDataType：sum / square_sum 恒 fp32（不随输入 dtype）。
 *        InferDataType 仅图场景使用，故落在 op_graph；InferShape 图与单算子共用，留在 op_host。
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferDataType4INTrainingReduceV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4INTrainingReduceV2");
    // 输出恒 fp32（不随输入 dtype）。
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4INTrainingReduceV2");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(INTrainingReduceV2).InferDataType(InferDataType4INTrainingReduceV2);
} // namespace ops
