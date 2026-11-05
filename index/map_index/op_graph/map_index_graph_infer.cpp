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
 * \file map_index_graph_infer.cpp
 * \brief MapIndex graph InferDataType implementation
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static constexpr int64_t IDX_0 = 0;

ge::graphStatus InferDataTypeMapIndex(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMapIndex");
    // 输出 y 为映射索引结果，固定 INT32，不随输入 dtype
    context->SetOutputDataType(IDX_0, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMapIndex");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MapIndex).InferDataType(InferDataTypeMapIndex);
} // namespace ops
