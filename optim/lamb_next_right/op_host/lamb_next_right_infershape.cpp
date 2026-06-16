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
 * \file lamb_next_right_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
constexpr size_t IN_SQUARE = 0;
constexpr size_t IN_MUL2 = 1;
constexpr size_t OUT_NUM = 2;

static ge::graphStatus InferShape4LambNextRight(gert::InferShapeContext* context)
{
    auto sq = context->GetInputShape(IN_SQUARE);
    OP_CHECK_NULL_WITH_CONTEXT(context, sq);
    auto v = context->GetInputShape(IN_MUL2);
    OP_CHECK_NULL_WITH_CONTEXT(context, v);
    for (size_t i = 0; i < OUT_NUM; i++) {
        auto out = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, out);
        OP_CHECK_IF(
            !BroadcastShape(sq, v, out),
            OP_LOGE(
                context->GetNodeName(), "shape %s and %s cannot broadcast!", ToString(*sq).c_str(),
                ToString(*v).c_str()),
            return ge::GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(LambNextRight).InferShape(InferShape4LambNextRight);
} // namespace ops
