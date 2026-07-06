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
 * \file lamb_update_with_lr_v2_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
constexpr size_t IN_X4 = 3;
constexpr size_t IN_X5 = 4;
constexpr size_t OUT_IDX = 0;

static ge::graphStatus InferShape4LambUpdateWithLrV2(gert::InferShapeContext* context)
{
    auto x4 = context->GetInputShape(IN_X4);
    OP_CHECK_NULL_WITH_CONTEXT(context, x4);
    auto x5 = context->GetInputShape(IN_X5);
    OP_CHECK_NULL_WITH_CONTEXT(context, x5);
    auto out = context->GetOutputShape(OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, out);
    OP_CHECK_IF(!BroadcastShape(x4, x5, out),
                OP_LOGE(context->GetNodeName(), "shape %s and %s cannot broadcast!", ToString(*x4).c_str(),
                        ToString(*x5).c_str()),
                return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(LambUpdateWithLrV2).InferShape(InferShape4LambUpdateWithLrV2);
} // namespace ops
