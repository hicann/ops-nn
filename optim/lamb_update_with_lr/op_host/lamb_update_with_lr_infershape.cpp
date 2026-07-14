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
 * \file lamb_update_with_lr_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
constexpr size_t IN_MUL1 = 4;
constexpr size_t IN_SUB = 5;
constexpr size_t OUT_IDX = 0;

static ge::graphStatus InferShape4LambUpdateWithLr(gert::InferShapeContext* context)
{
    auto upd = context->GetInputShape(IN_MUL1);
    OP_CHECK_NULL_WITH_CONTEXT(context, upd);
    auto param = context->GetInputShape(IN_SUB);
    OP_CHECK_NULL_WITH_CONTEXT(context, param);
    auto out = context->GetOutputShape(OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, out);
    OP_CHECK_IF(!BroadcastShape(upd, param, out),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input_mul1 and input_sub",
                                                       (ToString(*upd) + " and " + ToString(*param)).c_str(),
                                                       "shapes cannot broadcast"),
                return ge::GRAPH_FAILED);
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(LambUpdateWithLr).InferShape(InferShape4LambUpdateWithLr);
} // namespace ops
