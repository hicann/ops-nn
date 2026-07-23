/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4FusedMulApplyMomentum(gert::InferShapeContext* context)
{
    static const std::vector<int64_t> out_idxs{0, 1};
    auto in_shape = context->GetInputShape(0);
    if (in_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (int64_t idx : out_idxs) {
        auto out_shape = context->GetOutputShape(idx);
        if (out_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *out_shape = *in_shape;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedMulApplyKerasMomentum).InferShape(InferShape4FusedMulApplyMomentum);

} // namespace ops
