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
 * \file inplace_apply_adagrad_infershape.cpp
 * \brief InplaceApplyAdagrad infershape.
 */

#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
inline ge::graphStatus CopyShapeInputToOutputWithIdx(gert::InferShapeContext* context, int64_t inputIdx,
                                                     int64_t outputIdx)
{
    auto inShape = context->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto outShape = context->GetOutputShape(outputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    if (Ops::Base::IsUnknownRank(*inShape)) {
        Ops::Base::SetUnknownRank(*outShape);
        return ge::GRAPH_SUCCESS;
    }
    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4InplaceApplyAdagrad(gert::InferShapeContext* context)
{
    constexpr size_t inputVarIdx = 0;
    constexpr size_t inputAccumIdx = 1;
    constexpr size_t outputVarIdx = 0;
    constexpr size_t outputAccumIdx = 1;
    OP_CHECK_IF(CopyShapeInputToOutputWithIdx(context, inputVarIdx, outputVarIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "var output", "copy shape failed",
                                                      "Copy input var shape to output var failed"),
                return ge::GRAPH_FAILED);
    return CopyShapeInputToOutputWithIdx(context, inputAccumIdx, outputAccumIdx);
}

IMPL_OP_INFERSHAPE(InplaceApplyAdagrad).InferShape(InferShape4InplaceApplyAdagrad);

} // namespace ops
