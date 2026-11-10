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
 * \file max_pool_with_argmax_v3_graph_infer.cpp
 * \brief MaxPoolWithArgmaxV3 operator graph infer (InferDataType).
 *        InferShape is registered by op_host/max_pool_with_argmax_v3_infershape.cpp.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr size_t INDEX_DTYPE = 3;
constexpr size_t INDEX_OUT_MAX = 0;
constexpr size_t INDEX_OUT_INDICES = 1;
constexpr int64_t INT32_DTYPE = 3;
constexpr int64_t INT64_DTYPE = 9;
} // namespace

ge::graphStatus InferDataTypeMaxPoolWithArgmaxV3(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType x = context->GetInputDataType(INDEX_OUT_MAX);
    context->SetOutputDataType(INDEX_OUT_MAX, x);

    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType indicesDtype = *dstDtype == INT32_DTYPE ? ge::DT_INT32 : ge::DT_INT64;

    context->SetOutputDataType(INDEX_OUT_INDICES, indicesDtype);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MaxPoolWithArgmaxV3).InferDataType(InferDataTypeMaxPoolWithArgmaxV3);

} // namespace ops
