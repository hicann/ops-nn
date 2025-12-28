/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_put_v2.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
[[maybe_unused]] static ge::graphStatus InferShapeForIndexPutV2(gert::InferShapeContext* context) {
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexPutV2).InferShape(InferShapeForIndexPutV2);
}  // namespace ops
