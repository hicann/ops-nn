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
 * \file linear_index_v2_infershape.cpp
 * \brief InferShape implementation for LinearIndexV2 operator
 *
 * LinearIndexV2 converts multiple indices into a linear index.
 * Output shape is same as the first indices tensor shape.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {

/**
 * @brief InferShape for LinearIndexV2
 *
 * Input:
 *   - indices_list (DYNAMIC): multiple indices tensors
 *   - stride: stride tensor
 *   - value_size: value size tensor
 *
 * Output:
 *   - index: linear index tensor, shape same as indices_list[0]
 */
static ge::graphStatus InferShapeForLinearIndexV2(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    // Get output shape
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    // Get the first indices tensor shape (indices_list is dynamic input, index 0)
    auto indicesShape = context->GetDynamicInputShape(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);

    if (IsUnknownRank(*indicesShape)) {
        SetUnknownRank(*outShape);
        OP_LOGI(context->GetNodeName(), "End InferShapeForLinearIndexV2 (UNKNOWN RANK)");
        return ge::GRAPH_SUCCESS;
    }
    if (IsUnknownShape(*indicesShape)) {
        SetUnknownShape(1, *outShape);
        OP_LOGI(context->GetNodeName(), "End InferShapeForLinearIndexV2 (UNKNOWN SHAPE)");
        return ge::GRAPH_SUCCESS;
    }
    gert::Shape resultShape({indicesShape->GetShapeSize()});
    // Output shape is same as first indices tensor shape
    *outShape = resultShape;

    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForLinearIndexV2(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    auto indicesDtype = context->GetDynamicInputDataType(0, 0);
    context->SetOutputDataType(0, indicesDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LinearIndexV2).InferShape(InferShapeForLinearIndexV2).InferDataType(InferDataTypeForLinearIndexV2);

} // namespace ops