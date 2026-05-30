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
 * \file index_put_with_sort_v2_infershape.cpp
 * \brief InferShape implementation for IndexPutWithSortV2 operator
 * 
 * IndexPutWithSortV2 puts values into tensor at sorted indices.
 * Output shape is same as input self tensor shape.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {

/**
 * @brief InferShape for IndexPutWithSortV2
 * 
 * Input:
 *   - self (input 0): tensor to be modified
 *   - linear_index (input 1): linear index tensor
 *   - pos_idx (input 2): position index tensor
 *   - values (input 3): values to put
 * 
 * Output:
 *   - self (output 0): output tensor, shape same as input self
 */
static ge::graphStatus InferShapeForIndexPutWithSortV2(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    // Get input self shape
    auto inSelfShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inSelfShape);
    // Get output shape
    auto outSelfShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outSelfShape);

    if (IsUnknownRank(*inSelfShape)) {
        SetUnknownRank(*outSelfShape);
        OP_LOGI(context->GetNodeName(), "End InferShapeForIndexPutWithSortV2 (UNKNOWN RANK)");
        return ge::GRAPH_SUCCESS;
    }
    if (IsUnknownShape(*inSelfShape)) {
        auto rankNum = inSelfShape->GetDimNum();
        SetUnknownShape(rankNum, *outSelfShape);
        OP_LOGI(context->GetNodeName(), "End InferShapeForIndexPutWithSortV2 (UNKNOWN SHAPE)");
        return ge::GRAPH_SUCCESS;
    }
    // Output shape is same as input self shape
    *outSelfShape = *inSelfShape;
    
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForIndexPutWithSortV2(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const ge::DataType xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexPutWithSortV2).InferShape(InferShapeForIndexPutWithSortV2).InferDataType(InferDataTypeForIndexPutWithSortV2);

}  // namespace ops