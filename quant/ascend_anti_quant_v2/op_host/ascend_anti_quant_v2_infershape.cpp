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
 * \file ascend_quant_v2.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
constexpr size_t g_AttrDstType = 0;

static graphStatus InferShapeForAscendAntiQuantV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForAscendAntiQuantV2");
    const gert::Shape* inputXShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXShape);

    gert::Shape* outputShape = context->GetOutputShape(0);
    
    *outputShape = *inputXShape;
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForAscendAntiQuantV2");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForAscendAntiQuantV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForAscendAntiQuantV2");
    const int32_t* dstDtype = context->GetAttrs()->GetAttrPointer<int32_t>(g_AttrDstType);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    context->SetOutputDataType(0, outDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForAscendAntiQuantV2");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AscendAntiQuantV2).InferShape(InferShapeForAscendAntiQuantV2).InferDataType(InferDataTypeForAscendAntiQuantV2);
} // namespace ops