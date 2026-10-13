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
 * \file add_example_pypto_infershape.cpp
 * \brief AddExamplePypto算子的shape推理和数据类型推理实现
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

// 常量索引定义
static constexpr int64_t IDX_0 = 0;

/*!
 * \brief 推理AddExamplePypto算子的输出shape，输出shape与输入shape相同
 */
static ge::graphStatus InferShapeAddExamplePypto(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeAddExamplePypto");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto xShapeSize = xShape->GetDimNum();
    yShape->SetDimNum(xShapeSize);
    for (size_t i = 0; i < xShapeSize; i++) {
        yShape->SetDim(i, xShape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeAddExamplePypto");
    return GRAPH_SUCCESS;
}

/*!
 * \brief 推理AddExamplePypto算子的输出数据类型，输出dtype与输入dtype相同
 */
static ge::graphStatus InferDataTypeAddExamplePypto(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeAddExamplePypto");

    ge::DataType xDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, xDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeAddExamplePypto");
    return GRAPH_SUCCESS;
}

// infershape注册入口
IMPL_OP_INFERSHAPE(AddExamplePypto).InferShape(InferShapeAddExamplePypto).InferDataType(InferDataTypeAddExamplePypto);
} // namespace ops
