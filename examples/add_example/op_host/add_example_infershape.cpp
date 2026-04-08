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
 * \file add_example_infershape.cpp
 * \brief Add算子的shape推理和数据类型推理实现
 * 
 * 本文件提供了推理逻辑，用于确定Add算子的输出张量shape和数据类型。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

// 常量索引定义
static constexpr int64_t IDX_0 = 0;

/*!
 * \brief 推理Add算子的输出shape
 * 
 * 该函数检索输入张量的shape并将其传播到输出张量。
 * 对于Add算子，输出shape与输入shape相同。
 * 
 * @param context 指向shape推理上下文的指针
 * @return 如果推理成功则返回ge::graphStatus GRAPH_SUCCESS，否则返回错误代码
 */
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeAddExample");

    // 获取输入x的shape信息
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // 获取输出y的shape信息
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 填充输出shape的维度和大小
    // Add算子的输出shape与输入shape相同
    auto xShapeSize = xShape->GetDimNum();
    yShape->SetDimNum(xShapeSize);
    
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeAddExample");
    return GRAPH_SUCCESS;
}

/*!
 * \brief 推理Add算子的输出数据类型
 * 
 * 该函数检索输入张量的数据类型并将其传播到输出张量。
 * 对于Add算子，输出数据类型与输入数据类型相同。
 * 
 * @param context 指向数据类型推理上下文的指针
 * @return 如果推理成功则返回ge::graphStatus GRAPH_SUCCESS，否则返回错误代码
 */
static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeAddExample");

    // 设置输出的数据类型
    // Add算子的输出数据类型与输入数据类型相同
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, sizeDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeAddExample");
    return GRAPH_SUCCESS;
}

// infershape注册入口
// 将shape推理函数和数据类型推理函数注册到系统中
IMPL_OP_INFERSHAPE(AddExample).InferShape(InferShapeAddExample).InferDataType(InferDataTypeAddExample);
} // namespace ops