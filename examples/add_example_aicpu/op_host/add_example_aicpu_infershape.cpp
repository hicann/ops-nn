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
 * \file add_example_aicpu_infershape.cpp
 * \brief AddExampleAicpu算子的形状推导和数据类型推导实现
 * 
 * 本文件实现了在Ascend AI处理器上运行的AddExampleAicpu算子的形状推导和数据类型推导逻辑。
 * 形状推导是GE(Graph Engine)在图构建阶段自动进行的，用于确定输出张量的形状和数据类型，
 * 从而在执行前验证算子的正确性并优化内存分配。
 * 
 * 主要功能：
 * 1. InferShapeAddExample: 推导输出张量的形状信息
 * 2. InferDataTypeAddExample: 推导输出张量的数据类型
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

/*!
 * \brief 形状推导函数：确定输出张量的形状
 * 
 * 该函数在图编译阶段被调用，根据输入张量的形状推导输出张量的形状。
 * 对于加法操作，输出张量的形状与输入张量的形状完全相同。
 * 
 * 形状推导的流程：
 * 1. 从上下文中获取输入张量的形状
 * 2. 从上下文中获取输出张量的形状对象
 * 3. 将输入形状的维度数量和每个维度的大小复制到输出形状
 * 4. 返回成功状态
 * 
 * @param context 形状推导上下文，提供访问输入输出张量的接口
 * @return GRAPH_SUCCESS表示推导成功，其他值表示失败
 */
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeAddExample");

    // 获取输入张量0的形状指针
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // 获取输出张量0的形状指针
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // 将输出形状的维度数量设置为与输入相同
    auto xShapeSize = xShape->GetDimNum();
    yShape->SetDimNum(xShapeSize);
    
    // 遍历每个维度，将输入的维度大小复制到输出
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeAddExample");
    return GRAPH_SUCCESS;
}

/*!
 * \brief 数据类型推导函数：确定输出张量的数据类型
 * 
 * 该函数在图编译阶段被调用，根据输入张量的数据类型推导输出张量的数据类型。
 * 对于加法操作，输出张量的数据类型与输入张量的数据类型相同，以保持数值精度。
 * 
 * 数据类型推导的流程：
 * 1. 从上下文中获取输入张量的数据类型
 * 2. 将输出张量的数据类型设置为与输入相同
 * 3. 返回成功状态
 * 
 * @param context 数据类型推导上下文，提供访问输入输出张量的接口
 * @return GRAPH_SUCCESS表示推导成功，其他值表示失败
 */
static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeAddExample");

    // 获取输入张量0的数据类型
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    
    // 将输出张量0的数据类型设置为与输入相同
    context->SetOutputDataType(IDX_0, sizeDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeAddExample");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddExampleAicpu).InferShape(InferShapeAddExample).InferDataType(InferDataTypeAddExample);
} // namespace ops