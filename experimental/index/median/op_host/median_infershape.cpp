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
 * \file median_infershape.cpp
 * \brief Median 算子的 shape 推理和数据类型推理实现
 *
 * 本文件提供推理逻辑，确定 Median 算子的输出张量 shape 和数据类型。
 * Median 为 1 输入（input）/ 2 输出（values, indices），沿末轴 reduce。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

// 常量索引定义
static constexpr int64_t IDX_0 = 0;

/*!
 * \brief 推理 Median 算子的输出 shape
 *
 * values / indices 的 shape = 输入 shape 去掉末轴（kernel 为末轴 reduce 语义，
 * 非末轴 dim 由 L2 wrapper 经 Transpose 换到末轴）。
 *
 * @param context 指向 shape 推理上下文的指针
 * @return 推理成功返回 ge::graphStatus GRAPH_SUCCESS，否则返回错误代码
 */
static ge::graphStatus InferShapeMedian(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMedian");

    // 获取输入 input 的 shape 信息
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // 获取输出 values / indices 的 shape 信息
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* iShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, iShape);

    // values shape = input shape 去掉末轴；indices 与 values 同 shape
    auto n = xShape->GetDimNum();
    yShape->SetDimNum(n > 0 ? n - 1 : 0);
    for (size_t i = 0; i + 1 < n; i++)
        yShape->SetDim(i, xShape->GetDim(i));
    *iShape = *yShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeMedian");
    return GRAPH_SUCCESS;
}

/*!
 * \brief 推理 Median 算子的输出数据类型
 *
 * values 与 input 同 dtype；indices 固定为 int32。
 *
 * @param context 指向数据类型推理上下文的指针
 * @return 推理成功返回 ge::graphStatus GRAPH_SUCCESS，否则返回错误代码
 */
static ge::graphStatus InferDataTypeMedian(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMedian");

    // values 与 input 同 dtype；indices 为 int32
    context->SetOutputDataType(IDX_0, context->GetInputDataType(IDX_0));
    context->SetOutputDataType(1, ge::DT_INT32);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMedian");
    return GRAPH_SUCCESS;
}

// infershape 注册入口：注册 shape 推理函数与数据类型推理函数
IMPL_OP_INFERSHAPE(Median).InferShape(InferShapeMedian).InferDataType(InferDataTypeMedian);
} // namespace ops
