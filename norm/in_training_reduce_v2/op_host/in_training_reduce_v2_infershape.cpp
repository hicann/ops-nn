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
 * \file in_training_reduce_v2_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {

static ge::graphStatus InferShape4INTrainingReduceV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4INTrainingReduceV2");

    // 输入 x
    const gert::Shape* x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    // 输出 sum / square_sum
    gert::Shape* sum_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sum_shape);
    gert::Shape* square_sum_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, square_sum_shape);

    // 保留 N（dim 0）、C（dim 1），其余空间轴（H,W / D,H,W）置 1 —— keepdims。
    size_t x_dim_num = x_shape->GetDimNum();
    sum_shape->SetDimNum(x_dim_num);
    square_sum_shape->SetDimNum(x_dim_num);

    for (size_t i = 0; i < x_dim_num; i++) {
        if (i == 0 || i == 1) {
            sum_shape->SetDim(i, x_shape->GetDim(i));
            square_sum_shape->SetDim(i, x_shape->GetDim(i));
        } else {
            sum_shape->SetDim(i, 1);
            square_sum_shape->SetDim(i, 1);
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4INTrainingReduceV2");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4INTrainingReduceV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4INTrainingReduceV2");
    // 输出恒 fp32（不随输入 dtype）。
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4INTrainingReduceV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(INTrainingReduceV2)
    .InferShape(InferShape4INTrainingReduceV2)
    .InferDataType(InferDataType4INTrainingReduceV2);
} // namespace ops
