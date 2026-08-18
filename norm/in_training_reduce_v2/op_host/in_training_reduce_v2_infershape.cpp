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

    // C 轴的位置由 **origin format** 决定，不能硬编码 dim 1：
    //   channel-first（NCHW / NCDHW / ND）—— C 在 dim 1；
    //   channel-last （NHWC / NDHWC）      —— C 在最后一维。
    // def.cpp 的 Format 列声明的是 storage format（kernel 吃什么物理排布），
    // 约束不到 origin format —— 后者是用户网络自带的轴语义，TF 来源的网络天然是
    // NHWC，GE 会自行插 TransData 把数据转成我们声明的 storage format 再喂 kernel，
    // 但 InferShape 跑在 TransData 插入之前，看到的仍是原始语义，必须自行分支。
    // 与 canndev（op_proto/runtime/in_training_reduce_v2.cc）及同仓 instance_norm
    // （op_host/instance_norm_infershape.cpp）的既定契约保持一致。
    auto x_desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_desc);
    ge::Format x_ori_format = x_desc->GetOriginFormat();
    bool is_channel_last = (x_ori_format == ge::FORMAT_NHWC || x_ori_format == ge::FORMAT_NDHWC);

    // 保留 N（dim 0）与 C，其余空间轴（H,W / D,H,W）置 1 —— keepdims。
    size_t x_dim_num = x_shape->GetDimNum();
    size_t c_dim_idx = (is_channel_last && x_dim_num > 0) ? (x_dim_num - 1) : 1;
    sum_shape->SetDimNum(x_dim_num);
    square_sum_shape->SetDimNum(x_dim_num);

    for (size_t i = 0; i < x_dim_num; i++) {
        if (i == 0 || i == c_dim_idx) {
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

// InferDataType 仅图场景使用，已按交付件划分挪到
// op_graph/in_training_reduce_v2_graph_infer.cpp；此处只挂图与单算子共用的 InferShape。
IMPL_OP_INFERSHAPE(INTrainingReduceV2).InferShape(InferShape4INTrainingReduceV2);
} // namespace ops
