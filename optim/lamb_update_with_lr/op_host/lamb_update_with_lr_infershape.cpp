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
 * \file lamb_update_with_lr_infershape.cpp
 * \brief
 */

#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
// 输出 shape 只由 input_greater1(in0) 与 input_sub(in5) 的广播结果决定；
// 其余 7 个输入分别按 in0/in5 的 placeholder shape 使用，不参与额外广播。
constexpr size_t INPUT_GREATER1_IDX = 0;
constexpr size_t INPUT_SUB_IDX = 5;
constexpr size_t OUT_NUM = 1;

static ge::graphStatus InferShape4LambUpdateWithLr(gert::InferShapeContext* context)
{
    const gert::Shape* greater1Shape = context->GetInputShape(INPUT_GREATER1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, greater1Shape);
    const gert::Shape* subShape = context->GetInputShape(INPUT_SUB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, subShape);

    gert::Shape bcShape;
    // 逐对折叠广播: 只用两参数重载。vector 重载虽在 op_common/op_host/infershape_broadcast_util.h
    // 以 Ops::Base 声明, 但 libops_base.so 只导出两参数版, vector 版的实现在 libop_common.so 的
    // 小写 ops 命名空间下 —— 声明与实现命名空间不一致, 用它会编译期通过、加载期
    // undefined symbol 而装不上包。
    OP_CHECK_IF(!BroadcastShape(greater1Shape, subShape, &bcShape),
                OP_LOGE(context->GetNodeName(), "input_greater1 and input_sub shapes cannot broadcast together"),
                return ge::GRAPH_FAILED);
    if (bcShape.GetDimNum() == 0) {
        bcShape.SetDimNum(1);
        bcShape.SetDim(0, 1);
    }
    for (size_t i = 0; i < OUT_NUM; i++) {
        auto out = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, out);
        *out = bcShape;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4LambUpdateWithLr(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "InferDataType4LambUpdateWithLr enter");
    // y 与 input_greater1 同类型
    context->SetOutputDataType(0, context->GetInputDataType(0));
    OP_LOGD(context->GetNodeName(), "InferDataType4LambUpdateWithLr end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LambUpdateWithLr)
    .InferShape(InferShape4LambUpdateWithLr)
    .InferDataType(InferDataType4LambUpdateWithLr);
} // namespace ops
