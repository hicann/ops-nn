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
 * \file lamb_apply_optimizer_assign_infershape.cpp
 * \brief
 */

#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
// A2 语义: 本族算子的所有输入都是可广播的 ND Tensor(见 canndev
// ops/built-in/tbe/impl/lamb_*.py, 每一步 mul/sub/div 都先 shape_util.broadcast_shapes
// 再 tbe.broadcast), 输出形状为全部输入广播的结果。A2 的 op_proto 只声明了其中两个输入,
// 属声明宽松, 不作为支持面依据。
// inputv/inputm 同时是 ref 输出(原地更新的动量), 故广播结果必须恰好等于它们的形状,
// 否则原地写回会越过其显存边界。其余输入(含 In0 的 grad)可广播进这个形状。
constexpr size_t IN_NUM = 12;
constexpr size_t INPUTV_IDX = 1;
constexpr size_t INPUTM_IDX = 2;
constexpr size_t OUT_NUM = 3;

static ge::graphStatus InferShape4LambApplyOptimizerAssign(gert::InferShapeContext* context)
{
    std::vector<const gert::Shape*> inShapes;
    inShapes.reserve(IN_NUM);
    for (size_t i = 0; i < IN_NUM; i++) {
        auto in = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, in);
        inShapes.push_back(in);
    }
    gert::Shape bcShape;
    // 逐对折叠广播: 只用两参数重载。vector 重载虽在 op_common/op_host/infershape_broadcast_util.h
    // 以 Ops::Base 声明, 但 libops_base.so 只导出两参数版, vector 版的实现在 libop_common.so 的
    // 小写 ops 命名空间下 —— 声明与实现命名空间不一致, 用它会编译期通过、加载期
    // undefined symbol 而装不上包。
    bcShape = *inShapes[0];
    for (size_t i = 1; i < inShapes.size(); i++) {
        gert::Shape tmp;
        OP_CHECK_IF(!BroadcastShape(&bcShape, inShapes[i], &tmp),
                    OP_LOGE(context->GetNodeName(), "input shapes cannot broadcast together"), return ge::GRAPH_FAILED);
        bcShape = tmp;
    }
    // 标量归一: 全标量输入 broadcast 得 0 维空 shape (), 与 A2 的 shape_util.scalar2tensor_one
    // 对齐, 归一为 (1,)。否则动态 shape 编译期 DFX 生成会对空 shape 做 reduce 连乘(无初值)而报
    // TypeError 编译失败。
    if (bcShape.GetDimNum() == 0) {
        bcShape.SetDimNum(1);
        bcShape.SetDim(0, 1);
    }
    const gert::Shape& vShape = *inShapes[INPUTV_IDX];
    const gert::Shape& mShape = *inShapes[INPUTM_IDX];
    OP_CHECK_IF(
        !(vShape == mShape),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "inputv and inputm", (ToString(vShape) + " and " + ToString(mShape)).c_str(),
            "inputv and inputm are in-place updated moments and must have the same shape"),
        return ge::GRAPH_FAILED);
    gert::Shape normV = vShape;
    if (normV.GetDimNum() == 0) {
        normV.SetDimNum(1);
        normV.SetDim(0, 1);
    }
    OP_CHECK_IF(!(bcShape == normV),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "inputv", ToString(vShape).c_str(),
                    "inputv/inputm are in-place(ref) outputs, so the broadcast shape of all inputs must equal them"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < OUT_NUM; i++) {
        auto out = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, out);
        *out = normV;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4LambApplyOptimizerAssign(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "InferDataType4LambApplyOptimizerAssign enter");
    // output0 与 grad 同类型
    context->SetOutputDataType(0, context->GetInputDataType(0));
    // inputv 为同名 ref 输出
    context->SetOutputDataType(1, context->GetInputDataType(1));
    // inputm 为同名 ref 输出
    context->SetOutputDataType(2, context->GetInputDataType(2));
    OP_LOGD(context->GetNodeName(), "InferDataType4LambApplyOptimizerAssign end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LambApplyOptimizerAssign)
    .InferShape(InferShape4LambApplyOptimizerAssign)
    .InferDataType(InferDataType4LambApplyOptimizerAssign);
} // namespace ops
