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
 * \file bn3_d_training_reduce_grad_infershape.cpp
 * \brief BN3DTrainingReduceGrad 的 shape 推导（InferShape）
 */

#include "register/op_impl_registry.h"             // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h" // InferShapeContext, gert::Shape
#include "graph/types.h"                           // ge::UNKNOWN_DIM_NUM
#include "op_common/log/log.h"                     // OP_CHECK_NULL_WITH_CONTEXT macro

using namespace ge;

namespace ops {
namespace {

constexpr size_t kGradsIndex = 0U;
constexpr size_t kXIndex = 1U;
constexpr size_t kParamBegin = 2U; // diff_scale / diff_offset / scale / batch_mean / batch_variance
constexpr size_t kParamNum = 5U;
constexpr size_t kRank5 = 5U;
constexpr size_t kChannelAxisNchw = 1U;  // NCDHW 通道轴 = dim1
constexpr size_t kChannelAxisNdhwc = 4U; // NDHWC 通道轴 = dim4

// V2 gert Shape 中 UNKNOWN_RANK 表示为 dim_num == 1 且 dim0 == UNKNOWN_DIM_NUM。
bool IsUnknownRank(const gert::Shape* shape)
{
    return shape->GetDimNum() == 1U && shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

// 维度值为负（-1 动态维 / -2 未知秩标记）视为未知，不参与数值比较。
bool IsKnownDim(const int64_t dim) { return dim >= 0; }

// 形状中任一维为 0 → 空 tensor。
bool HasZeroDim(const gert::Shape* shape)
{
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        if (shape->GetDim(i) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

static ge::graphStatus InferShape4BN3DTrainingReduceGrad(gert::InferShapeContext* context)
{
    // GetInputShape(0/1): grads 与 x 的 shape。
    const gert::Shape* grads_shape = context->GetInputShape(kGradsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, grads_shape);
    const gert::Shape* x_shape = context->GetInputShape(kXIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    // GetOutputShape(0): output y's shape descriptor.
    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // y.shape 依赖未知 rank 输入时传播 UNKNOWN_RANK（动态签名场景）。
    if (IsUnknownRank(grads_shape)) {
        *output_shape = *grads_shape;
        return ge::GRAPH_SUCCESS;
    }

    // 校验 1: rank 校验（grads / x 必须 5D）。
    if (grads_shape->GetDimNum() != kRank5 || (!IsUnknownRank(x_shape) && x_shape->GetDimNum() != kRank5)) {
        return ge::GRAPH_FAILED;
    }
    // 参数张量必须 1D（未知秩参数跳过，运行时由 tiling 复核）。
    for (size_t i = 0; i < kParamNum; ++i) {
        const gert::Shape* param_shape = context->GetInputShape(kParamBegin + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, param_shape);
        if (IsUnknownRank(param_shape)) {
            continue;
        }
        if (param_shape->GetDimNum() != 1U) {
            return ge::GRAPH_FAILED;
        }
    }

    // 校验 2: 空 tensor 校验（任一维为 0 → null_input）。
    if (HasZeroDim(grads_shape) || (!IsUnknownRank(x_shape) && HasZeroDim(x_shape))) {
        return ge::GRAPH_FAILED;
    }

    // 校验 3: grads 与 x 同 shape（未知维仅跳过比较）。
    if (!IsUnknownRank(x_shape)) {
        if (x_shape->GetDimNum() != grads_shape->GetDimNum()) {
            return ge::GRAPH_FAILED;
        }
        for (size_t i = 0; i < grads_shape->GetDimNum(); ++i) {
            if (IsKnownDim(grads_shape->GetDim(i)) && IsKnownDim(x_shape->GetDim(i)) &&
                grads_shape->GetDim(i) != x_shape->GetDim(i)) {
                return ge::GRAPH_FAILED;
            }
        }
    }

    // 校验 4: 参数长度 = C（通道轴 NCDHW dim1 / NDHWC dim4，见函数头说明）。
    const int64_t channel_nchw = grads_shape->GetDim(kChannelAxisNchw);
    const int64_t channel_ndhwc = grads_shape->GetDim(kChannelAxisNdhwc);
    const bool has_known_channel = IsKnownDim(channel_nchw) || IsKnownDim(channel_ndhwc);
    if (has_known_channel) {
        for (size_t i = 0; i < kParamNum; ++i) {
            const gert::Shape* param_shape = context->GetInputShape(kParamBegin + i);
            OP_CHECK_NULL_WITH_CONTEXT(context, param_shape);
            if (IsUnknownRank(param_shape)) {
                continue;
            }
            const int64_t param_len = param_shape->GetDim(0);
            if (!IsKnownDim(param_len)) {
                continue;
            }
            const bool matches_channel = (IsKnownDim(channel_nchw) && param_len == channel_nchw) ||
                                         (IsKnownDim(channel_ndhwc) && param_len == channel_ndhwc);
            if (!matches_channel) {
                return ge::GRAPH_FAILED;
            }
        }
    }

    // 输出推导: y.shape = grads.shape（5D 逐维复制）。
    *output_shape = *grads_shape;

    return ge::GRAPH_SUCCESS;
}

// IMPL_OP_INFERSHAPE(BN3DTrainingReduceGrad).InferShape(func):
//   Registers InferShape4BN3DTrainingReduceGrad as the shape inference function
//   for the BN3DTrainingReduceGrad operator type.
//   To create FooBar: change BN3DTrainingReduceGrad → FooBar.
IMPL_OP_INFERSHAPE(BN3DTrainingReduceGrad).InferShape(InferShape4BN3DTrainingReduceGrad);

} // namespace ops
