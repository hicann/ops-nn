/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// norm/l2_normalize/op_host/l2_normalize_infershape.cpp
// =============================================================================
//
// ROLE: V2 shape inference (gert) for the L2Normalize operator.
//   动态图执行期由 gert executor 直接调用（编译期图推导走 op_graph/ 的 V1 注册，
//   两条路径规则一致）。
//   - y.shape = x.shape 恒等复制（动态 -1 维、空 0 维一并透传，纯结构级映射，
//     不读属性）；
//   - rank-0 标量拒绝（契约 rank 1–8）：GE 图管线会把 rank-0 规整为 (1) 后才到
//     tiling，动态执行期须在此拦截（GEIR L2 负向 dimval 场景）。
//   - dtype/format/属性值域校验不在本函数职责内（OpDef TensorType / V1 推导 /
//     tiling 硬门禁承载）。
//
// CONTENTS:
//   - InferShape4L2Normalize() — rank-0 gate + identity shape copy
//   - IMPL_OP_INFERSHAPE(L2Normalize).InferShape(...) — registration macro
//
// =============================================================================

#include "register/op_impl_registry.h"             // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h" // InferShapeContext, gert::Shape
#include "op_common/log/log.h"                     // OP_CHECK_NULL_WITH_CONTEXT macro
#include "util/shape_util.h"                       // IsUnknownRank, SetUnknownRank
#include <string>

using namespace ge;

namespace ops {

static constexpr size_t INPUT_X_IDX = 0;
static constexpr size_t OUTPUT_Y_IDX = 0;
static constexpr size_t MAX_INPUT_RANK = 8;

// ---------------------------------------------------------------------------
// InferShape4L2Normalize(context) — shape inference function
//
// rank-0 scalar rejection + identity copy: output shape = input shape.
//
// Parameters:
//   context — InferShapeContext providing access to input shapes and
//             allowing setting of output shapes
//
// Returns:
//   ge::graphStatus — ge::GRAPH_SUCCESS on success
// ---------------------------------------------------------------------------
static ge::graphStatus InferShape4L2Normalize(gert::InferShapeContext* context)
{
    // GetInputShape(0): reads the shape of the first input (index 0 = x)
    const gert::Shape* input_shape = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    // GetOutputShape(0): gets a mutable reference to output y's shape descriptor
    gert::Shape* output_shape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // 动态 rank 使用框架标准标记透传，避免把 {-2} 当成普通的一维 shape 参与后续 rank 校验。
    if (Ops::Base::IsUnknownRank(*input_shape)) {
        Ops::Base::SetUnknownRank(*output_shape);
        return ge::GRAPH_SUCCESS;
    }

    // rank-0 标量拒绝（契约 rank 1–8；rank-0 无合法 axis，且图管线规整后 tiling
    // 不可见，须在推导期拦截）
    const size_t rank = input_shape->GetDimNum();
    OP_CHECK_IF(input_shape->IsScalar() || rank > MAX_INPUT_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "rank", std::to_string(rank).c_str(),
                                                         "x rank must be in [1, 8] (rank-0 scalar is rejected)"),
                return ge::GRAPH_FAILED);

    // Copy: output shape = input shape（动态 -1 维、空 0 维一并透传）
    *output_shape = *input_shape;

    return ge::GRAPH_SUCCESS;
}

// IMPL_OP_INFERSHAPE(L2Normalize).InferShape(func):
//   Registers InferShape4L2Normalize as the shape inference function
//   for the L2Normalize operator type.
IMPL_OP_INFERSHAPE(L2Normalize).InferShape(InferShape4L2Normalize);

} // namespace ops
