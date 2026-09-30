/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// bn3d_training_update_grad_package/op_host/bn3d_training_update_grad_infershape.cpp
// =============================================================================
//
// ROLE: Shape inference for the BN3DTrainingUpdateGrad operator.
//   When the Graph Engine compiles a graph (static or dynamic shape), it calls
//   this InferShape function to determine the output shapes from the inputs.
//
//   BN3D training-backward parameter-gradient reduction keeps the Channel axis
//   and reduces Batch + all spatial dims (N,D,H,W). Both outputs are therefore
//   per-channel vectors whose shape follows the channel statistic batch_mean,
//   NOT the reduced grads/x:
//       diff_scale.shape  = batch_mean.shape   (input index 2)
//       diff_offset.shape = batch_mean.shape
//   This is a pure shape-follow: it does not depend on any input value nor on
//   the epsilon attribute, and it covers static, dynamic (-1) and unknown-rank
//   (-2) shapes because gert::Shape assignment copies those sentinels through.
//
// CONTENTS:
//   - InferShapeBN3DTrainingUpdateGrad() — copies batch_mean shape to both outputs
//   - IMPL_OP_INFERSHAPE(BN3DTrainingUpdateGrad).InferShape(...) — registration
//
// =============================================================================

#include "register/op_impl_registry.h"             // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h" // InferShapeContext, gert::Shape
#include "op_common/log/log.h"                     // OP_CHECK_NULL_WITH_CONTEXT macro

using namespace ge;

namespace ops {

// Input / output index constants.
constexpr size_t INPUT_BATCH_MEAN_INDEX = 2;
constexpr size_t OUTPUT_DIFF_SCALE_INDEX = 0;
constexpr size_t OUTPUT_DIFF_OFFSET_INDEX = 1;

// ---------------------------------------------------------------------------
// InferShapeBN3DTrainingUpdateGrad(context) — shape inference function
//
// Both outputs (diff_scale idx0, diff_offset idx1) take the shape of the
// channel statistic batch_mean (input idx2). The reduced grads/x spatial dims
// only affect the per-channel reduction count M inside the kernel, never the
// output shape. Copying batch_mean's gert::Shape also propagates dynamic (-1)
// dims and the unknown-rank (-2) sentinel, so static / dynamic / unknown-rank
// are all handled by the single assignment.
// ---------------------------------------------------------------------------
static ge::graphStatus InferShapeBN3DTrainingUpdateGrad(gert::InferShapeContext* context)
{
    // Read the channel statistic (batch_mean) shape — the sole source of the
    // two output shapes.
    const gert::Shape* batch_mean_shape = context->GetInputShape(INPUT_BATCH_MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, batch_mean_shape);

    gert::Shape* diff_scale_shape = context->GetOutputShape(OUTPUT_DIFF_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, diff_scale_shape);
    gert::Shape* diff_offset_shape = context->GetOutputShape(OUTPUT_DIFF_OFFSET_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, diff_offset_shape);

    // diff_scale.shape = diff_offset.shape = batch_mean.shape (shape-follow).
    *diff_scale_shape = *batch_mean_shape;
    *diff_offset_shape = *batch_mean_shape;

    return ge::GRAPH_SUCCESS;
}

// IMPL_OP_INFERSHAPE(BN3DTrainingUpdateGrad).InferShape(func):
//   Registers InferShapeBN3DTrainingUpdateGrad as the shape inference function
//   for the BN3DTrainingUpdateGrad operator type.
IMPL_OP_INFERSHAPE(BN3DTrainingUpdateGrad).InferShape(InferShapeBN3DTrainingUpdateGrad);

} // namespace ops
