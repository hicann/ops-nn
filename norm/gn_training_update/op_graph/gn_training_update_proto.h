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
// gn_training_update_package/op_graph/gn_training_update_proto.h
// =============================================================================
//
// ROLE: Graph IR operator prototype registration for GNTrainingUpdate.
//   This file defines the GNTrainingUpdate operator's interface for the CANN
//   Graph Engine (GE) IR layer, verbatim from the official IR
//   (canndev ops/built-in/op_proto/inc/reduce_ops.h REG_OP(GNTrainingUpdate)).
//   It registers:
//   - Input tensors: x (required), sum, square_sum (required),
//     scale/offset/mean/variance (optional)
//   - Output tensors: y, batch_mean, batch_variance
//   - Attributes: num_groups (int, default 2), epsilon (float, default 0.0001)
//
//   The REG_OP macro defines the operator's schema for graph-level
//   compilation. This is used by GE for:
//   - Type checking at graph construction time
//   - Shape inference (calls infershape registered in gn_training_update_infershape.cpp)
//   - Graph optimization passes
//
//   The TensorType({...}) lists specify which data types are allowed for
//   each tensor. These are checked at graph construction time.
//
// OPERATOR NAME VARIANTS:
//   official IR  : GNTrainingUpdate   — REG_OP macro argument, OP_END_FACTORY_REG
//                  (verbatim from canndev reduce_ops.h; note the all-caps GN,
//                  matching the official graph IR op type)
//   snake_case   : gn_training_update  — filename
//   UPPER_SNAKE  : GN_TRAINING_UPDATE  — header guard (used in this file)
//
// KEY MACRO: REG_OP(OpType)
//   Begins an operator registration block. Chained calls define inputs,
//   outputs, and attributes. OP_END_FACTORY_REG(OpType) closes the block.
//
// =============================================================================

#ifndef GN_TRAINING_UPDATE_PROTO_H
#define GN_TRAINING_UPDATE_PROTO_H

#include "graph/operator_reg.h" // REG_OP, OP_END_FACTORY_REG, TensorType macros

namespace ge {

// ---------------------------------------------------------------------------
// REG_OP(GNTrainingUpdate) — operator proto registration block
//
// Verbatim from the official IR (canndev ops/built-in/op_proto/inc/reduce_ops.h):
//
// INPUT(x): feature map to normalize. Supported dtypes: FP16, FP32.
// INPUT(sum): per-group element sum from GNTrainingReduce. dtype: FP32.
// INPUT(square_sum): per-group squared sum from GNTrainingReduce. dtype: FP32.
// OPTIONAL_INPUT(scale): affine gamma. dtype: FP32.
// OPTIONAL_INPUT(offset): affine beta. dtype: FP32.
// OPTIONAL_INPUT(mean): reserved, not used in computation. dtype: FP32.
// OPTIONAL_INPUT(variance): reserved, not used in computation. dtype: FP32.
// ATTR(num_groups, Int, 2): number of groups G, must divide C.
// ATTR(epsilon, Float, 0.0001): small value added to variance before sqrt.
// OUTPUT(y): normalized (and optionally affine-transformed) result, same
//   shape/dtype as x.
// OUTPUT(batch_mean): per-group mean (sum/M), dtype FP32.
// OUTPUT(batch_variance): per-group biased variance, dtype FP32.
// ---------------------------------------------------------------------------
REG_OP(GNTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(num_groups, Int, 2)
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(GNTrainingUpdate)
} // namespace ge

#endif
