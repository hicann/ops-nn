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
 * \file cross_entropy_sum_exp_and_index_logit_proto.h
 * \brief
 */

#ifndef OPS_LOSS_CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_PROTO_H_
#define OPS_LOSS_CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fused local computation stage of the vocab-parallel cross entropy loss, covering the segment between
* all_reduce(MAX) and all_reduce(SUM). For the local vocab shard it computes the shifted exponent
* exp(logits - global_max), its reduction along the last axis, the logit selected by target, and the local
* target offset with its out-of-range mask.

* @par Inputs:
* Three inputs, including:
* @li vocab_parallel_logits: A ND tensor of type float32 or bfloat16, the local vocab shard logits of the current
* tensor-parallel rank. Shape only supports 2D [N, V_local] or 3D [S, B, V_local], where the last axis is the local
* vocab size. Non-contiguous input is supported.
* @li target: A ND tensor of type int32, specifying the global vocab index.
* Shape is the same as vocab_parallel_logits.shape[:-1]. Non-contiguous input is supported.
* @li global_logits_max: A ND tensor of type float32 or bfloat16, the global maximum logit after all_reduce(MAX).
* Shape is the same as target, and dtype is the same as vocab_parallel_logits. Non-contiguous input is supported.

* @par Attributes:
* Two attributes, including:
* @li vocab_start_index: A required int. The start index of the vocab shard held by the current rank.
* @li vocab_end_index: A required int. The end index of the vocab shard held by the current rank. It must satisfy
* vocab_end_index > vocab_start_index and vocab_end_index - vocab_start_index == V_local.

* @par Outputs:
* Five outputs, including:
* @li predicted_logits: A ND tensor of type float32, the logit selected by target, shape is the same as target.
* The position is filled with 0 when target is not held by the current rank.
* @li sum_exp_logits: A ND tensor of type float32, the sum of exp(logits - global_logits_max) along the last axis,
* shape is the same as target.
* @li exp_logits: A ND tensor of type float32, exp(logits - global_logits_max),
* shape is the same as vocab_parallel_logits.
* @li target_offset: A ND tensor of type int32, target - vocab_start_index, shape is the same as target.
* The position is filled with 0 when target is not held by the current rank.
* @li target_mask: A ND tensor of type int32, shape is the same as target. The value 1 means target is not held by
* the current rank, and 0 means it is held.

* @attention Constraints:
* @li N, which equals prod(target.shape), is in the range [1, 32768]. V_local is in the range [16, 204800].
* @li V_local must be a multiple of 16 when vocab_parallel_logits is bfloat16, and a multiple of 8 when it is float32,
* so that the 32-byte alignment of the unified buffer is kept.
* @li target must be a non-negative global vocab index. A target outside [vocab_start_index, vocab_end_index) is
* masked, so target_mask is 1 and both predicted_logits and target_offset are 0 at that position.
* @li Intermediate computation is always performed in float32, so a bfloat16 input is promoted automatically.
* @li This operator is used for training only.
*/
REG_OP(CrossEntropySumExpAndIndexLogit)
    .INPUT(vocab_parallel_logits, TensorType({DT_FLOAT, DT_BF16}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(global_logits_max, TensorType({DT_FLOAT, DT_BF16}))
    .OUTPUT(predicted_logits, TensorType({DT_FLOAT}))
    .OUTPUT(sum_exp_logits, TensorType({DT_FLOAT}))
    .OUTPUT(exp_logits, TensorType({DT_FLOAT}))
    .OUTPUT(target_offset, TensorType({DT_INT32}))
    .OUTPUT(target_mask, TensorType({DT_INT32}))
    .REQUIRED_ATTR(vocab_start_index, Int)
    .REQUIRED_ATTR(vocab_end_index, Int)
    .OP_END_FACTORY_REG(CrossEntropySumExpAndIndexLogit)

} // namespace ge

#endif // OPS_LOSS_CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_PROTO_H_
