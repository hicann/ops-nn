/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ctc_loss_v2_proto.h
 * \brief ctc_loss_v2_proto
 */

#ifndef OPS_LOSS_CTC_LOSS_V2_OP_GRAPH_CTC_LOSS_V2_PROTO_H_
#define OPS_LOSS_CTC_LOSS_V2_OP_GRAPH_CTC_LOSS_V2_PROTO_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief The Connectionist Temporal Classification loss.

*@par Inputs:
*@li log_probs: A tensor. Must be one of the following types: float16, bfloat16, float32, double. Tensor of size (T, N, C), where T =input length, N =batch size,
                and C = number of classes (including blank).
                It represent the logarithmized probabilities of the outputs.
*@li targets: A tensor. Must be one of the following types: int32, int64. Tensor of size (N, S) or sum(target_lengths), where S = max target length.
             It represent the target sequences.
*@li input_lengths: A tensor. Must be one of the following types: int32, int64. Tuple or tensor of size (N). It represent the lengths of the inputs.
*@li target_lengths: A tensor. Must be one of the following types: int32, int64. Tuple or tensor of size (N). It represent lengths of the targets.

*@par Outputs:
*@li neg_log_likelihood: A tensor. Has the same dtype as log_probs, tuple or tensor of size (N). A loss value which is differentiable with respect to each input node.
*@li log_alpha: A tensor. Has the same dtype as log_probs, tensor of size (N, T, X), where X = 2 * max(target_lengths) + 1.
                The probability of possible trace of input to target.

*@par Attributes:
*@li blank: An optional Int. Blank label. Default 0.
*@li reduction: An optional String. Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
*@li zero_infinity: An optional Bool. Whether to zero infinite losses and the associated gradients. Default: false.

* @par Third-party framework compatibility:
* Compatible with Pytorch CTCLoss operator.

*@attention Constraints:
* The limit of Label’s length is 1K.
*/
REG_OP(CTCLossV2)
    .INPUT(log_probs, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(targets, TensorType({DT_INT32, DT_INT64}))
    .INPUT(input_lengths, TensorType({DT_INT32, DT_INT64}))
    .INPUT(target_lengths, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(neg_log_likelihood, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(log_alpha, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(blank, Int, 0)
    .ATTR(reduction, String, "mean")
    .ATTR(zero_infinity, Bool, false)
    .OP_END_FACTORY_REG(CTCLossV2)

} // namespace ge

#endif  // OPS_LOSS_CTC_LOSS_V2_OP_GRAPH_CTC_LOSS_V2_PROTO_H_