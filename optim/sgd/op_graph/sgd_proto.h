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
 * \file sgd_proto.h
 * \brief
 */
#ifndef OPS_NN_OPTIM_SGD_PROTO_H
#define OPS_NN_OPTIM_SGD_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 *@brief Updates "parameters" according to the SGD algorithm with momentum. \n
 *  computing process:
 *@code{.c}
 *  // d = dampening, wd = weight_decay, lr = learning_rate[0], m = momentum[0]
 *  if (wd != 0) {
 *      grad = gradient + parameters * wd
 *  } else {
 *      grad = gradient
 *  }
 *  accum_t = accum * m + grad            // unconditional
 *  if (d != 0) {
 *      accum_t -= grad * (1 - stat) * d
 *  }
 *  if (nesterov) {
 *      parameters -= grad * lr + accum_t * m * lr
 *  } else {
 *      parameters -= accum_t * lr
 *  }
 *  if (m != 0) {                          // writeback mask
 *      accum = accum_t
 *      stat  = 0
 *  }                                      // otherwise accum and stat keep their input values
 *@endcode
 *
 *@par Inputs:
 *@li parameters: A mutable tensor of ND. Must be of dtype float16, float32 or bfloat16.
 *    Specifying parameters to be updated. Should be from a Variable().
 *@li gradient: A tensor of ND. Must be of the same shape and dtype as "parameters".
 *    Specifying the gradient.
 *@li learning_rate: A scalar. Must be of the same dtype as "parameters".
 *    Accepted as a 0-D tensor or any tensor holding exactly one element (e.g. [1], [1, 1]).
 *    Specifying the learning rate.
 *@li accum: A mutable tensor of ND. Must be of the same shape and dtype as "parameters".
 *    Specifying the momentum accumulation. Should be from a Variable().
 *@li momentum: A scalar. Must be of the same dtype as "parameters". Specifying the momentum.
 *    Accepted as a 0-D tensor or any tensor holding exactly one element (e.g. [1], [1, 1]).
 *@li stat: A mutable tensor of ND. Must be of the same shape and dtype as "parameters".
 *    Per-element first-step flag: a value of 1 means "first step" and suppresses the
 *    dampening correction for that element. Should be from a Variable().
 *
 *@par Attributes:
 *@li dampening: An optional float. Defaults to "0.0". Must be 0 when "nesterov" is true.
 *@li weight_decay: An optional float. Defaults to "0.0". Must be greater than or equal to 0.
 *@li nesterov: An optional bool. Defaults to "false". If "true", uses Nesterov momentum.
 *
 *@par Outputs:
 * parameters: A mutable tensor. Has the same shape, dtype and format as input "parameters".
 *
 *@attention Constraints:
 *@li Only one output is declared on the graph, but the operator updates THREE tensors
 *    in place: "parameters", "accum" and "stat". "accum" and "stat" are returned by
 *    overwriting their input memory and are therefore not visible as graph outputs.
 *    Callers must treat them as mutable. This mirrors the 910B/910C behaviour
 *    (the TBE implementation declares reuse=('accum', 'parameters', 'stat')).
 *@li When "momentum" is 0 (including -0.0), "accum" and "stat" are NOT written at all
 *    and keep their input values bit-for-bit; "parameters" is still updated as usual.
 *@li The rank of "parameters" must be in the range [1, 8]; rank-0 (scalar) is rejected.
 *    Empty tensors (any axis being 0) are rejected as well.
 *
 *@par Third-party framework compatibility
 * The writeback mask matches PyTorch's torch.optim.SGD, which skips the whole momentum
 * block when momentum == 0. Note however that PyTorch applies "dampening" INSIDE that
 * block while this operator (like the 910B/910C implementation) applies it OUTSIDE.
 * Consequently, when momentum == 0 && dampening > 0 && stat == 0, "parameters" differs
 * from PyTorch by a factor of (1 - dampening).
 */

#ifndef OPS_PROTO_DEF_SGD
#define OPS_PROTO_DEF_SGD
REG_OP(SGD)
    .INPUT(parameters, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gradient, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(learning_rate, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(accum, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(momentum, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(stat, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(parameters, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(dampening, Float, 0.0)
    .ATTR(weight_decay, Float, 0.0)
    .ATTR(nesterov, Bool, false)
    .OP_END_FACTORY_REG(SGD)
#endif // OPS_PROTO_DEF_SGD
} // namespace ge

#endif // OPS_NN_OPTIM_SGD_PROTO_H
