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
 * \file poisson_nll_loss_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_POISSON_NLL_LOSS_H_
#define OPS_OP_PROTO_INC_POISSON_NLL_LOSS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Computes Poisson Negative Log Likelihood Loss.
 *
 * @par Inputs:
 * Two inputs, including:
 * @li input: A Tensor of predicted values. Must be one of the following types: float16, float32.
 * @li target: A Tensor of target values. Must be one of the following types: float16, float32.
 *             Must have the same shape as input or be broadcastable.
 *
 * @par Attributes:
 * @li log_input: An optional bool. If True, the loss is computed as exp(input) - target * input.
 *                If False, as input - target * log(input + eps). Defaults to True.
 * @li full: An optional bool. If True, adds Stirling approximation term. Defaults to False.
 * @li eps: An optional float. Small value to avoid log(0). Defaults to 1e-8.
 * @li reduction: An optional string from: "none", "mean", "sum". Specifies the reduction to apply.
 *                Defaults to "mean".
 *
 * @par Outputs:
 * @li output: A Tensor. Has the same type as input.
 *             Shape depends on reduction: (1,) for "mean"/"sum", same as input for "none".
 *
 * @par Third-party framework compatibility
 * Compatible with PyTorch operator PoissonNLLLoss.
 */
#ifndef OPS_PROTO_DEF_POISSONNLLLOSS
#define OPS_PROTO_DEF_POISSONNLLLOSS
REG_OP(PoissonNllLoss)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(log_input, Bool, true)
    .ATTR(full, Bool, false)
    .ATTR(eps, Float, 1e-8)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(PoissonNllLoss)
#endif // OPS_PROTO_DEF_POISSONNLLLOSS

} // namespace ge

#endif // OPS_OP_PROTO_INC_POISSON_NLL_LOSS_H_
