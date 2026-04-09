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
 * \file hard_sigmoid_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_HARD_SIGMOID_GRAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_HARD_SIGMOID_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes the gradient of the hard sigmoid activation function.
 *
 * @par Inputs:
 * @li grad_output: A Tensor. Support 1D~8D. Must be one of the following types:
 * float32, float16, bfloat16.
 * The upstream gradient tensor.
 * @li self: A Tensor. Has the same type, format and shape as "grad_output".
 * The input tensor of the corresponding HardSigmoid forward operation.
 *
 * @par Attributes:
 * @li alpha: An optional float. Slope of the linear region. Defaults to 1/6 (0.16666666).
 * @li beta: An optional float. Offset of the linear region. Defaults to 0.5.
 *
 * @par Outputs:
 * grad_input: A Tensor. Has the same type, format and shape as "grad_output".
 * The gradient with respect to the input of HardSigmoid.
 * Computed as: grad_input = grad_output * alpha  if 0 < alpha * self + beta < 1
 *                           0                    otherwise
 *
 * @par Third-party framework compatibility
 * Compatible with the PyTorch operator hardsigmoid_backward.
 */
REG_OP(HardSigmoidGrad)
    .INPUT(grad_output, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(self, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_input, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(alpha, Float, 0.16666666f)
    .ATTR(beta, Float, 0.5f)
    .OP_END_FACTORY_REG(HardSigmoidGrad)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_HARD_SIGMOID_GRAD_OPS_H_
