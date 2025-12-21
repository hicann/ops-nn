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
 * \file deep_norm_grad.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_DEEP_NORM_GRAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_DEEP_NORM_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief DeepNormGrad Operator.
 *
 * @par Inputs
 * @li dy: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li gx: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li mean: A Tensor. Support dtype: [float32], support format: [ND].
 * @li rstd: A Tensor. Support dtype: [float32], support format: [ND].

 * @par Attributes:
 * alpha: An optional attribute, the type is float. Defaults to 0.3.

 * @par Outputs
 * @li dx: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li dgx: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li dbeta: A Tensor. Support dtype: [float32], support format: [ND].
 * @li dgamma: A Tensor. Support dtype: [float32], support format: [ND].
 */
REG_OP(DeepNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType::({DT_FLOAT}))
    .INPUT(rstd, TensorType::({DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dgx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT}))
    .ATTR(alpha, Float, 0.3f)
    .OP_END_FACTORY_REG(DeepNormGrad);
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_DEEP_NORM_GRAD_OPS_H_