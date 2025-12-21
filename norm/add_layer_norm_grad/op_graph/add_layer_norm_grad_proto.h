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
 * \file add_layer_norm_grad_proto.h
 * \brief
 */
#ifndef OPS_NORM_ADD_LAYER_NORM_GRAD_PROTO_H_
#define OPS_NORM_ADD_LAYER_NORM_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief AddLayerNormGrad operator interface implementation.

* @par Inputs
* @li dy: Main grad input.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x1: Input x1 of the forward fused operator.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: Input x2 of the forward fused operator.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: Rstd of the sum of forward inputs x1 and x2.
*     A Tensor. Support dtype: [float32], support format: [ND].
* @li mean: Mean of the sum of forward inputs x1 and x2.
*     A Tensor. Support dtype: [float32], support format: [ND].
* @li gamma: Describing the weight.
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li dsum: Other grad input.
*      A optional input Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Outputs
* @li dx: The gradient of input "x", Has the same type and shape as "x".
*     A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li dgamma: The gradient of input "gamma", Has the same type and shape as "gamma".
*     A Tensor. Support dtype: [float32], support format: [ND].
* @li dbeta: The gradient of input "beta", Has the same type and shape as "beta".
*     A Tensor. Support dtype: [float32], support format: [ND].
*/
REG_OP(AddLayerNormGrad)
    .INPUT(dy, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(rstd, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .INPUT(mean, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(dsum, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(dx, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(dgamma, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(dbeta, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OP_END_FACTORY_REG(AddLayerNormGrad);

} // namespace ge

#endif // OPS_NORM_ADD_LAYER_NORM_GRAD_PROTO_H_