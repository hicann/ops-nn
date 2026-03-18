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
 * \file nn_norm_ops.h
 * \brief
 */
#ifndef OPS_NORM_ADA_LAYER_NORM_GRAD_PROTO_H_
#define OPS_NORM_ADA_LAYER_NORM_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief AdaLayerNormGrad operator interface implementation \n
* @code{.c}
*  Forward: out = LayerNorm(x) * (1 + scale) + shift \n
*  Backward calculations: \n
*  z = (x - mean) * rstd \n
*  dy_g = dy * gamma * (1 + scale) \n
*  temp_1 = 1/N * ∑(dy * gamma * (1 + scale)) \n
*  temp_2 = 1/N * (x - mean) * rstd * ∑(dy * gamma * (1 + scale) * (x - mean) * rstd) \n
*  pd_x = (dy * gamma * (1 + scale) - (temp_1 + temp_2)) * rstd \n
*  pd_scale = ∑(dy * ((x - mean) * rstd * gamma + beta)) \n
*  pd_shift = ∑dy \n
*  pd_gamma = ∑dy * (1 + scale) * (x - mean) * rstd \n
*  pd_beta = ∑dy * (1 + scale)
* @endcode

* @par Inputs:
* Seven inputs, including:
* @li dy: A tensor. The gradient tensor that represents the reverse calculation. 
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* The shape is [B, S, H], where B supports 0-6 dimensions.
* @li x: A tensor. First input of forward propagation.
* Must be one of the following types: float16, float32, bfloat16.
* The shape is the same as dy, which is [B, S, H], where B supports 0-6 dimensions.
* @li rstd: A tensor. Third output of forward propagation, indicates the reciprocal of the standard deviation of x. 
* Must be one of the following types: float32. The format must be ND.
* Has the shape [B, S, 1], where the last dimension is fixed to 1.
* @li mean: A tensor. Second output of forward propagation, indicates the mean value of x. 
* Must be one of the following types: float32. The format must be ND.
* Has the shape [B, S, 1], where the last dimension is fixed to 1.
* @li scale: A tensor. Indicates the adaptive scale parameter. 
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* The shape is [B, H] or [B, 1, H].
* @li gamma: A tensor. Indicates the normalization weight parameter. 
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* The shape is [H].
* @li beta: A tensor. Indicates the normalization bias parameter. 
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* The shape is [H].

* @par Outputs:
* Five outputs, including:
* @li pd_x: A tensor. Indicates the gradient of input x.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type, shape and format as x.
* @li pd_scale: A tensor. Indicates the gradient of scale.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type, shape and format as scale.
* @li pd_shift: A tensor. Indicates the gradient of adaptive offset parameter.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type and format as scale. The shape is [B, H] or [B, 1, H], where B supports 0-6 dimensions.
* @li pd_gamma: A tensor. Indicates the gradient of gamma.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type, shape and format as gamma.
* @li pd_beta: A tensor. Indicates the gradient of beta.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type, shape and format as beta.
*/

REG_OP(AdaLayerNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_shift, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(AdaLayerNormGrad)
}  // namespace ge
#endif  // OPS_NORM_ADA_LAYER_NORM_GRAD_PROTO_H_