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
 * \file nn_norm_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM1_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM1_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief LayerNormGrad operator interface implementation \n
*  calculating: dy, x, variance, mean, gamma \n
*  rstd = 1.0 / sqrt(variance + eps) \n
*  pd_xl = data_dy*data_gamma \n
*  pd_var = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
*           np.power(rstd, 3)),
*           reduce_axis, keepdims=True) \n
*  pd_mean = np.sum(((-1.0)*pd_xl*rstd), reduce_axis, keepdims=True)
*            + pd_var*(1.0/m)
*            np.sum(((-2.0)*(data_x - data_mean)), reduce_axis, keepdims=True) \n
*  pd_x = pd_xl*rstd +
*         pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m) \n
*  pd_gamma = np.sum((data_dy*(data_x - data_mean)*rstd), param_axis, keepdims=True) \n
*  pd_beta = np.sum(data_dy, param_axis, keepdims=True)

*@par Inputs:
*Five inputs, including:
* @li dy: A tensor. The gradient tensor that represents the reverse calculation.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* The shape is equal to the shape of x, that is, [A1, ...,Ai,R1, ...,Rj].
* Has the same type and format as x.
* @li x: A tensor. First input of forward propagation. Must be one of the following types: float16, float32, bfloat16.
* The shape is equal to the shape of dy, that is, [A1, ...,Ai,R1, ...,Rj].
* Has the same type and format as dy.
* @li variance: A tensor. Third output of forward propagation, indicating the variance value of input.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same shape as mean, which is [A1,...,Ai,1,...,1], where there are j 1s after Ai,
* and j is the length of the axis that requires normalization.
* Has the same type and format as mean.
* @li mean: A tensor. Second output of forward propagation, indicating the mean value of input.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same shape as variance, which is [A1,...,Ai,1,...,1], where there are j 1s after Ai,
* and j is the length of the axis that requires normalization.
* Has the same type and format as variance.
* @li gamma: A tensor. Indicates the weight tensor.
* Must be one of the following types: float16, float32, bfloat16.
* The format must be ND. Has the same type as x. The shape is [R1,...,Rj].

*@par Outputs:
*Three outputs, including:
* @li pd_x: A tensor. Indicates the first output of the forward calculation, which is the output tensor of the x derivative.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type, shape and format as x.
* @li pd_gamma: A tensor. Indicates the output tensor of gamma derivative.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type, shape and format as gamma.
* @li pd_beta: A tensor. The bias tensor of forward output.
* Must be one of the following types: float16, float32, bfloat16. The format must be ND.
* Has the same type, shape and format as gamma.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LayerNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(LayerNormGrad)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM1_OPS_H_
