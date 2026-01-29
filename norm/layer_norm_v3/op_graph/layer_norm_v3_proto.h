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
 * \file layer_norm_v3_proto.h
 * \brief
 */
#ifndef OPS_NORM_LAYER_NORM_V3_H_
#define OPS_NORM_LAYER_NORM_V3_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief LayernormV3 operator interface implementation \n
*  calculating: x, gamma, beta \n
*  mean  = np.mean(x, reduce_axis, keepdims=True) \n
*  rstd = np.rsqrt(np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = gamma*((x - mean) * rstd) + beta \n
*  Notes: the shape of gamma/beta is the slice shape of x(x_shape[begin_norm_axis])

*@par Inputs:
*Three inputs, including:
* @li x: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* The shape is [A1,...,Ai,R1,...,Rj].
* @li gamma: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same dtype and shape as beta. The shape is [R1,...,Rj].
* @li beta: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same dtype and shape as gamma. The shape is [R1,...,Rj]. \n

*@par Attributes:
* @li begin_norm_axis: An optional attribute, the dtype is int32. Defaults to 0.
* Indicates the index of the R1 axis in the shape of x.
* @li begin_params_axis: An optional attribute, the dtype is int32. Defaults to 0.
* In Ascend 950 AI Processor, begin_params_axis and begin_norm_axis refer to the same axis in the shape of x.
* @li epsilon: An optional attribute, the dtype is float32. Defaults to 1e-5 . \n

*@par Outputs:
*Three outputs, including:
* @li y: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same dtype, shape and format as x.
* @li mean: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same shape as rstd, which is [A1,...,Ai,1,...,1], where there are j 1's after Ai.
* @li rstd: A ND Tensor. Must be one of the following dtypes: float16, float32, bfloat16.
* Has the same shape as mean, which is [A1,...,Ai,1,...,1], where there are j 1's after Ai.
*/
REG_OP(LayerNormV3)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(begin_norm_axis, Int, 0)
    .ATTR(begin_params_axis, Int, 0)
    .ATTR(epsilon, Float, 0.00001f)
    .OP_END_FACTORY_REG(LayerNormV3)
} // namespace ge
#endif // OPS_NORM_LAYER_NORM_V3_H_