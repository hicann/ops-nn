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
#ifndef OPS_BUILT_IN_OP_PROTO_INC_GEMMA_RMS_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_GEMMA_RMS_NORM_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief GemmaRmsNorm operator interface implementation. \n
*  calculating: x, gamma \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = (1 + gamma) * (x * rstd)

* @par Inputs
* Two inputs, including:
* @li x: A tensor of input. Represens the input data to be normalized. The format must be ND. Shape support 1D ~ 8D.
* Must be one of the following types: float16, float32, bfloat16.
* @li gamma: A tensor of input. Represents a learnable scaling factor.
* Must be one of the following types: float16, float32, bfloat16. Must have the same type as "x".
* The format must be ND. Shape support 1D ~ 8D.
* The shape must meet the requirements of gamma_shape = x_shape[n:], n < x_shape.dims().

* @par Attributes
* epsilon: Optional attribute to prevent division by 0. The type is float32. Defaults to 1e-6.

* @par Outputs
* Two outputs, including:
* @li y: A tensor of output. Indicates normalized data. The format must be ND. Shape support 1D ~ 8D.
* Must be one of the following types: float16, float32, bfloat16. Must have the same type, format and shape as "x".
* @li rstd: A tensor of output. Represents root mean square. The format must be ND. The type is float32.
* The shape is the same as the first several dimensions of the "x" shape.
* The first several dimensions refer to the dimension of "x" minus the dimension of "gamma",
* indicating that the norm is not required.
*/
REG_OP(GemmaRmsNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(GemmaRmsNorm)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_GEMMA_RMS_NORM_OPS_H_