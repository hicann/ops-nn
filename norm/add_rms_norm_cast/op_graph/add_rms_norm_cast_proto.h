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
 * \file add_rms_norm_cast_proto.h
 * \brief
 */
#ifndef OPS_NORM_ADD_RMS_NORM_CAST_PROTO_H_
#define OPS_NORM_ADD_RMS_NORM_CAST_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief AddRmsNormCast operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  y2 = gamma * (x * rstd) \n
*  y1 = cast16232(y2) \n

* @par Inputs
* Three inputs, including:
* @li x1: A tensor. Support dtype: float16/bfloat16, support format: ND.
* @li x2: A tensor. Support dtype: float16/bfloat16, support format: ND.
* @li gamma: A tensor. Support dtype: float16/bfloat16, support format: ND.

* @par Attributes
* epsilon: Input eps in the formula, which is used to prevent division-by-zero errors.
* An optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Four outputs, including:
* @li y1: A tensor. Support dtype: float32, support format: ND.
* @li y2: A tensor. Support dtype: float16/bfloat16, support format: ND.
* @li rstd: A tensor. Describing the reciprocal of (x1 + x2)'s standard deviation.
*           Support dtype: float32, support format: ND.
* @li x: A tensor. Support dtype: float16/bfloat16, support format: ND.
*/
REG_OP(AddRmsNormCast)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(AddRmsNormCast)
} // namespace ge

#endif // OPS_NORM_ADD_RMS_NORM_CAST_PROTO_H_