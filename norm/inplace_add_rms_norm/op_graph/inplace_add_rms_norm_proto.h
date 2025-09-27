/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_add_rms_norm_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_INPLACE_ADD_RMS_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INPLACE_ADD_RMS_NORM_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief InplaceAddRmsNorm operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x2 = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  x1 = gamma * (x2 * rstd)

* @par Inputs
* Three inputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes:
* epsilon: A optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Three outputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(InplaceAddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(InplaceAddRmsNorm)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_INPLACE_ADD_RMS_NORM_OPS_H_