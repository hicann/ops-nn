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
 * \file deep_norm_proto.h
 * \brief
 */
#ifndef OPS_NORM_DEEP_NORM_PROTO_H_
#define OPS_NORM_DEEP_NORM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief DeepNorm operator. \n
 *  calculating: x, gx, gamma, beta, alpha \n
 *  new_x = x * alpha + gx \n
 *  y = gamma*(new_x - mean) / np.sqrt(variance + 1e-6) + beta

 * @par Inputs
 * Four inputs, including:
 * @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li gx: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 * @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

 * @par Attributes
 * @li alpha: An optional attribute, the type is float. Defaults to 0.3.
 * @li eps: An optional attribute, the type is float. Defaults to 1e-06.

 * @par Outputs
 * Three outputs, including:
 * @li mean: A Tensor. Support dtype: [float32], support format: [ND].
 * @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
 * @li y: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
 */
REG_OP(DeepNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(alpha, Float, 0.3f)
    .ATTR(epsilon, Float, 1e-06f)
    .OP_END_FACTORY_REG(DeepNorm);
} // namespace ge

#endif // OPS_NORM_DEEP_NORM_PROTO_H_