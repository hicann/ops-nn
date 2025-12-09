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
 * \file inplace_add_layer_norm_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_INPLACE_ADD_LAYER_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INPLACE_ADD_LAYER_NORM_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fused Operator of Add and LayerNorm.

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A optional input Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li mean: A Tensor. Support dtype: [float32], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(InplaceAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(mean, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(rstd, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(InplaceAddLayerNorm)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_INPLACE_ADD_LAYER_NORM_OPS_H_