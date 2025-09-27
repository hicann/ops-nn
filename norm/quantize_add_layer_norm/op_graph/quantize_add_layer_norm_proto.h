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
 * \file rms_norm_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_ADD_LAYER_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_ADD_LAYER_NORM_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief QuantizeAddLayerNorm operator interface implementation.

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li scales: A Tensor. Support dtype: [float32, bfloat16], support format: [ND].
* @li zero_points: A optional Tensor. Support dtype: [float32, bfloat16], support format: [ND].

* @par Attributes
* @li dtype: A required attribute, the type is int. No defaults value.
* @li axis: A optional attribute, the type is int. Defaults to -1.
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li y: A Tensor. Support dtype: [int8], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(QuantizeAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(bias, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(QuantizeAddLayerNorm)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_QUANTIZE_ADD_LAYER_NORM_OPS_H_