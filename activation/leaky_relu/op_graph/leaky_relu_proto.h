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
 * \file leaky_relu_proto.h
 * \brief
 */
#ifndef LEAKY_RELU_PROTO_H_
#define LEAKY_RELU_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Computes the output as x if x > 0 and negative_slope * x if x <= 0 .

* @par Inputs:
* One input:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bfloat16, float32, float16, double.
*
* @par Attributes:
* negative_slope: A float32. Defaults to "0.0".
*
* @par Outputs:
* y: A Tensor. Has the same type, format and shape as "x".
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator LeakyRelu.
*/
REG_OP(LeakyRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .ATTR(negative_slope, Float, 0.0)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(LeakyRelu)

}

#endif // LEAKY_RELU_PROTO_H_