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
 * \file leaky_relu_grad_proto.h
 * \brief
 */
#ifndef LEAKY_RELU_GRAD_PROTO_H_
#define LEAKY_RELU_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Computes the output as gradients if features > 0 and negative_slope * gradients if features <= 0. \n
* Gradients/features support broadcasting operations. \n

* @par Inputs:
* Two inputs, including:
* @li gradients: A tensor. Must be one of the following types: bfloat16, float16, float32, double. The backpropagated gradients to the corresponding LeakyRelu operation.
* @li features: A tensor. Has the same type as "gradients". The features passed as input to the corresponding LeakyRelu operation. \n

* @par Attributes:
* negative_slope: An optional float32. Negative Slope. Defaults to "0.0". \n

* @par Outputs:
* backprops: A tensor. Has the same type as "gradients". \n

* @attention Constraints:
* The corresponding LeakyRelu operator needs to be called before using this operator on the network. \n

* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator LeakyReluGrad.
* @li Compatible with the Pytorch operator leaky_relu_backward.
*/
REG_OP(LeakyReluGrad)
    .INPUT(gradients, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .ATTR(negative_slope, Float, 0.0)
    .OUTPUT(backprops, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(LeakyReluGrad)

}

#endif // LEAKY_RELU_GRAD_PROTO_H_