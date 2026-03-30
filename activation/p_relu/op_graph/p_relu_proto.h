/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file p_relu_proto.h
 * \brief
 */
#ifndef P_RELU_PROTO_H_
#define P_RELU_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs parametric ReLU .

* @par Inputs:
* Two inputs, including:
* @li x: A multi-dimensional Tensor of type bfloat16, float16 or float32.
* @li weight: A Scalar or 1D Tensor of type bfloat16, float16 or float32, specifying the weight,
* initial value of "a". The number of dimensions must be the same as the number of channels . \n

* @par Outputs:
* y: An activated Tensor. Has the same dimensions with "x" . \n

* @par Third-party framework compatibility
* Compatible with PyTorch and Caffe operator PReLU.
*/
REG_OP(PRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(PRelu)

} // namespace ge
#endif // P_RELU_PROTO_H_