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
 * \file adaptive_avg_pool2d_grad_proto.h
 * \brief Definition of AdaptiveAvgPool2dGrad operator registration proto.
 */
#ifndef OPS_POOLING_ADAPTIVE_AVG_POOL2D_GRAD_PROTO_H_
#define OPS_POOLING_ADAPTIVE_AVG_POOL2D_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Compute gradients of adaptive average pooling function.

* @par Inputs:
* @li input_grad: A Tensor. Must be one of the following data types:
* float16, float32, bfloat16. Supports NCHW and NCL formats.

* @par Attributes:
* @li orig_input_shape: A required tuple or list of type int64.

* @par Outputs:
* @li output_grad: A tensor with the same type and format as "input_grad".

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveAvgPool2dGrad.
*/
REG_OP(AdaptiveAvgPool2dGrad)
    .INPUT(input_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(output_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(orig_input_shape, ListInt)
    .OP_END_FACTORY_REG(AdaptiveAvgPool2dGrad)

} // namespace ge
#endif // OPS_POOLING_ADAPTIVE_AVG_POOL2D_GRAD_PROTO_H_
