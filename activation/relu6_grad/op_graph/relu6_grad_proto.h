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
 * \file relu6_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_RELU6_GRAD_H_
#define OPS_OP_PROTO_INC_RELU6_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Gradient of ReLU6. Strict open-interval semantics:
*        dx[i] = gradients[i]  if  0 < features[i] < 6
*        dx[i] = 0              otherwise (including features[i] == 0 and 6)
*        Broadcasting is NOT supported: features must have the same shape as
*        gradients, or be a scalar ([1]). backprops always has the same shape
*        as gradients.

* @par Inputs:
* Two inputs, including:
* @li gradients: A ND tensor. Upstream gradient (dy). Must be one of the
*     following types: float16, float32, bfloat16.
* @li features:  A ND tensor. Forward input x of ReLU6 (NOT the forward
*     output). Must have the same dtype as gradients; shape must be the same
*     as gradients, or be a scalar ([1]). \n

* @par Outputs:
* backprops: A ND tensor. Has the same dtype as gradients; shape is the
*            same as gradients. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Relu6Grad and PyTorch
* hardtanh_backward(min=0, max=6) at boundary x in {0, 6} returning 0.
*/
REG_OP(Relu6Grad)
    .INPUT(gradients, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(backprops, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(Relu6Grad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_RELU6_GRAD_H_
