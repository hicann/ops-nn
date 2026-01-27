/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!g
 * \file gelu_grad_proto.h
 * \brief
 */
#ifndef GELU_GRAD_PROTO_H_
#define GELU_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Computes the gradient for the gelu of "x" .

* @par Inputs:
* Three inputs, including:
* @li dy: A Tensor. Support 1D ~ 8D. Must be one of the following types:bfloat16, float16, float32.
* @li x: A Tensor of the same type, shape and format as "dy".
* @li y: A Tensor of the same type, shape and format as "dy" . \n

* @par Outputs:
* z: A Tensor. Has the same type, format and shape as "dy".
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator GeluGrad.
* @attention Constraints:
* In Ascend 910_95 AI Processor, inputs and outputs support broadcasting.
*/
REG_OP(GeluGrad)
    .INPUT(dy, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(GeluGrad)

} // namespace ge

#endif