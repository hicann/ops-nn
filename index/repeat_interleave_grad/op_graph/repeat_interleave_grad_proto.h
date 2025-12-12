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
 * \file repeat_interleave_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_REPEAT_INTERLEAVE_GRAD_OPS_H_
#define OPS_OP_PROTO_INC_REPEAT_INTERLEAVE_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Gradient op for RepeatInterleave op.
* @par Inputs:
* Two inputs:
* @li y_grad: A Tensor with any format. Support float, float16, bf16.
* @li repeats: A Tensor with dim = 1 or a Scalar. Support int32 or int64.

* @par Attributes:
* axis: An optional int32, specifying the axis to repeat. Defaults to -1.

* @par Outputs:
* x_grad: A Tensor, which is the same dtype as y_grad. Support float, float16, bf16.

* @attention Constraints:
* @li "axis" must be within the rank of the input tensor.
* @li The elements in "repeats" cannot all be zero.

* @par Third-party framework compatibility
* Compatible with the PyTorch operator the grad of RepeatInterleave.
*/
REG_OP(RepeatInterleaveGrad)
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(repeats, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(axis, Int, -1)
    .OP_END_FACTORY_REG(RepeatInterleaveGrad)

} // namespace ge

#endif