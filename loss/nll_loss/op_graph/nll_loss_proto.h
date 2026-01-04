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
 * \file nll_loss_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_NLL_LOSS_OPS_H_
#define OPS_OP_PROTO_INC_NLL_LOSS_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief The negative log likelihood loss .

* @par Inputs:
* The input x and weight must have the same type. Inputs include:
* @li x: A 2D or 4D tensor with shape (N, C) or (N, C, H, W). Support dtype: float32/bfloat16/float16.
* @li target: A 1D or 3D tensor with shape (N) or (N, H, W). When x is 2D(shape (N, C)),target should be 1D(shape(N)) or
scalar, when x is 4D(shape(N, C, H, W)), target should be 3d(shape(N, H, W)). Indicating the real label. Support dtype:
int32/int64/uint8.
* @li weight: A 1D tensor with shape (C) or none. Indicating the scale weight of each class. Support
dtype:float32/bfloat16/float16. \n

* @par Attributes:
* @li reduction: An optional attribute. Specifies the reduction to be applied to the output.
*                Type is string. Defaults to "mean" .
* @li ignore_index: An optional attribute. Specifying a target that is ignored and does not affect the input gradient.
*                   Type is int. Defaults to -100 . \n

* @par Outputs:
* @li y: if reduction is "none", a 1D or 3D tensor with shape (N) or (N, H, W). When x is 2D(shape (N, C)), y should be
1D(shape(N)) or scalar, when x is 4D(shape(N, C, H, W)), y should be 3d(shape(N, H, W)). Otherwise, shape is (1).
Support dtype: float32/bfloat16/float16.
* @li total_weight: A 1D tensor with shape (1). Support dtype: float32/bfloat16/float16. \n

* @par Third-party framework compatibility
* Compatible with pytorch NLLLoss operator
*/
REG_OP(NLLLoss)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(target, TensorType({DT_INT32, DT_INT64, DT_UINT8}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(total_weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
    .OP_END_FACTORY_REG(NLLLoss)
} // namespace ge

#endif // OPS_OP_PROTO_INC_NLL_LOSS_OPS_H_
