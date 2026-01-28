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
 * \file math_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Computes Lp norm.

* @par Inputs:
* x: A ND tensor of dtype float16, bfloat16, float32.
*
* @par Attributes:
* @li p: An optional float, "inf" or "-inf", indicates the order of norm. Default is 2.0.
* @li axes: ListInt, an optional attribute, indicates dimensions over which to compute the norm.
* Default is {}, meaning all axes will be computed.
* @li keepdim: An optional bool. If set to true, the reduced dimensions are retained in the result
* as dimensions with size one. Default is false.
* @li epsilon: An optional float. A value added to the denominator for numerical stability. Default is 1e-12.

* @par Outputs:
* y: A ND tensor has the same dtype as "x". The shape of "y" is depending on "axes" and "keepdim".

* @attention Constraints:
* @li When the attribute "p" is negative, there may be precision difference in the calculation results.
* @li When the attribute "axes" is specified as the axis with a shape dimension value of 1 in the input tensor,
* there may be precision difference in the calculation results.
* @li When the tensor "x" is empty and "p" < 0 or "p" is infinity, we cannot reduce the whole tensor or reduce
* over an empty dimension.

* @par Third-party framework compatibility
* Compatible with the Pytorch operator LpNorm.
*/
REG_OP(LpNormV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(p, Float, 2.0)
    .ATTR(axes, ListInt, {})
    .ATTR(keepdim, Bool, false)
    .ATTR(epsilon, Float, 1e-12f)
    .OP_END_FACTORY_REG(LpNormV2)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_MATH_OPS_H_
