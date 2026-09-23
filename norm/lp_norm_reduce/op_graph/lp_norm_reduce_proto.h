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
 * \file lp_norm_reduce_proto.h
 * \brief LpNormReduce graph mode operator prototype definition.
 */
#ifndef OPS_NORM_LP_NORM_REDUCE_PROTO_H_
#define OPS_NORM_LP_NORM_REDUCE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes LpNormReduce.

 * @par Inputs:
 * x: A ND tensor of type float16, bfloat16, float32.
 *
 * @par Attributes:
 * @li p: An optional int, "inf" or "-inf", default value is 2, p >= 0.
 * @li axes: ListInt, an optional attribute, indicates dimensions over which to compute the norm.
 * Default is {}, meaning all axes will be computed.
 * @li keepdim: An optional bool. If set to true, the reduced dimensions are retained in the result
 * as dimensions with size one. Default is false.
 * @li epsilon: An optional float. A value added to the denominator for numerical stability. Default is 1e-12.

 * @par Outputs:
 * y: A ND tensor has the same dtype as "x". The shape of "y" is depending on "axes" and "keepdim".

 * @attention Constraints:
 * @li When the attribute "axes" is specified as the axis with a shape dimension value of 1 in the input tensor,
 * there may be precision difference in the calculation results.
 * @li When the tensor "x" is empty and "p" is infinity, we cannot reduce the whole tensor or reduce over an empty
 * dimension.
 * @li This operator will be deprecated in the future. Replace it with LpNormReduceV2 operator.

 * @par Third-party framework compatibility
 * Compatible with the Pytorch operator LpNormReduce.
 */
#ifndef OPS_PROTO_DEF_LPNORMREDUCE
#define OPS_PROTO_DEF_LPNORMREDUCE
REG_OP(LpNormReduce)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(p, Int, 2)
    .ATTR(axes, ListInt, {})
    .ATTR(keepdim, Bool, false)
    .ATTR(epsilon, Float, 1e-12f)
    .OP_END_FACTORY_REG(LpNormReduce)
#endif // OPS_PROTO_DEF_LPNORMREDUCE
} // namespace ge

#endif // OPS_NORM_LP_NORM_REDUCE_PROTO_H_
