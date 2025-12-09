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
 * \file group_norm_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_GROUP_NORM_GRAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_GROUP_NORM_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief backward operator for group normalization. \n
 * @par Inputs:
 * Five input, including:
 * @li dy: A tensor. Group grad. Datatype support float32, float16, bfloat16. Format support ND.
 * "dy" supports 2-8 dimensions (N, C, *), the calculation logic only cares about the first two dimensions (N and C),
 * and the rest can all be combined into one dimension.
 * @li mean: A tensor. Mean of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * Must be 2D (N, num_groups).
 * @li rstd: A tensor. Reciprocal standard deviation of each group. Datatype support float32, float16, bfloat16. Format
 support ND.
 * Must be 2D (N, num_groups).
 * @li x: A Tensor. Specifies the offset. Datatype support float32, float16, bfloat16. Format support ND.
 * "x" supports 2-8 dimensions (N, C, *), the calculation logic only cares about the first two dimensions (N and C),
 * and the rest can all be combined into one dimension.
 * @li gamma: A tensor. Specifies the scaling factor. Datatype support float32, float16, bfloat16. Format support ND.
 * Must be 1D. The value of "gamma" needs to be consistent with the C-axis value of "x".

 * @par Attributes:
 * @li num_groups: Int. Number specifying the number of group.
 * @li data_format: An optional string. Defaults to NCHW.
 * @li dx_is_require: An optional bool, controls whether to return dx. Defaults to true.
 * @li dgamma_is_require: An optional bool, controls whether to return dgamma. Defaults to true.
 * @li dbeta_is_require: An optional bool, controls whether to return dbeta. Defaults to true.

 * @par Outputs:
 * Three output, including:
 * @li dx: A tensor. x factor grad. Datatype is the same as the input datatype. Has the same format and shape as "x".
 * @li dgamma: A tensor. Scale factor grad. Has the same datatype, format and shape as "gamma".
 * @li dbeta: A tensor. Offset factor grad. Has the same datatype, format and shape as "gamma".
 * @par Third-party framework compatibility
 * @li Compatible with the backward of PyTorch operator GroupNorm.
 */

REG_OP(GroupNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(dx_is_require, Bool, true)
    .ATTR(dgamma_is_require, Bool, true)
    .ATTR(dbeta_is_require, Bool, true)
    .OP_END_FACTORY_REG(GroupNormGrad)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_GROUP_NORM_GRAD_OPS_H_