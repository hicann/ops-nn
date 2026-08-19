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
 * \file group_norm_proto.h
 * \brief
 */
#ifndef GROUP_NORM_PROTO_H_
#define GROUP_NORM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Applies group normalization to an ND tensor.
 *
 * @par Inputs:
 * @li x: A 2D-8D ND tensor of type float16 or float32, interpreted as (N, C, *).
 * @li gamma: A 1D tensor with C elements. The dtype must be the same as x.
 * @li beta: A 1D tensor with C elements. The dtype must be the same as x.
 *
 * @par Attributes:
 * @li num_groups: Required int. C must be divisible by num_groups.
 * @li data_format: Optional string. Defaults to "NCHW" and is reserved for compatibility.
 * @li eps: Optional float. Defaults to 0.0001.
 * @li is_training: Optional bool. Defaults to true and is reserved for compatibility.
 *
 * @par Outputs:
 * @li y: Normalized tensor with the same shape and dtype as x.
 * @li mean: Per-group mean with shape (N, num_groups) and the same dtype as x.
 * @li variance: Per-group population variance with shape (N, num_groups) and the same dtype as x.
 */
#ifndef OPS_PROTO_DEF_GROUPNORM
#define OPS_PROTO_DEF_GROUPNORM
REG_OP(GroupNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(eps, Float, 0.0001f)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNorm)
#endif // OPS_PROTO_DEF_GROUPNORM
} // namespace ge

#endif // GROUP_NORM_PROTO_H_
