/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_APPLY_CAME_PART3_PROTO_H
#define OPS_NN_APPLY_CAME_PART3_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes the ApplyCamePart3. Came: Commutative Adam with Momentum Estimator. \n
 * hat_u = u / Maximum(1, sum_square_u / (M * N) / clip_threshold) \n
 * m = beta1 * m + (1 - beta1) * hat_u, if use_first_moment == true \n
 * U = (hat_u - m)^2 \n
 * sum_u_r = SUM(U + eps, -1) \n
 * sum_u_c = SUM(U + eps, -2) \n
 * sum_u_rc = SUM(U + eps, [-1, -2])
 *
 * @par Inputs:
 * including:
 * @li u: A 2-dimensional float32 tensor, shape is [n, m].
 * @li m: A 2-dimensional tensor, shape is [n, m], support dtypes:
 * float16, float32, bfloat16.
 * @li eps: A 1-dimensional float32 tensor, shape must be [1].
 * @li beta1: A 1-dimensional float32 tensor, shape must be [1].
 * @li clip_threshold: A 1-dimensional float32 tensor, shape must be [1].
 * @li sum_square_u: A 1-dimensional float32 tensor, shape must be [1].
 * @li global_shape: An optional 1-dimensional int64 tensor, shape must be [2],
 * value specifies the original shape (n, m), corresponding to N, M in the formula. \n
 * Default equals to u's shape. \n
 *
 * @par Attributes:
 * use_first_moment: An optional Bool. If true, update the computed output m.
 * Default: false. \n
 *
 * @par Outputs:
 * @li m: A mutable tensor. Must have the same shape as input "m", shape is [n, m].
 * Same dtype as "m".
 * @li sum_u_r:  A mutable float32 tensor. Must have the same type as input "u",
 * shape is [n].
 * @li sum_u_c:  A mutable float32 tensor. Must have the same type as input "u",
 * shape is [m].
 * @li sum_u_rc: A mutable float32 tensor. Must have the same type as input "u",
 * shape must be [1]. \n
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
#ifndef OPS_PROTO_DEF_APPLYCAMEPART3
#define OPS_PROTO_DEF_APPLYCAMEPART3
REG_OP(ApplyCamePart3)
    .INPUT(u, TensorType({DT_FLOAT}))
    .INPUT(m, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(eps, TensorType({DT_FLOAT}))
    .INPUT(beta1, TensorType({DT_FLOAT}))
    .INPUT(clip_threshold, TensorType({DT_FLOAT}))
    .INPUT(sum_square_u, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(global_shape, TensorType({DT_INT64}))
    .OUTPUT(m, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sum_u_r, TensorType({DT_FLOAT}))
    .OUTPUT(sum_u_c, TensorType({DT_FLOAT}))
    .OUTPUT(sum_u_rc, TensorType({DT_FLOAT}))
    .ATTR(use_first_moment, Bool, false)
    .OP_END_FACTORY_REG(ApplyCamePart3)
#endif
} // namespace ge

#endif // OPS_NN_APPLY_CAME_PART3_PROTO_H
