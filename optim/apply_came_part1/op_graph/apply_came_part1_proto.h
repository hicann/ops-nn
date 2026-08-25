/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NN_APPLY_CAME_PART1_PROTO_H
#define OPS_NN_APPLY_CAME_PART1_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes the ApplyCamePart1.
 *
 * @par Inputs:
 * including:
 * @li grad: A mutable Tensor with rank 2, such as [n, m] , support types:
 * float16, float32, bfloat16.
 * @li eps: A scalar, support types: float32. \n
 *
 * @par Outputs:
 * @li sum_grad_r: A 1-dimensional Tensor, such as [n], support types: float32.
 * @li sum_grad_c: A 1-dimensional Tensor, such as [m], support types: float32.
 * @li sum_grad_rc: A 1-dimensional Tensor, such as [1], support
 * types: float32. \n
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
#ifndef OPS_PROTO_DEF_APPLYCAMEPART1
#define OPS_PROTO_DEF_APPLYCAMEPART1
REG_OP(ApplyCamePart1)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(eps, TensorType({DT_FLOAT}))
    .OUTPUT(sum_grad_r, TensorType({DT_FLOAT}))
    .OUTPUT(sum_grad_c, TensorType({DT_FLOAT}))
    .OUTPUT(sum_grad_rc, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ApplyCamePart1)
#endif
} // namespace ge
#endif // OPS_NN_APPLY_CAME_PART1_PROTO_H
