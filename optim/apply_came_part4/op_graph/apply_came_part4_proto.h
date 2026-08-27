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
 * \file apply_came_part4_proto.h
 * \brief ApplyCamePart4 图模式算子原型定义
 */
#ifndef OPS_OP_PROTO_INC_APPLY_CAME_PART4_H_
#define OPS_OP_PROTO_INC_APPLY_CAME_PART4_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief ApplyCamePart4.
 *
 * @par Inputs:
 * including:
 * @li param: A mutable Tensor with rank 2, shape is [n, m], support dtypes:
 * float16, float32, bfloat16.
 * @li m: A mutable Tensor with rank 2, shape is [n, m], support dtypes:
 * float16, float32, bfloat16.
 * @li r: A 1-dimensional Tensor, shape is [n], support dtypes:
 * float16, float32, bfloat16.
 * @li c: A 1-dimensional Tensor, shape is [m], support dtypes:
 * float16, float32, bfloat16.
 * @li weight_decay: A 1-dimensional float32 Tensor specifying weight decay, shape must be [1].
 * @li lr: A 1-dimensional float32 Tensor specifying the learning rate, shape must be [1].
 * @li beta3: A 1-dimensional float32 Tensor, shape must be [1].
 * @li sum_r: A 1-dimensional float32 Tensor, shape must be [1].
 * @li sum_u_r: A 1-dimensional Tensor, shape is [n], support dtypes:
 * float16, float32, bfloat16.
 * @li sum_u_c: A 1-dimensional Tensor, shape is [m], support dtypes:
 * float16, float32, bfloat16.
 * @li sum_u_rc: A 1-dimensional float32 Tensor, shape must be [1].
 * @li global_shape: A 1-dimensional float32 Tensor specifying the original shape N and M, shape must be [2]. \n
 *
 * @par Outputs:
 * @li param: A mutable tensor. Must have the same shape and type as input "param".
 * @li r: A mutable tensor. Must have the same shape and type as input "r".
 * @li c: A mutable tensor. Must have the same shape and type as input "c".
 *
 * @par Restrictions:
 * Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
#ifndef OPS_PROTO_DEF_APPLYCAMEPART4
#define OPS_PROTO_DEF_APPLYCAMEPART4
REG_OP(ApplyCamePart4)
    .INPUT(param, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(m, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(r, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(c, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(weight_decay, TensorType({DT_FLOAT}))
    .INPUT(lr, TensorType({DT_FLOAT}))
    .INPUT(beta3, TensorType({DT_FLOAT}))
    .INPUT(sum_u_r, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum_u_c, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum_u_rc, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(sum_r, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(global_shape, TensorType({DT_INT64}))
    .OUTPUT(param, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(r, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(c, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(ApplyCamePart4)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_APPLY_CAME_PART4_H_
