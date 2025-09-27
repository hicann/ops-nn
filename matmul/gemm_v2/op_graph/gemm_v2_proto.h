/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_v2_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_GEMM_V2_PROTO_H_
#define OPS_MATMUL_GEMM_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief GemmV2 matrix "a" by matrix "b" and add matrix "c", producing "alpha * op(a) * op(b) + beta * op(c)". \n
* @par Inputs:
* Four inputs, including:
* @li a: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND].
* @li b: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND].
* @li alpha: A 1D Tensor. Must be one of the following types: bfloat16,
* float16. Has format [ND]. \n
* @li beta: A 1D Tensor. Must be one of the following types: bfloat16,
* float16. Has format [ND]. \n
* @li c: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND].

* @par Attributes:
* @li transpose_a: A bool. If True, changes the shape of "a" from [M, K] to
* [K, M] before multiplication.
* @li transpose_b: A bool. If True, changes the shape of "b" from [K, N] to
* [N, K] before multiplication. \n

* @par Outputs:
* c: The result matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND]. \n
*/
REG_OP(GemmV2)
    .INPUT(a, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(alpha, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(c, TensorType({DT_FLOAT}))
    .OUTPUT(c, TensorType({DT_FLOAT}))
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .OP_END_FACTORY_REG(GemmV2)
} // namespace ge

#endif // OPS_MATMUL_GEMM_V2_PROTO_H_