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
 * \file transpose_batch_mat_mul_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_TRANSPOSE_BATCH_MAT_MUL_PROTO_H_
#define OPS_MATMUL_TRANSPOSE_BATCH_MAT_MUL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b". \n
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix tensor. Must be one of the following types:
* float32, float16, bfloat16. 3D. Has format ND.
* @li x2: A matrix tensor. Must be one of the following types:
* float32, float16, bfloat16. 3D. Has format ND.
* @li bias: An optional tensor. Bias for batchmatmul. Must be one of the following types:
* float32, float16, bfloat16. 1D. Has format ND.
* @li scale: A matrix tensor, quantization parameter.
             Must be one of the following types: uint64, int64. The format
             supports ND. The shape is 1D (t,), with t equal to b*n, where b, n is the same as that of x2.

* @par Attributes:
* Four attributes, including:
* @li perm_x1: A list int. "x1" is permuted to shape [B, M, K] before multiplication, the default value is no permutation.
* @li perm_x2: A list int. "x2" is permuted to shape [B, K, N] before multiplication, the default value is no permutation.
* @li perm_y: A list int. "y" is permuted after multiplication.
* @li enable_hf32: An optional bool. If True, enable enable_hi_float_32_execution.
* @li batch_split_factor: An optional int. Declares factor of output_batch. Default to be 1.

* @par Outputs:
* y: The result matrix tensor. 3D. Must be one of the following
* types: float32, float16, int8, bfloat16. 3D. Has format ND. \n
*/
REG_OP(TransposeBatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_INT64, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(perm_x1, ListInt, {})
    .ATTR(perm_x2, ListInt, {})
    .ATTR(perm_y, ListInt, {})
    .ATTR(enable_hf32, Bool, false)
    .ATTR(batch_split_factor, Int, 1)
    .OP_END_FACTORY_REG(TransposeBatchMatMul)
} // namespace ge

#endif // OPS_MATMUL_TRANSPOSE_BATCH_MAT_MUL_PROTO_H_