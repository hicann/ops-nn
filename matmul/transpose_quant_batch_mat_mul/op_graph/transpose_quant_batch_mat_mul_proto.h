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
 * \file transpose_quant_batch_mat_mul_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_TRANSPOSE_QUANT_BATCH_MAT_MUL_PROTO_H_
#define OPS_MATMUL_TRANSPOSE_QUANT_BATCH_MAT_MUL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Multiplies matrix "a" by matrix "b", producing "a @ b". \n
* @par Inputs:
* Five inputs, including:
* @li x1: A matrix tensor. Must be one of the following types:
* float8_e4m3fn, float8_e5m2. 3D. Has format ND.
* @li x2: A matrix tensor. Must be one of the following types:
* float8_e4m3fn, float8_e5m2. 3D. Has format ND.
* @li bias: An optional tensor. Bias for batchmatmul. Must be one of the following types:
* float32, float16, bfloat16. 1D. Has format ND.
* @li x1_scale: A matrix tensor, quantization parameter.
             Must be one of the following types: float32. The format
             supports ND. The shape is 1D (t,), with t equal to m, where m is the same as that of x1.
* @li x2_scale: A matrix tensor, quantization parameter.
             Must be one of the following types: float32. The format
             supports ND. The shape is 1D (t,), with t equal to n, where n is the same as that of x2.

* @par Attributes:
* Six attributes, including:
* @li dtype: An int. Declare the output type, supports  1(float16), 27(bfloat16). Default: 1(float16).
* @li group_size: An optional int. Indicating the ratio between x1_scale/x2_scale and x1/x2 in group dequantization. Default to be 0.
* @li perm_x1: A list int. "x1" is permuted to shape [B, M, K] before multiplication, the default value is [1, 0, 2].
* @li perm_x2: A list int. "x2" is permuted to shape [B, K, N] before multiplication, the default value is [0, 1, 2].
* @li perm_y: A list int. "y" is permuted after multiplication, the default value is [1, 0, 2].
* @li batch_split_factor: An optional int. Declares factor of output_batch. Default to be 1.

* @par Outputs:
* One output, including:
* y: A matrix Tensor. Must be one of the following types: float16, bfloat16. 
  The format supports ND. The shape dim must be 3D. \n
*/
REG_OP(TransposeQuantBatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(x2, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(dtype, Int, 1)
    .ATTR(group_size, Int, 0)
    .ATTR(perm_x1, ListInt, {1, 0, 2})
    .ATTR(perm_x2, ListInt, {0, 1, 2})
    .ATTR(perm_y, ListInt, {1, 0, 2})
    .ATTR(batch_split_factor, Int, 1)
    .OP_END_FACTORY_REG(TransposeQuantBatchMatMul)
} // namespace ge

#endif // OPS_MATMUL_TRANSPOSE_QUANT_BATCH_MAT_MUL_PROTO_H_