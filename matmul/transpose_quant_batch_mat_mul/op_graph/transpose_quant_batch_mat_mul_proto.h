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
* float8_e4m3fn, float8_e5m2, hifloat8. 3D. Has format ND.
* @li x2: A matrix tensor. Must be one of the following types:
* float8_e4m3fn, float8_e5m2, hifloat8. 3D. Has format ND.
* @li bias: An optional tensor. Reserved parameter, currently not supported.
* @li x1_scale: A matrix tensor, quantization parameter.
             Must be one of the following types: float32、float8_e8m0、uint64_t. The format
             supports ND. \n
             - In K-C quantification, the shape is 1D (m,), where m is the same as that of x1. \n
             - In MX quantification, the shape is 4D (m, b, ceil(k / 64), 2), where m, b, k match those of x1. \n
             - In T-C quantification, the shape can be empty or 1D (1,).
* @li x2_scale: A matrix tensor, quantization parameter.
             Must be one of the following types: float32、float8_e8m0、uint64_t. The format
             supports ND. \n
             - In K-C and T-C quantification, the shape is 1D (n,), where n is the same as that of x2. \n
              - In MX quantification, the shape is 4D. When perm_x2 is [0, 1, 2], the shape is (b, ceil(k / 64), n, 2);
                when perm_x2 is [0, 2, 1], the shape is (b, n, ceil(k / 64), 2), where b, k, n match those of x2.

* @par Attributes:
* Six attributes, including:
* @li dtype: An int. Declare the output type, supports  1(float16), 27(bfloat16), 34(hifloat8).
* @li group_size: An optional int. Indicating the ratio between x1_scale/x2_scale and x1/x2 in group dequantization.
* The group_size is composed of the group_size_m, group_size_n, and group_size_k, total occupying 48 bits.
* 0-15 bits of group_size indicate group_size_k, 16-31 bits indicate group_size_n, 32-47 bits indicate group_size_m,
* 48-63 bits of group_size are noneffective. \n
* If any of group_size_m, group_size_n, group_size_k calculated by group_size is 0, recalculate it by
* input shape, eg: group_size_m = m / scale_m (m % scale_m must be 0). \n
* In MX quantification, group_size_m and group_size_n only support 0 or 1, and group_size_k only supports 32.
* In K-C and T-C quantification, group_size only supports 0, other values do not take effect. Default to be 0.
* @li perm_x1: A list int. "x1" is permuted to shape [B, M, K] before multiplication.
* Supports [1, 0, 2], the default value is [1, 0, 2].
* @li perm_x2: A list int. "x2" is permuted to shape [B, K, N] before multiplication.
* In K-C quantification, supports [0, 1, 2]. In MX and T-C quantification, supports [0, 1, 2] or [0, 2, 1].
* The default value is [0, 1, 2].
* @li perm_y: A list int. "y" is permuted after multiplication. Supports [1, 0, 2], the default value is [1, 0, 2].
* @li batch_split_factor: An optional int. Declares factor of output_batch. Default to be 1.

* @par Outputs:
* One output, including:
* y: A matrix Tensor. Must be one of the following types: float16, bfloat16, hifloat8.
  The format supports ND. The shape dim must be 3D. \n
*/
REG_OP(TransposeQuantBatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0, DT_UINT64}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(group_size, Int, 0)
    .ATTR(perm_x1, ListInt, {1, 0, 2})
    .ATTR(perm_x2, ListInt, {0, 1, 2})
    .ATTR(perm_y, ListInt, {1, 0, 2})
    .ATTR(batch_split_factor, Int, 1)
    .OP_END_FACTORY_REG(TransposeQuantBatchMatMul)
} // namespace ge

#endif // OPS_MATMUL_TRANSPOSE_QUANT_BATCH_MAT_MUL_PROTO_H_
