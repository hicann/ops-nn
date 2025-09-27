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
 * \file quant_batch_matmul_v4_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_QUANT_BATCH_MATMUL_V4_PROTO_H_
#define OPS_MATMUL_QUANT_BATCH_MATMUL_V4_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The fusion operator of antiquant and matmul for A8W4 and MxA8W4.

* @par Inputs:
* @li x1: A matrix Tensor. The shape supports (m, k), and the format supports ND.
* The data type supports float8_e5m2, float8_e4m3fn. The m and k value must be in [1, 65535]. The k value
* must be in [1, 65535] and must be a multiple of 64.
* @li x2: A matrix Tensor of quantized weight. The shape supports (n, k), and the format supports ND/FRACTAL_NZ.\n
* - In MxA8W4 scenario:
* For ND format, the data type supports float4_e2m1. For FRACTAL_NZ format, the data type supports float4_e2m1 and float4_e1m2.\n
* - In A8W4 scenario:
* The data type supports float4_e2m1.
* The k, n value must be in [1, 65535] and k, n value must be a multiple of 64.
* The shape must be (k1, n1, n0, k0) (n0 = 16, k0 = 32) when the format is FRACTAL_NZ.
* @li bias: An Optional Tensor.
* The shape supprts(1, n), format supports ND, the type supports bfloat16, float16.
* @li x1_scale: An Optional Tensor for quantization parameters.
* The type supports float8_e8m0, bfloat16, bfloat16, format supports ND.
* The shape supprts(m, k/group_size) when type is float8_e8m0.
* Tne type bfloat16, bfloat16 is not supported yet.
* @li x2_scale: An Optional Tensor for quantization parameters.
* The type supports float8_e8m0, bfloat16, bfloat16, format supports ND, The shape supprts(n, k/group_size).
* @li y_scale: An Optional Tensor for quantization parameters.
* The type support uint64, format supports ND, The shape supprts(1, n).
* @li x1_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li x2_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li y_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li x2_table: An Optional Tensor for quantization parameters. It's not supported yet.
*
* @par Attributes:
* @li dtype: An int32. The data type of y. It supports 1(float16) and 27(bfloat16).
* @li compute_type: An int32. It's compute type. The value must be -1(auto) or 4(f8f4).
* @li transpose_x1: A bool. x1 is transposed if true. Default: false.
* When transpose_x1 is true, x1's shape is (k, m). Currently, it should always be false.
* @li transpose_x2: A bool. x2 is transposed if true. Default: false.
* When transpose_x2 is true, x2's shape is (n, k), x2_scale's shape should be (n, k / group_size).
* Currently, it should always be true.
* @li group_size: An int32. The value must be 32.

* @par Outputs:
* y: A matrix Tensor. The data type support bfloat16, float16. The format supports ND.
*/
REG_OP(QuantBatchMatmulV4)
    .INPUT(x1, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT4_E1M2, DT_FLOAT4_E2M1, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0, DT_UINT64}))
    .OPTIONAL_INPUT(y_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(y_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_table, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(compute_type, Int, -1)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, -1)
    .OP_END_FACTORY_REG(QuantBatchMatmulV4)
} // namespace ge

#endif // OPS_MATMUL_QUANT_BATCH_MATMUL_V4_PROTO_H_