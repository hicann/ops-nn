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
 * \file fused_quant_mat_mul_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_FUSED_QUANT_MAT_MUL_PROTO_H_
#define OPS_MATMUL_FUSED_QUANT_MAT_MUL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief FusedQuantMatMul computes a quant matmul + vector operation fused operator.

* @par Inputs:
* @li x1: A matrix Tensor. The shape supports (m, k), and the format supports ND.
* The data type supports int8/int4. The m and k value must be in [1, 65535].
* @li x2: A matrix Tensor of quantized weight. The shape supports (k, n) for ND format if fused_op_type is not "swiglu".
* only supports (2, n1, k1, k0, n0) for NZ format, and ori shape should be (2, k, n) if fused_op_type is "swiglu".
* If fused_op_type is "gelu_erf" or "gelu_tanh", the format supports ND/NZ, the dtype supports int4/int8.
* The format only supports ND in a4w4 scenario if fused_op_type is "gelu_erf" or "gelu_tanh".
* The data type supports int8/int4. The k, n value must be in [1, 65535], k0 should be 16, n0 should be 64.
* @li bias: An Optional Tensor.
* The shape supports (1,), (n,) or (batch, 1, n), format supports ND, the type supports int32, bfloat16, float16 and float32, swiglu type is not supported yet.
* @li x1_scale: An Optional Tensor for quantization parameters. 
* The type supports float32, format supports ND if fused_op_type is "gelu_erf" or "gelu_tanh".
* @li x2_scale: An Optional Tensor for quantization parameters.
* The type supports uint64, int64, float32, bfloat16, format supports ND.
* The type only supports float32, bfloat16 if fused_op_type is "gelu_erf" or "gelu_tanh".
* The shape supprts(1), (n) if fused_op_type is not "swiglu".
* The shape supports (2, n) or (2 * n) if fused_op_type is "swiglu".
* The shape supports (n) or (1) if fused_op_type is "gelu_tanh" or "gelu_erf".
* @li y_scale:  An Optional Tensor for quantization parameters. It's not supported yet.
* @li x1_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li x2_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li y_offset: An Optional Tensor for quantization parameters. It's not supported yet.
* @li x2_table: An Optional Tensor for quantization parameters. It's not supported yet.
* @li x3: An Optional tensor.
* The data type supports float only.
* The shape supports (n,), only supports fused_op_type is "swiglu" and output type is int8 for quant scale fp16 to int8
*
* @par Attributes:
* @li dtype: An int32. The data type of y. It supports 1(float16), 2(int8) and 27(bfloat16).
* @li compute_type: An int32. It's compute type. It's not supported yet.
* @li transpose_x1: A bool. x1 is transposed if true. Default: false.
* When transpose_x1 is true, x1's shape is (k, m). Currently, it should always be false.
* @li transpose_x2: A bool. x2 is transposed if true. Default: false.
* When transpose_x2 is true, x2's shape is (n, k). Currently, it should always be false.
* @li group_size: An int32. It's not supported yet.

* @li fused_op_type: A string. The fused_op_type include "add","mul","gelu_erf","gelu_tanh","relu","swiglu".
* Currently, only "relu", "swiglu", "gelu_tanh" and "gelu_erf" is supported.
* Default type is defined as "".
* "relu":
* out = relu(matmul(x1, x2) + bias) * x2_scale.
* "swiglu":
* out = silu(matmul(x1, x2[0]) * x2_scale[0]) * (matmul(x1, x2[1]) * x2_scale[1]) * x3.
* "gelu_tanh":
* out = gelu_tanh(qbmmout).
* "gelu_erf":
* out = gelu_erf(qbmmout).

* @par Outputs:
* y: A matrix Tensor. The data type supports int8, float16, bfloat16. The format supports ND.
* The type only supports float16, bfloat16 if fused_op_type is "gelu_erf" or "gelu_tanh".
*/

REG_OP(FusedQuantMatMul)
    .INPUT(x1, TensorType({DT_INT8, DT_INT4}))
    .INPUT(x2, TensorType({DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32, DT_INT32}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_BF16, DT_UINT64, DT_INT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(y_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(y_offset, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_table, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(compute_type, Int, -1)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, -1)
    .ATTR(fused_op_type, String, "")
    .OP_END_FACTORY_REG(FusedQuantMatMul)
} // namespace ge

#endif // OPS_MATMUL_FUSED_MAT_MUL_PROTO_H_