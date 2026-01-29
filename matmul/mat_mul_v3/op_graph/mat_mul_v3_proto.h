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
 * \file mat_mul_v3_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_MAT_MUL_V3_PROTO_H_
#define OPS_MATMUL_MAT_MUL_V3_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Multiplies matrix "x1" by matrix "x2", producing "x1 @ x2". \n
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16, float32. Has format [ND, NHWC, NCHW].
* @li x2: A matrix Tensor. 2D/4D(NZ格式). Must be one of the following types: bfloat16,
* float16, float32. Has format [ND, NHWC, NCHW, NZ].
* @li bias: A 1D Tensor. Must be one of the following types: float32,
* float16, float32. Has format [ND, NHWC, NCHW]. \n
* bfloat16 is only supported in Ascend 950 AI processor
* @li offset_w: An optional matrix Tensor. 2D. Must be one of the following
* types: int8, int4.

* @par Attributes:
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [M, K] to
* [K, M] before multiplication.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [K, N] to
* [N, K] before multiplication.
* @li offset_x: An optional integer for quantized MatMulV2Compress.
* The negative offset added to the input x1 for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0".
* @li opImplMode: An optional integer for MatMulV3. op_impl_mode_enum: 0x1: default 
* 0x2: high_performance 0x4: high_precision 0x8: super_performance
* 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
* before multiplication. \n

* @par Outputs:
* y: The result matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16, float32. Has format [ND, NHWC, NCHW]. \n
*/
REG_OP(MatMulV3)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .ATTR(opImplMode, Int, 0x1)
    .OP_END_FACTORY_REG(MatMulV3)
} // namespace ge

#endif // OPS_MATMUL_MAT_MUL_V3_PROTO_H_