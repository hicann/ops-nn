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
 * \file batch_mat_mul_v3_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_BATCH_MAT_MUL_V3_PROTO_H_
#define OPS_MATMUL_BATCH_MAT_MUL_V3_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Multiplies matrix "x1" by matrix "x2", producing "x1 * x2".
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float16,
* float32, bfloat16. 2D-6D. Has format [ND, NHWC, NCHW].
* @li x2: A matrix Tensor. Must be one of the following types: float16,
* float32, bfloat16. 2D-6D. Has format [ND, NHWC, NCHW].
* @li bias: A optional Tensor. Must be one of the following types: float16,
* float32, bfloat16. Has format [ND, NHWC, NCHW].
* @li offset_w: A optional Tensor. Must be one of the following types:
* int8, int4. Has format [ND, NHWC, NCHW].

* @par Attributes:
* @li adj_x1: A bool. If True, changes the shape of "x1" from [B, M, K] to
* [B, K, M] before multiplication.
* @li adj_x2: A bool. If True, changes the shape of "x2" from [B, K, N] to
* [B, N, K] before multiplication.
* @li offset_x: An optional integer for quantized BatchMatMulV3.
* @li enable_hf32: An optional bool for BatchMatMulV3. If True, enable enable_hi_float_32_execution
* before multiplication. \n

* @par Outputs:
* y: The result matrix Tensor. Must be one of the following types: float16,
* float32, bfloat16. 2D-6D. Has format [ND, NHWC, NCHW]. BatchMatMulV3 supports broadcasting in the batch dimensions.
*/

REG_OP(BatchMatMulV3)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMulV3)
} // namespace ge

#endif // OPS_MATMUL_BATCH_MAT_MUL_V3_PROTO_H_