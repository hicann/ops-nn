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
 * \file quant_matmul_reduce_sum_proto.h
 * \brief
 */
#ifndef OPS_QUANT_MATMUL_REDUCE_SUM_PROTO_H_
#define OPS_QUANT_MATMUL_REDUCE_SUM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The fusion operator of QuantBatchMatmul and ReduceSum.

* @par Inputs:
* @li x1: A matrix tensor, with data type int8, format ND, 3D shape (batch, m, k).
* @li x2: A matrix tensor, with data type int8, format FRACTAL_NZ, 3D original shape (batch, k, n).
* @li dims: A 1D list or tuple of int64. Specifies the dimensions to reduce.
* @li bias: Reserved parameter, not supported now, please pass nullptr.
* @li x1_scale: A matrix tensor, with data type float32, format ND, 2D shape (b, m).
* @li x2_scale: A tensor, with data type bfloat16, format ND, 1D shape (n,).
* @li y_scale: Reserved parameter, not supported now, please pass nullptr.
* @li x1_offset: Reserved parameter, not supported now, please pass nullptr.
* @li x2_offset: Reserved parameter, not supported now, please pass nullptr.
* @li y_offset: Reserved parameter, not supported now, please pass nullptr.
* @li x2_table: Reserved parameter, not supported now, please pass nullptr. \n

* @par Attributes:
* @li dtype: A Int. Declare the output dtype, supports 27(bfloat16).
* @li compute_type: Reserved attributes, not supported now. Default is -1.
* @li transpose_x1: A bool. If true, changes the shape of "x1" from [m, k] to [k, m] before multiplication.
* Default: false. Only supports false now.
* @li transpose_x2: A bool. If true, changes the shape of "x2" from [k, n] to [n, k] before multiplication.
* Default: false. Only supports false now.
* @li group_size: Reverved attributes, not supported now. Default is -1.
* @li keep_dims: A bool. If true, keeps the original input dims in the output.
* Default: false. Only support false now. \n

* @par Outputs:
* y: A matrix tensor. The data type is bfloat16. The format supports ND. \n

* Atlas A2 Trainging Series Product/Atlas 800I A2 Inference Product or Atlas A3 Training Series Product: \n
* | x1   | x2   | x1Scale  | x2Scale  | yScale | x1Offset | x2Offset | yOffset | bias  | out     |
* |------|------|----------|----------|--------|----------|----------|---------|-------|---------|
* | INT8 | INT8 | FLOAT32  | BFLOAT16 | null   | null     | null     | null    | null  |BFLOAT16 |
*/
REG_OP(QuantMatmulReduceSum)
    .INPUT(x1, TensorType({DT_INT8}))
    .INPUT(x2, TensorType({DT_INT8}))
    .INPUT(dims, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(y_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(y_offset, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(x2_table, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(compute_type, Int, -1)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, -1)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(QuantMatmulReduceSum)
} // namespace ge

#endif // OPS_QUANT_MATMUL_REDUCE_SUM_PROTO_H_
