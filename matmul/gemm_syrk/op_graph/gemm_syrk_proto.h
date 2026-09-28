/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_syrk_proto.h
 * \brief GemmSyrk IR definition: C = alpha * (A @ A^T) + beta * C, in-place on C.
 */

#ifndef OPS_GEMM_SYRK_PROTO_H
#define OPS_GEMM_SYRK_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes the symmetric rank-k update: "C = alpha * (a @ a^T) + beta * C",
 * producing the complete symmetric matrix in-place on "c". \n
 * @par Inputs:
 * Two inputs, including:
 * @li a: A matrix tensor. Must be one of the following types: float16, bfloat16.
 * The format supports ND. The shape is (m, k) or (..., m, k) with 2 to 6 dims,
 * where the leading "..." axes are batch axes. \n
 * - When transpose_x is false, "a" is the row-major (..., m, k) storage and the m axis
 * is the second-to-last dim of "a". \n
 * - When transpose_x is true, "a" is the transposed (..., k, m) storage (cublas syrk
 * OP_T semantics) and the m axis is the last dim of "a"; the op computes
 * "C = alpha * (a^T @ a) + beta * C".
 * @li c: A symmetric matrix tensor, the in-place input and output (GE aliases the
 * same-name input/output to one device buffer). Must be the same type as "a".
 * The format supports ND. The shape is (m, m) or (..., m, m) with the same dims,
 * batch axes and m extent as "a"; in-place update does not support broadcast.
 * The kernel contract requires the input "c" to be symmetric: the lower-triangle
 * region of the result is written as the transposed mirror of the upper-triangle
 * region.
 *
 * @par Attributes:
 * Four attributes, including:
 * @li alpha: An optional float. Scale factor of the "a @ a^T" term. Default to be 1.0.
 * @li beta: An optional float. Scale factor of the in-place "c" term. Default to be 1.0.
 * @li transpose_x: An optional bool. Declares the storage layout of "a": false keeps
 * the row-major (..., m, k) ND view; true binds the transposed (..., k, m) storage.
 * Default to be false.
 * @li fill_mode: An optional string. Declares the output region of "c": "full" writes
 * the complete symmetric matrix (both the upper and lower triangles); "up"/"low" are
 * reserved and not implemented yet, an error is returned when they are passed.
 * Default to be "full".
 * @li op_impl_mode: An optional integer for GemmSyrk. op_impl_mode_enum: 0x1: default
 * 0x2: high_performance 0x4: high_precision 0x8: super_performance
 * 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
 * before multiplication. \n
 *
 * @par Outputs:
 * One output, including:
 * c: The in-place symmetric matrix result, identical to the input "c" tensor.
 * Must be one of the following types: float16, bfloat16. The format supports ND.
 * The shape is (m, m) or (..., m, m). \n
 * Special execution paths: when k is 0, "a @ a^T" is a zero matrix and the op degrades
 * to the element-wise "C = beta * C"; when m or the product of the batch axes is 0,
 * the op returns directly without launching the kernel.
 */
REG_OP(GemmSyrk)
    .INPUT(a, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(c, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(c, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 1.0)
    .ATTR(transpose_x, Bool, false)
    .ATTR(fill_mode, String, "full")
    .ATTR(op_impl_mode, Int, 0x1)
    .OP_END_FACTORY_REG(GemmSyrk)
} // namespace ge

#endif // OPS_GEMM_SYRK_PROTO_H
