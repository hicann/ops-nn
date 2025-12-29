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
 * \file fused_mat_mul_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_FUSED_MAT_MUL_PROTO_H_
#define OPS_MATMUL_FUSED_MAT_MUL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Multiplies matrix "a" by matrix "b", producing "a @ b".
 * @par Inputs:
 * Four inputs, including:
 * @li x1: A matrix tensor. Must be one of the following types: float16, bfloat16, float32.
 * float32 is supported for the the following fused_op_types: "","relu","add","mul".
 * @li x2: A matrix tensor. Must be one of the following types: float16, bfloat16, float32.
 * float32 is supported for the the following fused_op_types: "","relu","add","mul".
 * @li bias: An optional tensor. 1D. Must be one of the following types: float16, bfloat16, float32.
 * bfloat16 is only supported in Ascend 910_95 AI processor
 * Must be one of the following: "","relu","add","mul".
 * @li x3: An Optional tensor. Must be one of the following types: float16, bfloat16, float32.
 * Must be one of the following: "add",mul".
 *
 * @par Attributes:
 * @li transpose_x1: A bool. If True, changes the shape of "x1" from [M, K] to
 * [K, M] before multiplication.
 * @li transpose_x2: A bool. If True, changes the shape of "x2" from [K, N] to
 * [N, K] before multiplication.
 * @li enable_hf32: A bool. This input is supported for the the following fused_op_types: "","relu","add","mul".
 * @li fused_op_type: A string. The fused_op_type include "","add","mul","gelu_erf","gelu_tanh","relu".
 * Default type is defined as "".
 * @par Outputs:
 * y: The result matrix tensor. Must be one of the following types: float16, bfloat16, float32,
 */
REG_OP(FusedMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(enable_hf32, Bool, false)
    .ATTR(fused_op_type, String, "")
    .OP_END_FACTORY_REG(FusedMatMul)
} // namespace ge

#endif // OPS_MATMUL_FUSED_MAT_MUL_PROTO_H_