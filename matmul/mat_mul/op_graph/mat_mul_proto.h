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
 * \file mat_mul_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_MAT_MUL_PROTO_H_
#define OPS_MATMUL_MAT_MUL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Multiplies matrix "a" by matrix "b", producing "a @ b".
* @par Inputs:
* Three inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].
* @li x2: A matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].
* @li bias: A optional 1D Tensor. Must be one of the following types: float16,
* float32, int32, bfloat16. Has format [ND, NHWC, NCHW].
* Reserved.

* @par Attributes:
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [M, K] to
* [K, M] before multiplication.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [K, N] to
* [N, K] before multiplication.

* @par Outputs:
* y: The result matrix Tensor. 2D. Must be one of the following types: float16,
* float32, int32, hifloat8. Has format [ND, NHWC, NCHW].

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MatMul.
*/
#ifndef OPS_PROTO_DEF_MATMUL
#define OPS_PROTO_DEF_MATMUL
REG_OP(MatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(MatMul)
#endif
} // namespace ge

#endif // OPS_MATMUL_MAT_MUL_PROTO_H_
