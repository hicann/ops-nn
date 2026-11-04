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
 * \file mat_mul_v2_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_MAT_MUL_V2_PROTO_H_
#define OPS_MATMUL_MAT_MUL_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Multiplies matrix "a" by matrix "b", producing "a @ b".
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: float32,
* float16, int32, int8, int4, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].
* @li x2: A matrix Tensor. 2D. Must be one of the following types: float32,
* float16, int32, int8, int4, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].
* @li bias: A 1D Tensor. Must be one of the following types: float32,
* float16, int32, bfloat16. Has format [ND, NHWC, NCHW].
* @li offset_w: A Optional 1D Tensor for quantized inference. Type is int8, int4, bfloat16.
* Reserved.

* @par Attributes:
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication.
* @li offset_x: An optional integer for quantized MatMulV2.
* The negative offset added to the input x1 for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0".

* @par Outputs:
* y: The result matrix Tensor. 2D. Must be one of the following types: float32,
* float16, int32, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].

* @attention Constraints:
* if performances better in format NZ, please close
* "MatmulTransdataFusionPass" in fusion configuration.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MatMul.
*/
#ifndef OPS_PROTO_DEF_MATMULV2
#define OPS_PROTO_DEF_MATMULV2
REG_OP(MatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(MatMulV2)
#endif
} // namespace ge

#endif // OPS_MATMUL_MAT_MUL_V2_PROTO_H_
