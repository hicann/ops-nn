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
 * \file batch_mat_mul_proto_extend.h
 * \brief
 */
#ifndef OPS_MATMUL_BATCH_MAT_MUL_PROTO_EXTEND_H_
#define OPS_MATMUL_BATCH_MAT_MUL_PROTO_EXTEND_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Multiplies matrix "a" by matrix "b", producing "a @ b".
 * @par Inputs:
 * Two inputs, including:
 * @li x1: A matrix Tensor. Must be one of the following types: float16,
 * float32, int32, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW].
 * @li x2: A matrix Tensor. Must be one of the following types: float16,
 * float32, int32, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW].
 * @par Attributes:
 * @li adj_x1: A bool. If True, changes the shape of "x1" from [B, M, K]
 * to [B, K, M] before multiplication.
 * @li adj_x2: A bool. If True, changes the shape of "x2" from [B, K, N]
 * to [B, N, K] before multiplication.
 * @par Outputs:
 * y: The result matrix Tensor. Must be one of the following types: float16,
 * float32, int32, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW]. BatchMatMul supports broadcasting in the
 * batch dimensions.
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator BatchMatmul.
 */
#ifndef OPS_PROTO_DEF_BATCHMATMUL
#define OPS_PROTO_DEF_BATCHMATMUL
REG_OP(BatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMul)
#endif

/**
 * @brief Multiplies matrix "a" by matrix "b", producing "a @ b" .
 * @par Inputs:
 * Four inputs, including:
 * @li x1: A matrix Tensor. Must be one of the following types: float16,
 * float32, int32, int8, int4, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW].
 * @li x2: A matrix Tensor. Must be one of the following types: float16,
 * float32, int32, int8, int4, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW].
 * @li bias: A optional Tensor. Must be one of the following types:
 * float16, float32, int32, bfloat16. Has format [ND, NHWC, NCHW].
 * @li offset_w: A optional Tensor. Must be one of the following types:
 * int8, int4. Has format [ND, NHWC, NCHW].
 * @par Attributes:
 * @li adj_x1: A bool. If True, changes the shape of "x1" from [B, M, K] to
 * [B, K, M] before multiplication.
 * @li adj_x2: A bool. If True, changes the shape of "x2" from [B, K, N] to
 * [B, N, K] before multiplication.
 * @li offset_x: An optional integer for quantized BatchMatMulV2.
 * @par Outputs:
 * y: The result matrix Tensor. Must be one of the following types: float16,
 * float32, int32, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC]. Has the same shape
 * length as "x1" and "x2".
 * @attention Constraints:
 * if performances better in format NZ, please close
 * "MatmulTransdataFusionPass" in fusion configuration.
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator BatchMatmul.
 */
#ifndef OPS_PROTO_DEF_BATCHMATMULV2
#define OPS_PROTO_DEF_BATCHMATMULV2
        REG_OP(BatchMatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(BatchMatMulV2)
#endif
} // namespace ge

#endif // OPS_MATMUL_BATCH_MAT_MUL_PROTO_EXTEND_H_
