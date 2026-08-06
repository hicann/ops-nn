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
 * \file matmul_emu_split_weight_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_EMU_SPLIT_WEIGHT_PROTO_H_
#define OPS_MATMUL_EMU_SPLIT_WEIGHT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs dual-path BF16 GEMM fusion to simulate FP32 precision matrix multiplication.
* The FP32 weight is split offline into a high-bit BF16 weight and a low-bit residual BF16 weight.
* At inference time, two BF16 GEMMs are executed and linearly combined. \n
*
* @par Inputs:
* @li x: A tensor. Activation matrix.
* Must be one of the following types: bfloat16. The format supports ND.
* @li w_high: A tensor. High-bit weight, obtained by truncating FP32 weight to BF16.
* Must be one of the following types: bfloat16. Has the same type as input "x".
* The format supports ND.
* @li w_low: A tensor. Low-bit residual weight, obtained by dividing the residual by scale and truncating to BF16.
* Must be one of the following types: bfloat16. Has the same type as input "x".
* The shape must be identical to w_high. The format supports ND. \n
*
* @par Outputs:
 * @li y: A tensor. Output matrix.
 * Must be one of the following types: float32. The format supports ND. \n
*
* @par Attributes:
* @li w_low_scale: A required float. Scale factor for the low-bit residual weight. Defaults to 0.00390625 (1/256). \n
* @li transpose_x: An optional bool. Specifies whether to transpose input x. Defaults to false. \n
* @li transpose_w: An optional bool. Specifies whether to transpose weights w_high/w_low. Defaults to false. \n
 * @li y_dtype: A required int. Specifies the output data type of y. 0 for FP32. Defaults to 0. \n

* @par Constraints:
* Ascend 950 Series Products: \n
* - x, w_high, w_low must be BFLOAT16 and ND format.
* - x, w_high, w_low, y must be 2-D tensors.
* - x shape is [M, K], w_high shape is [K, N], w_low shape must be identical to w_high.
* - K dimension of x must match K dimension of w_high/w_low.
* - y shape must be [M, N].
* - M, K, N range: (0, INT32_MAX].
* - w_low_scale only supports 1/256 (0.00390625), must not be NaN or Inf.
 * - y_dtype only supports 0 (FP32).
* - transpose_x and transpose_w default to false. The aclnn interface does not accept transpose attributes;
*   callers should permute tensors before passing them in. \n

 * | x     | w_high | w_low | y      |
 * |-------|--------|-------|--------|
 * | BF16  | BF16   | BF16  | FLOAT  |
 */
REG_OP(MatmulEmuSplitWeight)
    .INPUT(x, TensorType({DT_BF16}))
    .INPUT(w_high, TensorType({DT_BF16}))
    .INPUT(w_low, TensorType({DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(w_low_scale, Float, 0.00390625)
    .ATTR(transpose_x, Bool, false)
    .ATTR(transpose_w, Bool, false)
    .ATTR(y_dtype, Int, 0)
    .OP_END_FACTORY_REG(MatmulEmuSplitWeight)
} // namespace ge

#endif // OPS_MATMUL_EMU_SPLIT_WEIGHT_PROTO_H_
