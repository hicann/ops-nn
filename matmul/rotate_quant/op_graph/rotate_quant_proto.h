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
 * \file rotate_quant_proto.h
 * \brief
 */
#ifndef OPS_ROTATE_QUANT_PROTO_H_
#define OPS_ROTATE_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs rotation transformation on the input tensor x using a rotation matrix,
* followed by per-token symmetric dynamic quantization.
*
* @par Inputs:
* @li x: Input tensor for rotation and quantization. Data type supports FLOAT16 and BFLOAT16.
* Format ND, 2D shape (M, N). Supports non-contiguous tensors.
* @li rotation: Rotation matrix. Data type must be the same as x. Format ND, 2D shape (K, K).
* Must be a square matrix. Supports non-contiguous tensors. \n

* @par Attributes:
* @li y_dtype: An Int. Specifies the output data type of y. Default is 0.
* @li alpha: A Float. Scaling coefficient for clamp range limitation. Default is 0.0 (no clamp). \n

* @par Outputs:
* @li y: Quantized output tensor. Data type supports INT8 and INT4. Format ND, 2D shape (M, N).
* @li scale: Per-token dynamic quantization scaling factors. Data type is FLOAT32. Format ND, 1D shape (M). \n

* @par Constraints:
* Atlas A3 Training Series Products/Atlas A3 Inference Series Products, Atlas A2 Training Series Products/Atlas A2 Inference Series Products: \n
* - x shape is (M, N), rotation shape is (K, K). rotation must be a square matrix.
* - N must be a multiple of K, and N must be divisible by 8.
* - x and rotation must have the same data type.
* - scale output shape must be (M).
* - N range: [128, 16000], K range: [16, 1024]. \n

* | x        | rotation | y    | scale   |
* |----------|----------|------|---------|
* | FLOAT16  | FLOAT16  | INT8 | FLOAT32 |
* | BF16     | BF16     | INT8 | FLOAT32 |
* | FLOAT16  | FLOAT16  | INT4 | FLOAT32 |
* | BF16     | BF16     | INT4 | FLOAT32 |
*/
REG_OP(RotateQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
 	.INPUT(rotation, TensorType({DT_FLOAT16, DT_BF16}))
 	.OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
 	.OUTPUT(scale, TensorType({DT_FLOAT32}))
 	.ATTR(y_dtype, Int, 0)
 	.ATTR(alpha, Float, 0.0)
 	.OP_END_FACTORY_REG(RotateQuant)
} // namespace ge

#endif // OPS_ROTATE_QUANT_PROTO_H_
