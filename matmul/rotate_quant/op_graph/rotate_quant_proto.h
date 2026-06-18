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
* followed by per-token symmetric dynamic quantization or per-group dynamic MX quantization. \n
*
* @par Inputs:
* @li x: A tensor. Input data for rotation and quantization. 
* Must be one of the following types: float16, bfloat16. The format supports ND.
* @li rot: A tensor. Rotation matrix. 
* Must be one of the following types: float16, bfloat16. Has the same type as input "x".
* The format supports ND.
* @li alpha: A tensor. Optional scaling coefficient for clamp range limitation.
* Only supported when y_dtype is float4_e2m1, float8_e4m3fn, or float8_e5m2. Not supported when y_dtype is int4 or int8.
* Must be bfloat16. The format supports ND. \n
*
* @par Outputs:
* @li y: When the output data type is int4 or int8, the shape is [M, N].
* When the output data type is float4_e2m1, float8_e4m3fn, or float8_e5m2, the shape is [M, N]. \n
* @li scale: When the output data type is float32, the shape is [M].
* When the output data type is float8_e8m0, the shape is [M, ceilDiv(N,64), 2]. \n
*
* @par Attributes:
* @li y_dtype: An optional int. Specifies the output data type of y. Defaults to DT_INT8. \n
* @li axis: An optional int. Specifies the axis for quantization. Defaults to -1. \n
* @li round_mode: An optional string. Specifies the rounding mode. Defaults to "rint". \n
* @li scale_alg: An optional int. Specifies the scale algorithm. Defaults to 0. \n
* @li dst_type_max: An optional float. Specifies the max value of destination type. Defaults to 0.0. \n
* @li trans: An optional bool. Specifies whether to transpose. Defaults to false. \n

* @par Constraints:
* Atlas A3 supports per-token dynamic quantization.Ascend 950PR/Ascend 950DT supports per-group dynamic MX quantization.
* Atlas A3 Training Series Products/Atlas A3 Inference Series Products, Atlas A2 Training Series Products/Atlas A2 Inference Series Products, Atlas 950 Series Products: \n
* - x shape is [M, N], rot shape is [K, K]. rot must be a square matrix.
* - N must be a multiple of K, and N must be divisible by 8.
* - x and rot must have the same data type.
* - scale output shape must be [M].
* - N range: [128, 16000], K range: [16, 1024]. \n

* Ascend 950PR/Ascend 950DT Series Products:
* - x is 1-D to 7-D, with the last dimension being N.
* - rot shape is [K, K] or [blockNum, K, K], where blockNum = K/N, and N must divide K.
* - x and rot must have the same data type.
* - scale output shape is [*, ceilDiv(N,64), 2].
* - K must be one of {32, 64, 128}. \n

* | x        | rot      | alpha | y              | scale       |
* |----------|----------|-------|----------------|-------------|
* | BF16     | BF16     | N/A   | INT4           | FLOAT32     |
* | BF16     | BF16     | N/A   | INT8           | FLOAT32     |
* | FLOAT16  | FLOAT16  | N/A   | INT4           | FLOAT32     |
* | FLOAT16  | FLOAT16  | N/A   | INT8           | FLOAT32     |
* | FLOAT16  | FLOAT16  | BF16  | FLOAT4_E2M1    | FLOAT8_E8M0 |
* | BF16     | BF16     | BF16  | FLOAT4_E2M1    | FLOAT8_E8M0 |
* | FLOAT16  | FLOAT16  | BF16  | FLOAT8_E4M3FN  | FLOAT8_E8M0 |
* | BF16     | BF16     | BF16  | FLOAT8_E4M3FN  | FLOAT8_E8M0 |
* | FLOAT16  | FLOAT16  | BF16  | FLOAT8_E5M2    | FLOAT8_E8M0 |
* | BF16     | BF16     | BF16  | FLOAT8_E5M2    | FLOAT8_E8M0 |
*/
REG_OP(RotateQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
 	.INPUT(rot, TensorType({DT_FLOAT16, DT_BF16}))
 	.OPTIONAL_INPUT(alpha, TensorType({DT_BF16}))
 	.OUTPUT(y, TensorType({DT_INT8, DT_INT4, DT_FLOAT4_E2M1, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
 	.OUTPUT(scale, TensorType({DT_FLOAT32, DT_FLOAT8_E8M0}))
 	.ATTR(y_dtype, Int, DT_INT8)
 	.ATTR(axis, Int, -1)
 	.ATTR(round_mode, String, "rint")
 	.ATTR(scale_alg, Int, 0)
 	.ATTR(dst_type_max, Float, 0.0)
 	.ATTR(trans, Bool, false)
 	.OP_END_FACTORY_REG(RotateQuant)
} // namespace ge

#endif // OPS_ROTATE_QUANT_PROTO_H_
