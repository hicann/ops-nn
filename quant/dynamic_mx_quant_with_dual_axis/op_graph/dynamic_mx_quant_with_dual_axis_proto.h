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
 * \file dynamic_mx_quant_with_dual_axis_proto.h
 * \brief
 */
 
#ifndef OPS_NN_DEV_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_PROTO_H
#define OPS_NN_DEV_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Performs dynamic MX quantization on input tensor.
* Quantizes the input tensor along the specified axis using block-wise scaling factors.
* Supports various 4-bit and 8-bit floating point output formats.

* @par Inputs:
* @li x: An input tensor of type float16 or bfloat16.
* The shape supports at least 2 dimensions, and at most 7 dimensions.

* @par Attributes:
* @li round_mode: An optional string. Defaults to "rint".
* @li dst_type: An optional int. Declare the output y dtype. Support FLOAT4_E2M1, FLOAT4_E1M2,
* FLOAT8_E4M3FN or FLOAT8_E5M2. Defaults to FLOAT4_E2M1.
* @li scale_alg: An optional int.The algorithm for the scale in quantization.Default to 0.
* Support MxFP8(OCP , count 0)

* @par Outputs:
* @li y1: Quantized output tensor. It has the same shape and rank as input x.
* @li mxscale1: An output tensor of type FLOAT8_E8M0. Shape needs to meet the following conditions: \n
* - rank(mxscale) = rank(x) + 1.
* - axis= -1.
* - axis_change = axis if axis >= 0 else axis + rank(x).
* - mxscale.shape[axis_change] = (ceil(x.shape[axis] / blocksize) + 2 - 1) / 2.
* - mxscale.shape[rank(x)] = 2.
* - Other dimensions match input x.
* @li y2: Quantized output tensor. It has the same shape and rank as input x.
* @li mxscale2: An output tensor of type FLOAT8_E8M0. Shape needs to meet the following conditions: \n
* - rank(mxscale) = rank(x) + 1.
* - axis= -2.
* - axis_change = axis if axis >= 0 else axis + rank(x).
* - mxscale.shape[axis_change] = (ceil(x.shape[axis] / blocksize) + 2 - 1) / 2.
* - mxscale.shape[rank(x)] = 2.
* - Other dimensions match input x.
* mxscale tensor is padded with zeros to ensure its size along the quantized axis is even.

* @attention Constraints:
* @li When dst_type is DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN, round_mode only supports "rint".
* @li When dst_type is DT_FLOAT4_E2M1 or DT_FLOAT4_E1M2, round_mode supports "rint", "floor" and "round".
* @li If dst_type is DT_FLOAT4_E2M1 or DT_FLOAT4_E1M2, the input x last dimension of the shape must be divisible by 2.

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, ONNX, TensorFlow, or PyTorch.
*/
REG_OP(DynamicMxQuantWithDualAxis)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(mxscale1, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y2, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(mxscale2, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(round_mode, String, "rint")
    .ATTR(dst_type, Int, DT_FLOAT4_E2M1)
    .ATTR(scale_alg, Int, 0)
    .OP_END_FACTORY_REG(DynamicMxQuantWithDualAxis)

} // namespace ge

#endif // OPS_NN_DEV_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_PROTO_H
