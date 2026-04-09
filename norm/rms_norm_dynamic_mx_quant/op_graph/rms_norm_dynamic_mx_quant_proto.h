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
 * \file rms_norm_dynamic_mx_quant_proto.h
 * \brief
 */

#ifndef NORM_RMS_NORM_DYNAMIC_MX_QUANT_PROTO_H_
#define NORM_RMS_NORM_DYNAMIC_MX_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Performs RMS normalization followed by dynamic MX quantization on input tensor.
* Applies RMS normalization to the input tensor and then quantizes it using block-wise scaling factors.
* Supports various 4-bit and 8-bit floating point output formats.

* @par Inputs:
* @li x: An input tensor of type float16 or bfloat16.
* The shape supports at least 1 dimension, and at most 7 dimensions.
* @li gamma: A scale tensor of type float16, bfloat16 or float32.
* The shape must match the normalized dimension of x.
* @li beta: An optional bias tensor of type float16, bfloat16 or float32.
* The shape must match the normalized dimension of x.

* @par Attributes:
* @li epsilon: An optional float. A small value added to the variance for numerical stability. Defaults to 1e-06.
* @li scale_alg: An optional int. The quantization algorithm. Defaults to 0.
* Support OCP(0) or CUSTOM_NV(1).
* @li round_mode: An optional string. Defaults to "rint".
* @li dst_type: An optional int. Declare the output y dtype. Support FLOAT4_E2M1, FLOAT4_E1M2,
* FLOAT8_E4M3FN or FLOAT8_E5M2. Defaults to FLOAT4_E2M1.
* @li output_rstd: An optional bool. Whether to output rstd. Defaults to false.

* @par Outputs:
* @li y: Quantized output tensor. It has the same shape and rank as input x.
* @li mxscale: An output tensor of type FLOAT8_E8M0. The scale tensor for MX quantization.
* @li rstd: An output tensor of type FLOAT32. The reciprocal of standard deviation from RMS normalization.

* @attention Constraints:
* @li When dst_type is DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN, round_mode only supports "rint".
* @li When dst_type is DT_FLOAT4_E2M1 or DT_FLOAT4_E1M2, round_mode supports "rint", "floor" and "round".
* @li If dst_type is DT_FLOAT4_E2M1 or DT_FLOAT4_E1M2, the input x last dimension of the shape must be divisible by 2.

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, ONNX, TensorFlow, or PyTorch.
*/
REG_OP(RmsNormDynamicMxQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(mxscale, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(rstd, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-06)
    .ATTR(scale_alg, Int, 0)
    .ATTR(round_mode, String, "rint")
    .ATTR(dst_type, Int, 40)
    .ATTR(output_rstd, Bool, false)
    .OP_END_FACTORY_REG(RmsNormDynamicMxQuant)

} // namespace ge

#endif // NORM_RMS_NORM_DYNAMIC_MX_QUANT_PROTO_H_
