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
 * \file anti_mx_quant_proto.h
 * \brief
 */
#ifndef QUANTIZE_ANTI_MX_QUANT_PROTO_H
#define QUANTIZE_ANTI_MX_QUANT_PROTO_H
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Dequantize the quantized FLOAT4/FLOAT8 input into FLOAT16/BFLOAT16/FLOAT32 format. \n

* @par Inputs:
* @li x: A tensor of type FLOAT4_E2M1/FLOAT4_E1M2 or FLOAT8_E5M2/FLOAT8_E4M3FN, specifying the input.
* The shape only supports 1-7 dimensions.
* @li mxscale: A tensor of type FLOAT8_E8M0, specifying the quantization coefficient.
* The shape only supports 2-8 dimensions.
* - mxscale.shape[axis - 1] = Even(Ceil(x.shape[axis], 32)).
* - mxscale.shape[-1] = 2.
* - Other dimensions maintain the same shape as x.

* @par Attributes:
* @li axis: An optional int, specifying the dequantization axis.
* Defaults to -1.
* @li dst_type: An optional int, specifying the dtype of output y. Target data type enum value:
* - 0: DT_FLOAT32
* - 1: DT_FLOAT16
* - 27: DT_BF16
* Defaults to BFLOAT16.

* @par Outputs:
* @li y: An output tensor of type FLOAT16, BFLOAT16 or FLOAT32. It has the same shape as input x.

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, Onnx, Tensorflow or Pytorch.
*/
REG_OP(AntiMxQuant)
    .INPUT(x, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(mxscale, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(axis, Int, -1)
    .ATTR(dst_type, Int, DT_BF16)
    .OP_END_FACTORY_REG(AntiMxQuant)
} // namespace ge

#endif // QUANTIZE_ANTI_DYNAMIC_MX_QUANT_PROTO_H
