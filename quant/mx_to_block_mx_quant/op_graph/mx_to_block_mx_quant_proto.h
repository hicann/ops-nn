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
 * \file mx_to_block_mx_quant_proto.h
 * \brief
 */
#ifndef MX_TO_BLOCK_MX_QUANT_PROTO_H
#define MX_TO_BLOCK_MX_QUANT_PROTO_H
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Convert FP4-MX format data into FP8 block-quantized format. \n

* @par Inputs:
* @li x: A tensor of type FLOAT4_E2M1 or FLOAT4_E1M2, specifying the input.
* The shape only supports 2-3 dimensions.
* @li mxscale: A tensor of type FLOAT8_E8M0. Shape needs to meet the following conditions: \n
* The shape only supports 3-4 dimensions.
* - mxscale.shape[-2] = (Ceil(x.shape[axis], 32) + 2 - 1) / 2.
* - mxscale.shape[-1] = 2.
* - Other dimensions maintain the same shape as x.

* @par Outputs:
* @li y: An output tensor of type FLOAT8_E5M2 or FLOAT8_E4M3FN. It has the same shape as input x.
* @li scale1: An output tensor of type FLOAT8_E8M0. Shape needs to meet the following conditions: \n
* - It has the same shape as input mxscale.
* @li scale2: An output tensor of type DT_FLOAT8_E8M0. Shape needs to meet the following conditions: \n
* - rank(scale2) = rank(x) + 1.
* - scale2.shape[-3] = ((Ceil(x.shape[-2], 32) + 2 - 1) / 2) * 2 / 2.
* - scale2.shape[-2] = x.shape[-1].
* - scale2.shape[-1] = 2.
* - Other dimensions match input x.
* - scale2 tensor is padded with zeros to ensure its size along the quantized axis is even.

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, Onnx, Tensorflow or PyTorch.
*/
REG_OP(MxToBlockMxQuant)
    .INPUT(x, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .INPUT(mxscale, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OUTPUT(scale1, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(scale2, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(dst_type, Int, DT_FLOAT8_E4M3FN)
    .OP_END_FACTORY_REG(MxToBlockMxQuant)
} // namespace ge

#endif // MX_TO_BLOCK_MX_QUANT_PROTO_H
