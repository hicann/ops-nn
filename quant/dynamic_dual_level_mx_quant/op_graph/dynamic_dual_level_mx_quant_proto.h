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
 * \file dynamic_dual_level_mx_quant_proto.h
 * \brief
 */
 
#ifndef OPS_NN_DEV_DYNAMIC_DUAL_LEVEL_MX_QUANT_PROTO_H
#define OPS_NN_DEV_DYNAMIC_DUAL_LEVEL_MX_QUANT_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Online quantizes the input tensor per block.Then performs dynamic MX quantization on the output tensor.

* @par Inputs:
* @li x: An input tensor of type float16 or bfloat16.the input x last dimension of the shape must be divisible by 2.
* The shape supports at least 1 dimension, and at most 7 dimensions.
* @li smooth_scale: An optional tensor, the smooth scale for x  

* @par Attributes:
* @li round_mode: An optional string.
* The value range is ["rint", "round", "floor"]. Defaults to "rint".
* @li level0_block_size: An optional int, specifying the block size of block quantization.
* Defaults and only supports 512.
* @li level1_block_size: An optional int, specifying the block size of mx quantization.
* Defaults and only supports 32.

* @par Outputs:
* @li y: Quantized output tensor. It has the same shape and rank as input x.
* @li level0_scale: An output tensor of type float. if x is [M,N],level0_scale shape is [M,ceil(N/level0_block_size)]
* @li level1_scale: An output tensor of type FLOAT8_E8M0. Shape needs to meet the following conditions: \n
* - if x is [M,N]
* - axis_change = axis if axis >= 0 else axis + rank(x).
* - level1_scale shape is [M,ceil(N/(level1_block_size * 2)),2]
* - level1_scale tensor is padded with zeros to ensure its size along the quantized axis is even.

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, ONNX, TensorFlow, or PyTorch.
*/
REG_OP(DynamicDualLevelMxQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT4_E2M1}))
    .OUTPUT(level0_scale, TensorType({DT_FLOAT}))
    .OUTPUT(level1_scale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(round_mode, String, "rint")
    .ATTR(level0_block_size, Int, 512)
    .ATTR(level1_block_size, Int, 32)
    .OP_END_FACTORY_REG(DynamicDualLevelMxQuant)
} // namespace ge

#endif // OPS_NN_DEV_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_PROTO_H
