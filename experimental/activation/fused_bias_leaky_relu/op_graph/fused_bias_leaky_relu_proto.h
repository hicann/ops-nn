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
 * \file fused_bias_leaky_relu_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSED_BIAS_LEAKY_RELU_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSED_BIAS_LEAKY_RELU_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the output as scale * (x + bias) if x+bias > 0 and scale * negative_slope * (x+bias)
* if x+bias <= 0 . \n

* @par Inputs:
* Two input:
* x: A Tensor. Must be one of the following types: float32, float16.
* bias: A Tensor. Must be one of the following types: float32, float16.
*
* @par Attributes:
* negative_slope: A float32. Defaults to "0.2".
* scale: A float32. Defaults to "1.414213562373f".
*
* @par Outputs:
* y: A Tensor. Has the same type as "x".
* @par Third-party framework compatibility
* Compatible with the mmcv operator FusedBiasLeakyrelu.
*/
REG_OP(FusedBiasLeakyRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(negative_slope, Float, 0.2f)
    .ATTR(scale, Float, 1.414213562373f)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(FusedBiasLeakyRelu)

} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSED_BIAS_LEAKY_RELU_OPS_H_
