/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_layer_norm_quant_proto.h
 * \brief
 */
#ifndef OPS_NORM_ADD_LAYER_NORM_QUANT_PROTO_H_
#define OPS_NORM_ADD_LAYER_NORM_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fused Operator of AddLayerNorm and Quantize/DynamicQuant/AscendQuantV2. \n
*  calculating: x1, x2, gamma, beta, bias, scales1, scales2, zero_points1, zero_points2 \n
*  x = x1 + x2 + bias \n
*  rstd = 1 / (sqrt(Var(x) + eps)) \n
*  y = (x - E(x)) * rstd * gamma + beta \n
*  when quant_mode = "static" and div_mode=true, output out_scales1 and out_scales2 are invalid: \n
*  y1 = round(y / scales1) + zero_points1 \n
*  y2 = round(y / scales2) + zero_points2 \n
*  when quant_mode = "static" and div_mode=false, output out_scales1 and out_scales2 are invalid: \n
*  y1 = round(y * scales1) + zero_points1 \n
*  y2 = round(y * scales2) + zero_points2 \n
*  when quant_mode = "dynamic": \n
*  tmp1 = y * scales1 \n
*  tmp2 = y * scales2 \n
*  out_scales1 = reduce_max(abs(tmp1))/127 \n
*  out_scales2 = reduce_max(abs(tmp2))/127 \n
*  y1 = round(tmp1 / out_scales1) \n
*  y2 = round(tmp2 / out_scales2) \n

* @par Inputs
* @li x1: A tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li x2: A tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li gamma: A tensor for layer norm weight params. Support dtype: float32, float16, bfloat16, support format: ND.
* @li beta: A tensor for layer norm weight params. Support dtype: float32, float16, bfloat16, support format: ND.
* @li bias: An optional input tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li scales1: An optional input tensor for one of quant scale. Support dtype: float32, float16, bfloat16, support
format: ND.
* @li scales2: An optional input tensor for another quant scale. Support dtype: float32, float16, bfloat16, support
format: ND.
* @li zero_points1: An optional input tensor for one of quant offset. Support dtype: float32, float16, bfloat16,
support format: ND.
* @li zero_points2: An optional input tensor for another quant offset. Support dtype: float32, float16, bfloat16,
support format: ND.

* @par Attributes
* @li quant_mode: An optional attribute utilized to select quant mode, can be "dynamic" or "static", the type is
string. Defaults to "dynamic".
* @li epsilon: An optional attribute for layer norm compute, the type is float. Defaults to 1e-5.
* @li additional_output: An optional attribute control whether output x valid or invalid, the type is bool. Defaults
to false, which means x output is invalid.
* @li div_mode: An optional attribute control static quant algorithm, the type is bool. Defaults
to true, which means scales while be divided by normlization output.

* @par Outputs
* @li y1: Quantize result 1.
*     A tensor. Support dtype: int8, support format: ND.
* @li y2: Quantize result 2.
*     A tensor. Support dtype: int8, support format: ND.
* @li x: Describing the result of x1 + x2 + bias.
*     A tensor. Support dtype: float32, float16, bfloat16, support format: ND.
* @li out_scales1: Describing the result of dynamic quantize scales computed via scales1.
*     A tensor. Support dtype: float32, support format: ND.
* @li out_scales2: Describing the result of dynamic quantize scales computed via scales2.
*     A tensor. Support dtype: float32, support format: ND.
*/
REG_OP(AddLayerNormQuant)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(scales1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(scales2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y1, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(y2, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(out_scales1, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(out_scales2, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(quant_mode, String, "dynamic")
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .ATTR(div_mode, Bool, true)
    .OP_END_FACTORY_REG(AddLayerNormQuant)

} // namespace ge

#endif // OPS_NORM_ADD_LAYER_NORM_QUANT_PROTO_H_