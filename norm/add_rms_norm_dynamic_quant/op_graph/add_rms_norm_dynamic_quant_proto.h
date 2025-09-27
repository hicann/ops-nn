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
 * \file add_rms_norm_dynamic_quant_proto.h
 * \brief
 */
#ifndef OPS_NORM_ADD_RMS_NORM_DYNAMIC_QUANT_PROTO_H_
#define OPS_NORM_ADD_RMS_NORM_DYNAMIC_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fused Operator of Add, RmsNorm and DynamicQuant.
* Calculating input: x1, x2, gamma, smooth_scale1, smooth_scale2 \n
* Calculating process: \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  rmsnorm_out = x * rstd * gamma \n
*  if smooth_scales1 exist: \n
*    scale1 = row_max(abs(rmsnorm_out * smooth_scale1)) / 127 \n
*  if smooth_scales1 not exist: \n
*    scale1 = row_max(abs(rmsnorm_out)) / 127 \n
*  y1 = round(rmsnorm_out / scale1) \n
*  if smooth_scales2 exist: \n
*    scale2 = row_max(abs(rmsnorm_out * smooth_scale2)) / 127 \n
*    y2 = round(rmsnorm_out / scale2) \n
*  if smooth_scales2 not exist:  \n
*    not calculate scale2 and y2. \n

* @par Inputs
* @li x1: A tensor. Input x1 for the add operation.
*         Support dtype: float16/bfloat16, support format: ND.
* @li x2: A tensor. Input x2 for the add operation.
*         Support dtype: float16/bfloat16, support format: ND.
* @li gamma: A tensor. Describing the weight of the rmsnorm operation.
*            Support dtype: float16/bfloat16, support format: ND.
* @li smooth_scale1: A tensor. Describing the weight of the first dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND.
* @li smooth_scale2: An optional input tensor. Describing the weight of the secend dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND.
* @li beta: An optional input tensor. Describing the offset value of dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND. Has the same dtype and shape as "gamma".
* @par Attributes
* epsilon: An optional attribute. Describing the epsilon of the rmsnorm operation.
*          The type is float. Defaults to 1e-6.

* @par Outputs
* @li y1: A tensor. Describing the output of the first dynamic quantization.
*                   Support dtype: int8, support format: ND.
* @li y2: A tensor. Describing the output of the second dynamic quantization.
*                   Support dtype: int8, support format: ND.
* @li x: A tensor. Describing the output of the x1+x2 add operation.
*                  Support dtype: float16/bfloat16, support format: ND.
* @li scale1: A tensor. Describing of the factor for the first dynamic quantization.
*                  Support dtype: float32, support format: ND.
* @li scale2: A tensor. Describing of the factor for the second dynamic quantization.
*                  Support dtype: float32, support format: ND.
*/
REG_OP(AddRmsNormDynamicQuant)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8, DT_INT8}))
    .OUTPUT(y2, TensorType({DT_INT8, DT_INT8}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(scale1, TensorType({DT_FLOAT, DT_FLOAT}))
    .OUTPUT(scale2, TensorType({DT_FLOAT, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6)
    .ATTR(output_mask, ListBool, {})
    .OP_END_FACTORY_REG(AddRmsNormDynamicQuant)
} // namespace ge

#endif // OPS_NORM_ADD_RMS_NORM_DYNAMIC_QUANT_PROTO_H_