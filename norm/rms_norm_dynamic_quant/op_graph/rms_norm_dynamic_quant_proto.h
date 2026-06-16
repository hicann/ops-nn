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
 * \file rms_norm_dynamic_quant_proto.h
 * \brief
 */
#ifndef OPS_RMS_NORM_DYNAMIC_QUANT_PROTO_H_
#define OPS_RMS_NORM_DYNAMIC_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fused Operator of RmsNorm and DynamicQuant.
* Calculating input: x, gamma, smooth_scales \n
* Calculating process: \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  rmsnorm_out = x * rstd * gamma \n
*  if smooth_scales exist: \n
*    input = rmsnorm_out * smooth_scales \n
*    scale = row_max(abs(input)) / max_val \n
*  if smooth_scales not exist: \n
*    scale = row_max(abs(rmsnorm_out)) / max_val \n
*  y = round(input / scale) \n

* @par Inputs
* @li x: A tensor. Input x for the operation.
*         Support dtype: float16/bfloat16, support format: ND.
* @li gamma: A tensor. Describing the weight of the rmsnorm operation.
*            Support dtype: float16/bfloat16, support format: ND.
* @li smooth_scales: An optional input tensor. Describing the weight of dynamic quantization.
*              Support dtype: float16/bfloat16, support format: ND.
* @li beta: An optional input tensor. Describing the offset value of rmsnorm operation.
*              Support dtype: float16/bfloat16, support format: ND. Has the same dtype and shape as "gamma".
* @par Attributes
* @li epsilon: An optional attribute. Describing the epsilon of the rmsnorm operation.
*          The type is float. Defaults to 1e-6.
* @li dst_type: An optional int32. Output y data type enum value. Support DT_INT8.
*               Defaults to DT_INT8.

* @par Outputs
* @li y: A tensor. Describing the output of dynamic quantization.
*                   Support dtype: int8, support format: ND.
* @li scale: A tensor. Describing of the factor for dynamic quantization.
*                  Support dtype: float32, support format: ND.
*/

REG_OP(RmsNormDynamicQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6)
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(RmsNormDynamicQuant)

} // namespace ge

#endif // OPS_RMS_NORM_DYNAMIC_QUANT_PROTO_H_
