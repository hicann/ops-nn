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
 * \file add_rms_norm_dynamic_quant_v2_proto.h
 * \brief
 */
#ifndef OPS_NORM_ADD_RMS_NORM_DYNAMIC_QUANT_V2_PROTO_H_
#define OPS_NORM_ADD_RMS_NORM_DYNAMIC_QUANT_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fused Operator of AddRmsNorm and DynamicQuant . \n

* @par Inputs:
* @li x1: A tensor of type float16/bfloat16. Supported format "ND". \n
* @li x2: A tensor of type float16/bfloat16. Supported format "ND". \n
* @li gamma: A tensor of type float16/bfloat16. Supported format "ND". \n
* @li smooth_scale1: Optional Input. A tensor of type float16/bfloat16. Supported format "ND". \n
* @li smooth_scale2: Optional Input. A tensor of type float16/bfloat16. Supported format "ND". \n

* @par Attributes:
* epsilon: An optional float, default value is 1e-6.

* @par Outputs:
* @li y1: A tensor of type int8, quantize result for rmsnorm(x1+x2)*smooth1. Supported format "ND". \n
* @li y2: A tensor of type int8, quantize result for rmsnorm(x1+x2)*smooth2. Supported format "ND". \n
* @li y3: A tensor of type float32, cast result for rmsnorm(x1+x2). Supported format "ND". \n
* @li y4: A tensor of type float16/bfloat16, describe the result for rmsnorm(x1+x2). Supported format "ND". \n
* @li x: A tensor of type float16/bfloat16, describing the result of x1 + x2. Supported format "ND". \n
* @li scale1: A tensor of type float32, describing the result of dynamic quantize scales. Supported format "ND". \n
* @li scale2: A tensor of type float32, describing the result of dynamic quantize scales. Supported format "ND". \n
*/
REG_OP(AddRmsNormDynamicQuantV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8}))
    .OUTPUT(y2, TensorType({DT_INT8}))
    .OUTPUT(y3, TensorType({DT_FP32}))
    .OUTPUT(y4, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(scale1, TensorType({DT_FLOAT}))
    .OUTPUT(scale2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6)
    .OP_END_FACTORY_REG(AddRmsNormDynamicQuantV2)
} // namespace ge

#endif // OPS_NORM_ADD_RMS_NORM_DYNAMIC_QUANT_V2_PROTO_H_