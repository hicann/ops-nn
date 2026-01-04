/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_ASCEND_QUANT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ASCEND_QUANT_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Quantizes the input.

* @par Inputs:
* x: A tensor of type float16 or float32, specifying the input.
* The format must be NC1HWC0, FRACTAL_NZ, NDC1HWC0 or ND. Shape supports 1D ~ 8D.
* If "dst_type" is 29, the last dimension of the shape must be divisible by 2. \n

* @par Attributes:
* @li scale: A required float32, specifying the scaling ratio.
* @li offset: A required float32, specifying the offset.
* @li sqrt_mode: An optional bool, specifying whether to perform square on "scale", either "True" or "False".
* Defaults to "False".
* @li round_mode: An optional string, specifying the cast mode.
* The value range is [Round, Floor, Ceil, Trunc, Hybrid]. Defaults to "Round".
* @li dst_type: An optional int32, specifying the output data type.
* Defaults to "2", represents dtype "DT_INT8". "29" represents dtype "DT_INT4", "34" represents dtype "DT_HIFLOAT8",
"35" represents dtype "DT_FLOAT8_E5M2", "36" represents dtype "DT_FLOAT8_E4M3FN". \n

* @par Outputs:
* y: The quantized output tensor of type int8, int4, hifloat8, float8_e5m2 or float8_e4m3fn.
* The format must be NC1HWC0, FRACTAL_NZ, NDC1HWC0 or ND. Shape supports 1D ~ 8D.
* Has the same format and shape as input "x". \n

* @attention Constraints:
* @li round_mode value range is [Round, Floor, Ceil, Trunc, Hybrid]. \n
* Round: round to nearest, tie to even(c language rint). \n
* Floor: round to minus infinity(c language floor). \n
* Ceil: round to positive infinity(c language ceil). \n
* Trunc: round to zero(c language trunc). \n
* Hybrid: only valid when output dtype is hifloat8. \n
* The following constraints apply to products other than Ascend 910_95 AI Processor: \n
* @li When format is FRACTAL_NZ, shape supports 4D ~ 8D.
* @li When "x" is dynamic shape, shape [-2] is not supported.
* @li When "x" is dynamic shape, the data type of output "y" does not support int4.
* @li When the format of "x" is ND, the data type of output "y" does not support int4. \n

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*/
REG_OP(AscendQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .REQUIRED_ATTR(scale, Float)
    .REQUIRED_ATTR(offset, Float)
    .ATTR(sqrt_mode, Bool, false)
    .ATTR(round_mode, String, "Round")
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(AscendQuant)
} // namespace ge
#endif