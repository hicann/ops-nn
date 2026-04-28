/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_QUANT_MAX_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_QUANT_MAX_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Quantizes the input tensor x with the given scale and calculates the
 *        absolute maximum value (amax) of x.
 *
 * @par Inputs:
 * @li x: A required tensor of type FLOAT32, FLOAT16 or BFLOAT16, specifying the input tensor
 *        to be quantized. The format must be ND. Shape supports 1D to 8D. \n
 * @li scale: A required 1D tensor of type FLOAT32 with shape [1], specifying the
 *            quantization scaling factor. \n
 *
 * @par Attributes:
 * @li round_mode: An optional string, specifying the rounding mode for the cast operation.
 *                 Default value: "rint". Valid values depend on dst_type:
 *                 - For FLOAT8_E5M2 (35) or FLOAT8_E4M3FN (36): only "rint" is supported.
 *                 - For HIFLOAT8 (34): "round" or "hybrid" is supported. \n
 * @li dst_type: An optional int64, specifying the output quantized data type.
 *               Default value: 35 (FLOAT8_E5M2).
 *               Valid values: 34 (HIFLOAT8), 35 (FLOAT8_E5M2), 36 (FLOAT8_E4M3FN). \n
 *
 * @par Outputs:
 * @li y: A required tensor of type HIFLOAT8, FLOAT8_E5M2 or FLOAT8_E4M3FN,
 *        specifying the quantized output. The format must be ND.
 *        Shape is the same as input "x". Data type is determined by dst_type. \n
 * @li amax: A required 1D tensor of type FLOAT32, FLOAT16 or BFLOAT16 with shape [1],
 *           specifying the absolute maximum value of input "x".
 *           Data type must be the same as input "x". \n
 *
 * @par Third-party framework compatibility
 * It is a custom operator on Ascend NPU.
 */
REG_OP(QuantMax)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OUTPUT(amax, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(round_mode, String, "rint")
    .ATTR(dst_type, Int, DT_FLOAT8_E5M2)
    .OP_END_FACTORY_REG(QuantMax)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_QUANT_MAX_OPS_H_
