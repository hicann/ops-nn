/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_QUANT_MAX_V2_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_QUANT_MAX_V2_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Quantizes the input tensor x per group (or per tensor when group_list
 *        is absent) and calculates the per-group absolute maximum amax of x.
 *        Static mode takes the reciprocal scale xScale; dynamic mode derives
 *        the scale from amax and additionally outputs y_scale.
 *
 * @par Inputs:
 * @li x: A required tensor of type FLOAT32, FLOAT16 or BFLOAT16, specifying
 *        the input tensor to be quantized. The format must be ND. Shape
 *        supports 1D to 8D. \n
 * @li scale: An optional 1D tensor of type FLOAT32, specifying the reciprocal
 *            quantization scaling factor per group ([1] for per-tensor).
 *            Present = static mode; absent = dynamic mode. \n
 * @li group_list: An optional 1D tensor of type INT64 with shape [G], whose
 *                 element g is the cumulative end row of the first g+1 groups
 *                 along dim-0 (cumsum semantics). Absent = per-tensor. \n
 *
 * @par Attributes:
 * @li round_mode: A required string, specifying the rounding mode for the cast
 *                 operation. Valid values depend on dst_type:
 *                 - For FLOAT8_E5M2 (35) or FLOAT8_E4M3FN (36): only "rint".
 *                 - For HIFLOAT8 (34): "round" or "hybrid". \n
 * @li dst_type: A required int64, specifying the output quantized data type.
 *               Valid values: 34 (HIFLOAT8), 35 (FLOAT8_E5M2),
 *               36 (FLOAT8_E4M3FN). \n
 * @li dst_type_max: An optional float (default 15.0), specifying the target
 *                   type maximum used to derive the dynamic quantization
 *                   scale. Only effective for HIFLOAT8 (34); for
 *                   FLOAT8_E5M2/FLOAT8_E4M3FN the type maximum
 *                   (57344/448) is used regardless. \n
 *
 * @par Outputs:
 * @li y: A required tensor of type HIFLOAT8, FLOAT8_E5M2 or FLOAT8_E4M3FN,
 *        specifying the quantized output. The format must be ND. Shape is the
 *        same as input "x". Data type is determined by dst_type. \n
 * @li amax: A required 1D tensor of type FLOAT32, FLOAT16 or BFLOAT16 with
 *           shape [G] ([1] for per-tensor), specifying the per-group absolute
 *           maximum of input "x". Data type must be the same as input "x". \n
 * @li y_scale: A 1D tensor of type FLOAT32 with shape [G] ([1] for
 *              per-tensor), specifying the output quantization scale that
 *              satisfies x ≈ y × y_scale. Produced only in dynamic mode;
 *              static mode leaves it unconnected. \n
 *
 * @par Third-party framework compatibility
 * It is a custom operator on Ascend NPU.
 */
REG_OP(QuantMaxV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OUTPUT(amax, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(round_mode, String)
    .REQUIRED_ATTR(dst_type, Int)
    .ATTR(dst_type_max, Float, 15.0)
    .OP_END_FACTORY_REG(QuantMaxV2)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_QUANT_MAX_V2_OPS_H_
