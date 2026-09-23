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
 * \file swiglu_group_quant_with_dual_axis_proto.h
 * \brief Operator prototype definition for SwigluGroupQuantWithDualAxis.
 */

#ifndef OPS_NN_SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_PROTO_H
#define OPS_NN_SWIGLU_GROUP_QUANT_WITH_DUAL_AXIS_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Fuses clipped SwiGLU with ordinary and grouped MX quantization.
 *
 * x must be a two-dimensional [T, 2H] tensor.
 * group_index is an optional one-dimensional INT64 cumsum vector. It only
 * defines route-2 group boundaries; SwiGLU and route 1 always process all rows.
 * When present, endpoints are non-negative and non-decreasing, and the final
 * endpoint is T. Repeated endpoints represent empty groups.
 * y_origin contains the clipped SwiGLU activation before applying weight and
 * has shape [T, H] when output_origin is true; otherwise it is a [0] placeholder.
 */
REG_OP(SwigluGroupQuantWithDualAxis)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OUTPUT(y1, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(mxscale1, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y2, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(mxscale2, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y_origin, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(dst_type, Int, DT_FLOAT8_E4M3FN)
    .ATTR(quant_mode, Int, 1)
    .ATTR(clamp_limit, Float, -1.0f)
    .ATTR(output_origin, Bool, false)
    .ATTR(alpha, Float, 1.0f)
    .ATTR(bias, Float, 0.0f)
    .OP_END_FACTORY_REG(SwigluGroupQuantWithDualAxis)
} // namespace ge

#endif
