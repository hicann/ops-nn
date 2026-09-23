/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_ACTIVATION_SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_PROTO_H_
#define OPS_ACTIVATION_SWIGLU_BACKWARD_GROUP_QUANT_WITH_DUAL_AXIS_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(SwigluBackwardGroupQuantWithDualAxis)
    .INPUT(grad_y, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(y_origin, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OUTPUT(y1, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(scale1, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y2, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(scale2, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(grad_weight, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(clamp_limit, Float, -1.0f)
    .ATTR(alpha, Float, 1.0f)
    .ATTR(bias, Float, 0.0f)
    .ATTR(quant_mode, Int, 1)
    .ATTR(dst_type, Int, 36)
    .OP_END_FACTORY_REG(SwigluBackwardGroupQuantWithDualAxis)
} // namespace ge
#endif
