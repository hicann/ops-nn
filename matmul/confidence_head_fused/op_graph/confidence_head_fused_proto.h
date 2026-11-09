/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATMUL_CONFIDENCE_HEAD_FUSED_PROTO_H_
#define OPS_MATMUL_CONFIDENCE_HEAD_FUSED_PROTO_H_

#include <cstdint>
#include <limits>
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Projects hidden and optional markov features with weight, adds optional bias, then applies temperature
 * scaling, optional clamping and sigmoid to produce confidence scores.
 *
 * @par Inputs:
 * @li hidden: A tensor with shape [N, K1] or [B, N, K1]. Supports float16 and bfloat16 in ND format.
 * @li weight: A tensor with shape [K1 + K2], shared by all rows. Has the same type as hidden. Supports ND format.
 * @li markov_emb: An optional tensor with shape [N, K2] or [B, N, K2]. Its leading dimensions and type must match
 * hidden. Supports ND format. If omitted, K2 is 0.
 * @li temperature: An optional float32 tensor with shape [1] in ND format. If omitted or its value is zero,
 * the effective temperature is 1.
 * @li bias: An optional float32 tensor with shape [N] in ND format, broadcast across batches.
 *
 * @par Attributes:
 * @li clamp_low: An optional float. Lower bound for the sigmoid exponent input (negated projected score divided
 * by the effective temperature). The default value is
 * -2147483648.0 and disables the lower bound.
 * @li clamp_high: An optional float. Upper bound for the sigmoid exponent input. The default value is
 * 2147483648.0 and disables the upper bound. The lower bound is applied before the upper bound.
 *
 * @par Outputs:
 * @li out: A float32 tensor in ND format with shape [N] or [B, N], matching the leading dimensions of hidden.
 */
REG_OP(ConfidenceHeadFused)
    .INPUT(hidden, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(markov_emb, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(temperature, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(out, TensorType({DT_FLOAT}))
    .ATTR(clamp_low, Float, -static_cast<float>(std::numeric_limits<int32_t>::max()))
    .ATTR(clamp_high, Float, static_cast<float>(std::numeric_limits<int32_t>::max()))
    .OP_END_FACTORY_REG(ConfidenceHeadFused)

} // namespace ge

#endif // OPS_MATMUL_CONFIDENCE_HEAD_FUSED_PROTO_H_
