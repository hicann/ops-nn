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
 * \file fake_quant_with_min_max_vars_proto.h
 * \brief GE IR operator registration for FakeQuantWithMinMaxVars
 */

#ifndef OPS_OP_PROTO_INC_FakeQuantWithMinMaxVars_H_
#define OPS_OP_PROTO_INC_FakeQuantWithMinMaxVars_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Per-tensor fake quantization for QAT training.
 *
 * Simulates quantize-dequantize: computes scale and nudged zero_point from
 * min/max bounds, clamps and rounds input, then rescales back to float32.
 * When min == max, outputs all zeros.
 *
 * @par Inputs:
 * @li x:   A tensor of type float32, rank 1~8.
 * @li min: A scalar tensor of type float32, shape=[1].
 * @li max: A scalar tensor of type float32, shape=[1].
 *
 * @par Attributes:
 * @li num_bits:     int64, range [2,16], default 8.
 * @li narrow_range: bool, default false.
 *
 * @par Outputs:
 * @li y: A tensor of type float32, same shape as x.
 */
REG_OP(FakeQuantWithMinMaxVars)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(min, TensorType({DT_FLOAT}))
    .INPUT(max, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxVars)
} // namespace ge

#endif // OPS_OP_PROTO_INC_FakeQuantWithMinMaxVars_H_