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
 * \file fake_quant_with_min_max_args_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FAKE_QUANT_WITH_MIN_MAX_ARGS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FAKE_QUANT_WITH_MIN_MAX_ARGS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fake quantizes the 'inputs' tensor of type float to 'outputs' tensor of same type and shape.
*        y = dequant(quant(clip(x, nudgedMin, nudgedMax)))

* @par Inputs:
* x: A tensor of type float32. Values to be fake-quantized.

* @par Attributes:
* @li min: (Optional) Float, default -6.0. Lower quantization bound. Must satisfy min < max.
* @li max: (Optional) Float, default 6.0. Upper quantization bound.
* @li num_bits: (Optional) Int, default 8. Number of quantization bits, range [2, 16].
* @li narrow_range: (Optional) Bool, default false. Use [1, 2^num_bits-1] range when true,
*                   else [0, 2^num_bits-1].

* @par Outputs:
* y: A tensor of type float32, same shape and dtype as x. Fake-quantized result.

* @par Third-party framework compatibility:
* Aligned with TensorFlow `tf.quantization.fake_quant_with_min_max_args`.
*/
REG_OP(FakeQuantWithMinMaxArgs)
    .INPUT(x, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(min, Float, -6.0f)
    .ATTR(max, Float, 6.0f)
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxArgs)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_FAKE_QUANT_WITH_MIN_MAX_ARGS_H_
