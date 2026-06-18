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
 * \file fake_quant_with_min_max_vars_gradient_proto.h
 * \brief IR operator definition for FakeQuantWithMinMaxVarsGradient
 *
 * 4 inputs: gradients(float32), x(float32), min(float32,shape(1,)), max(float32,shape(1,))
 * 3 outputs: backprops_wrt_x(float32), backprop_wrt_min(float32,shape(1,)), backprop_wrt_max(float32,shape(1,))
 * 2 attrs: num_bits(int,[2,16],default=8), narrow_range(bool,default=false)
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_H_
#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Gradient of FakeQuantWithMinMaxVars for quantization-aware training.
 *        Computes backprops_wrt_x, backprop_wrt_min, backprop_wrt_max.
 *
 * @par Inputs:
 * @li gradients: A tensor of type float32. Backpropagated gradients.
 * @li x: A tensor of type float32, same shape as gradients.
 * @li min: A tensor of type float32, shape (1,). Quantization lower bound.
 * @li max: A tensor of type float32, shape (1,). Quantization upper bound.
 *
 * @par Attributes:
 * @li num_bits: (Optional) Int, default 8. Quantization bits, range [2, 16].
 * @li narrow_range: (Optional) Bool, default false. Use [1, 2^num_bits-1] when true.
 *
 * @par Outputs:
 * @li backprops_wrt_x: A tensor of type float32, same shape as x.
 * @li backprop_wrt_min: A tensor of type float32, shape (1,).
 * @li backprop_wrt_max: A tensor of type float32, shape (1,).
 *
 * @par Third-party framework compatibility:
 * Aligned with TensorFlow `tf.raw_ops.FakeQuantWithMinMaxVarsGradient`.
 */
REG_OP(FakeQuantWithMinMaxVarsGradient)
    .INPUT(gradients, TensorType({DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(min, TensorType({DT_FLOAT}))
    .INPUT(max, TensorType({DT_FLOAT}))
    .OUTPUT(backprops_wrt_x, TensorType({DT_FLOAT}))
    .OUTPUT(backprop_wrt_min, TensorType({DT_FLOAT}))
    .OUTPUT(backprop_wrt_max, TensorType({DT_FLOAT}))
    .ATTR(num_bits, Int, 8)
    .ATTR(narrow_range, Bool, false)
    .OP_END_FACTORY_REG(FakeQuantWithMinMaxVarsGradient)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_H_
