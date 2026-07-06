/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file hard_shrink_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_HARD_SHRINK_GRAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_HARD_SHRINK_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Calculate the hard shrink grad function.
 *
 * Computes the gradient for the HardShrink: if x > lambd or x < -lambd, x, otherwise 0.
 *
 * @par Inputs:
 * Two inputs, including:
 * @li gradients: A Tensor. Support 1D~8D. Must be one of the following types:
 * float16, float32, bfloat16. The upstream gradient tensor.
 * @li features: A Tensor. Has the same type, format and shape as "gradients".
 * The input tensor of the corresponding HardShrink forward operation.
 *
 * @par Attributes:
 * lambd: An optional float. Threshold of the hard-shrink. Defaults to 0.5.
 *
 * @par Outputs:
 * backprops: A Tensor with the same type, format and shape as "features".
 * The gradient with respect to the input of HardShrink.
 *
 * @par Third-party framework compatibility
 * Compatible with the PyTorch operator hardshrink_backward.
 */
REG_OP(HardShrinkGrad)
    .INPUT(gradients, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(backprops, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(lambd, Float, 0.5)
    .OP_END_FACTORY_REG(HardShrinkGrad)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_HARD_SHRINK_GRAD_OPS_H_
