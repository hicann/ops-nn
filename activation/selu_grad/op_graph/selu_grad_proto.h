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
 * \file selu_grad_proto.h
 * \brief
 */
#ifndef SELU_GRAD_PROTO_H_
#define SELU_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes the gradient of SELU:
 *    y = scale * gradients                          if outputs >= 0
 *    y = gradients * (outputs + scale * alpha)      if outputs < 0
 *
 *    where alpha = 1.6732632423543772848170429916717
 *          scale = 1.0507009873554804934193349852946
 *
 * @par Inputs:
 * Two inputs:
 * gradients: A Tensor. Support 1D ~ 8D. Must be one of the following types: float16, float,
 * bfloat16, int32, int8, uint8. format:ND.
 * outputs: A Tensor. Has the same type, shape and format as "gradients".
 *
 * @par Outputs:
 * y: A Tensor. Has the same type, shape and format as "gradients".
 *
 * @par Third-party framework compatibility
 * @li Compatible with the Pytorch operator selu_backward.
 */
REG_OP(SeluGrad)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(outputs, TensorType::RealNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(SeluGrad)
} // namespace ge
#endif // SELU_GRAD_PROTO_H_
