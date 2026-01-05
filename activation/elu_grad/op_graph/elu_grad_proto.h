/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file elu_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ELU_GRAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ELU_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes gradients for the exponential linear (Elu) operation.
*
* @par Inputs:
* @li grads: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* bfloat16, float16, float32, float64.
* The backpropagated gradients to the corresponding Elu operation.
* @li activations: A tensor. Has the same type, format and shape as "grads".
* The outputs of the corresponding Elu operation.
*
* @par Outputs:
* y: A tensor. Has the same type, format and shape as "grads".
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator EluGrad.
*
*/
REG_OP(EluGrad)
    .INPUT(grads, TensorType({FloatingDataType, DT_BF16}))
    .INPUT(activations, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(EluGrad)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_ELU_GRAD_OPS_H_
