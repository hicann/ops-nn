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
 * \file relu_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes rectified linear gradients for a ReLU operation.
*  Gradients/features support broadcasting operations. \n
*
* @par Inputs:
* Two inputs, including:
* @li gradients: A tensor of type RealNumberType. The backpropagated gradients to the corresponding Relu operation .
* @li features: A tensor of type RealNumberType. The features passed as input to the corresponding Relu operation . \n
*
* @par Outputs:
* backprops: A tensor of type RealNumberType . \n
*
* @attention Constraints:
* The corresponding Relu operator needs to be called before using this operator on the network . \n
*
* @par Third-party framework compatibility
* Compatible with TensorFlow operator ReluGrad.
*/
REG_OP(ReluGrad)
    .INPUT(gradients, TensorType::RealNumberType())
    .INPUT(features, TensorType::RealNumberType())
    .OUTPUT(backprops, TensorType::RealNumberType())
    .OP_END_FACTORY_REG(ReluGrad)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_