/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sigmoid_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SIGMOID_GRAD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SIGMOID_GRAD_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Computes z = (y - y*y)*dy .

* @par Inputs:
* @li y: The input is Tensor, dtype is UnaryDataType.
* @li dy: The input is Tensor, dtype is UnaryDataType.

* @par Attributes:
* @li complex_conj: An optional attribute indicates whether to use conjugate operations for complex dtype. Defaults to "false"

* @par Outputs:
* z: The shape of output, dtype is UnaryDataType.
*/
REG_OP(SigmoidGrad)
    .INPUT(y, TensorType(UnaryDataType))
    .INPUT(dy, TensorType(UnaryDataType))
    .OUTPUT(z, TensorType(UnaryDataType))
    .ATTR(complex_conj, Bool, false)
    .OP_END_FACTORY_REG(SigmoidGrad)

} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_SIGMOID_GRAD_H_
