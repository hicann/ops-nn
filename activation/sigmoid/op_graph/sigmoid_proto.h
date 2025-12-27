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
 * \file sigmoid_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NONLINEAR_FUC_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Compute sigmoid of "x" element-wise .

* @par Inputs:
* A Tensor of type complex64, complex128, bfloat16, float16, float32 or double . \n

* @par Outputs:
* A Tensor. Has the same type as "x" . \n

* @see Relu()

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Sigmoid.
*/
REG_OP(Sigmoid)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Sigmoid)
}
#endif