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
 * \file l2_loss_proto.h
 * \brief
 */
#ifndef L2_LOSS_OP_PROTO_H_
#define L2_LOSS_OP_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Computes half the L2 norm of a tensor without the sqrt .

* @par Inputs:
* x: A Tensor. TensorType::FloatingDataType() or bfloat16. Format support ND. \n

* @par Outputs:
* y: A 0-D (scalar) Tensor. Has the same type and format as "x". \n

* @attention Constraints:
* The output "y" is always a 0-D scalar regardless of the shape of "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator L2Loss.
*/
REG_OP(L2Loss)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(L2Loss)
} // namespace ge
#endif // L2_LOSS_OP_PROTO_H_
