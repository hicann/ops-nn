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
 * \file matrix_calculation_ops.h
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
* x: A Tensor. TensorType::FloatingDataType() or bfloat16. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x". \n

* @attention Constraints:
* if performances better in format NZ, please close
* "MatmulTransdataFusionPass" in fusion configuration. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator L2Loss.
*/
REG_OP(L2Loss)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(L2Loss)
}
#endif // L2_LOSS_OP_PROTO_H_