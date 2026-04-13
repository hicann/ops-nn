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
 * \file softsign_proto.h
 * \brief Softsign operator GE IR registration
 */
#ifndef OPS_OP_PROTO_INC_SOFTSIGN_H_
#define OPS_OP_PROTO_INC_SOFTSIGN_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes softsign: x/(abs(x) + 1) .

*@par Inputs:
* One input:
*x: A Tensor. Support 1D ~ 8D. Must be one of the following types: bfloat16, float16, float32 or double.

*@par Outputs:
*y: The activations tensor. Has the same type and format as "x"

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Softsign.
*/
REG_OP(Softsign)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Softsign)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SOFTSIGN_H_
