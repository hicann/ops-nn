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
 * \file selu_proto.h
 * \brief
 */
#ifndef SELU_PROTO_H_
#define SELU_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes scaled exponential linear:
*    y = scale * [max(0, x) + min(0, alpha * (exp(x) - 1))]
*
*    where alpha = 1.6732632423543772848170429916717
*          scale = 1.0507009873554804934193349852946
*
* @par Inputs:
* One input:
* x: A Tensor. Support 1D ~ 8D. Must be one of the following types: float16, float,
* int32, int8, bfloat16. format:ND.
*
* @par Outputs:
* y: A Tensor. Has the same type, shape and format as input "x". format:ND.
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Selu.
* @li Compatible with Pytorch's selu Operator.
*/
REG_OP(Selu)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT,DT_DOUBLE, DT_INT8, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(Selu)
} // namespace ge
#endif // SELU_PROTO_H_
