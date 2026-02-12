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
 * \file log_sigmoid_proto.h
 * \brief
 */
#ifndef LOG_SIGMOID_PROTO_H_
#define LOG_SIGMOID_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Calculate -ln(1+e^(-x)).

*@par Inputs:
*One inputs, including:
* x: A tensor. Must be one of the following types:
*       float16, float32, bfloat16. \n

*@par Outputs:
*One outputs, including:
* y: A tensor with the same type and shape of x's. \n

*@par Third-party framework compatibility
*Compatible with the Pytorch operator LogSigmoid. \n
*/
REG_OP(LogSigmoid)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16})) /* "input:x" */
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))  /* "output:y" */
    .OP_END_FACTORY_REG(LogSigmoid)
} // namespace ge
#endif // LOG_SIGMOID_PROTO_H_