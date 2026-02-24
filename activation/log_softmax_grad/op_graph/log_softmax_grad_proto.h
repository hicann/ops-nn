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
 * \file log_softmax_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_LOG_SOFTMAX_GRAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_LOG_SOFTMAX_GRAD_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
*@brief Computes the gradient for log softmax activations.

*@par Inputs:
* @li grad: A ND tensor. Must be one of the following data types: float16, bfloat16, float32.
* @li x: A ND tensor. Has the same data type and shape as "grad". \n

*@par Attributes:
* axis: An optional list of ints. Multi-axis reduction is supported. Defaults to "{-1}".
* In Ascend 950 AI Processor, only single-axis reduction is supported. \n

*@par Outputs:
* y: A ND tensor. Has the same data type and shape as "grad". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogSoftmaxGrad.
*/

REG_OP(LogSoftmaxGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(axis, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxGrad)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_LOG_SOFTMAX_GRAD_OPS_H_
