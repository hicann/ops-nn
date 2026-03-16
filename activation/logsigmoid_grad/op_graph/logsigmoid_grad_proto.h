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
 * \file logsigmoid_grad_proto.h
 * \brief
 */
#ifndef LOGSIGMOID_GRAD_PROTO_H_
#define LOGSIGMOID_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Calculate the gradient of log simoid.

*@par Inputs:
*Two inputs, including:
* @li grads: A tensor, gradient of previous layer. Must be one of the following types:
*       float16, float32, bfloat16. \n
* @li features: A tensor with the same type as 'grads', input of log sigmoid. \n
*
*@attention Constraints:
* LogSigmoidGrad supports broadcasting.
*
*@par Outputs:
*One output, including:
* @li backprops: A tensor with the same type and shape as 'grads'. \n

*@par Third-party framework compatibility
*Compatible with the Pytorch operator LogSigmoidBackward. \n
*/
REG_OP(LogSigmoidGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(backprops, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(LogSigmoidGrad)
} // namespace ge
#endif // LOG_SIGMOID_PROTO_H_