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
 * \file softmax_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SOFTMAX_GRAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SOFTMAX_GRAD_OPS_H_

#include "graph/operator_reg.h"
#include "nn_norm.h"

namespace ge{
    /**
*@brief Computes gradients for a softmax operation.

*@par Inputs:
* Two inputs, including:
* @li softmax: A ND tensor. Output of the softmax operator. Must be one of the following
* data types: float16, bfloat16, float32.
* @li grad_softmax: A ND tensor. Has the same shape and data type as "softmax". \n

*@par Attributes:
* axes: An optional list of ints. Multi-axis reduction is supported. Defaults to "{-1}".
* In Ascend 910_95 AI Processor, only single-axis reduction is supported. \n

*@par Outputs:
*grad_x: A ND tensor. Has the same shape and data type as "softmax" . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow operator SoftmaxGrad.
*/
REG_OP(SoftmaxGrad)
    .INPUT(softmax, TensorType({ DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .INPUT(grad_softmax, TensorType({ DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .OUTPUT(grad_x, TensorType({ DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(SoftmaxGrad)
}
#endif