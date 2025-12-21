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
 * \file hard_swish_grad_v2_proto.h
 * \brief
 */

#ifndef OPS_ACTIVATION_HARD_SWISH_GRAD_V2_GRAPH_PLUGIN_HARD_SWISH_GRAD_V2_PROTO_H_
#define OPS_ACTIVATION_HARD_SWISH_GRAD_V2_GRAPH_PLUGIN_HARD_SWISH_GRAD_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge{
/**
*@brief Computes the gradient for the hard_swish of "self" .
*calculating formula:
*@code{.c}
*gradSelf_{i} = begin{cases}
* 0, self_{i} <= -3, 
* self_{i} / 3 + 0.5, -3 < self_{i} < 3,
* 1, self_{i} >= 3
* end{cases}
*
* out_{i} = gradOutput_{i} times gradSelf_{i}
*@endcode
*
*@par Inputs:
*Two inputs, including:
* @li gradOutput: A tensor. Must be one of the following types: float16, float32, bfloat16
* @li self: A tensor with the same type as "gradOutput" . \n
*@par Outputs:
* out: A tensor with the same type as "gradOutput".
*@par Third-party framework compatibility
* Compatible with the PyTorch operator HardSwishGrad after v2.8.
*/
REG_OP(HardSwishGradV2)
    .INPUT(gradOutput, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(self, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(out, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(HardSwishGradV2)
} // namespace ge
#endif  // OPS_ACTIVATION_HARD_SWISH_GRAD_V2_GRAPH_PLUGIN_HARD_SWISH_GRAD_V2_PROTO_H_
