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
 * \file hard_sigmoid_proto.h
 * \brief HardSigmoid 图模式算子原型定义
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_HARD_SIGMOID_H_
#define OPS_BUILT_IN_OP_PROTO_INC_HARD_SIGMOID_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Calculate the hard sigmoid function: output_y = max(0, min(1, alpha * input_x + beta)).
 *
 * @par Inputs:
 * input_x: An ND tensor. The shape should be within the range of 0D to 8D.
 * Must be one of the following types: float16, float32, int32, bfloat16.
 *
 * @par Attributes:
 * @li alpha: An optional float. Slope of the operator, defaults to 0.16666666.
 * @li beta: An optional float. Offset of the operator, defaults to 0.5.
 *
 * @par Outputs:
 * output_y: An ND tensor with the same dtype and shape as "input_x".
 *
 * @par Third-party framework compatibility
 * Compatible with the PyTorch operator torch.nn.Hardsigmoid.
 */
REG_OP(HardSigmoid)
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT32}))
    .OUTPUT(output_y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT32}))
    .ATTR(alpha, Float, 0.16666666)
    .ATTR(beta, Float, 0.5)
    .OP_END_FACTORY_REG(HardSigmoid)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_HARD_SIGMOID_H_
