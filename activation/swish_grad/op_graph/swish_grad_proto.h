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
 * \file swish_grad_proto.h
 * \brief swish_grad_proto
 */
#ifndef SWISH_GRAD_PROTO_H_
#define SWISH_GRAD_PROTO_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes the gradient for the Swish of "x" .

*@par Inputs:
*Three inputs, including:
* @li grad: A tensor, which supports 1D-8D defaultly. Format support ND, NC1HWC0, FRACTAL_NZ and must be one of the
* following types: float16, bfloat16, float32.
* @li x: A tensor of the same type, shape and format as "grad".
* @li y: A tensor of the same type, shape and format as "grad", and y = x / (1 + e ^ (-scale * x)). \n
* @par Attributes:
* scale: An optional scalar, the multiplier of x. The data type is float, default value = 1.0. \n
*@par Outputs:
*grad_x: A tensor, which is the gradient of x. Has the same type, shape and format as "grad".
*@par Third-party framework compatibility
*Compatible with the PyTorch operator SwishGrad.
*/
REG_OP(SwishGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(scale, Float, 1.0)
    .OP_END_FACTORY_REG(SwishGrad)
} // namespace ge

#endif // SWISH_GRAD_PROTO_H_