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
 * \file nonlinear_fuc_ops.h
 * \brief
 */
#ifndef OPS_ACTIVATION_SWISH_OPS_H_
#define OPS_ACTIVATION_SWISH_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes the for the Swish of "x" .

*@par Inputs:
*One input, including:
* x: A tensor, which supports 1D-8D defaultly and must be one of the following types: float16, bfloat16, float32. \n

*@par Outputs:
* y: A tensor of the same type, shape and format as "x", and y = x / (1 + e ^ (-scale * x)). \n

*@par Attributes:
* scale: scalar parameter, the multiplier of x. Must be one of the following types: float. Default value = 1.0. \n

*@par Third-party framework compatibility
*Compatible with the Torch operator Swish
*/
REG_OP(Swish)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(scale, Float, 1.0)
    .OP_END_FACTORY_REG(Swish)
} // namespace ge
#endif  // OPS_ACTIVATION_SWISH_OPS_H_