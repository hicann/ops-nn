/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_ACTIVATION_GLU_GRAD_GRAPH_PLUGIN_GLU_GRAD_PROTO_H_
#define OPS_ACTIVATION_GLU_GRAD_GRAPH_PLUGIN_GLU_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Calculates the backward outputs of the function "GLU".

* @par Inputs:
* @li y_grad: A Tensor. Must be one of the following types: float16, float32.
* @li x: A Tensor of the same type as `y_grad`, but with a size that is twice as large as `y_grad` along the axis `dim`.

* @par Attributes:
* dim: An optional int, specifying the dimension along which the GLU is performed.
* It should be in the range [-rank(x), rank(x)). Defaults to -1.
* @par Outputs:
* x_grad: A Tensor of the same type as `y_grad` and of the same shape as `x`.

* @par Third-party framework compatibility
* Compatible with the Pytorch operator GLUGrad.
*/

REG_OP(GLUGrad)
      .INPUT(y_grad, TensorType::FloatingDataType())
      .INPUT(x, TensorType::FloatingDataType())
      .OUTPUT(x_grad, TensorType::FloatingDataType())
      .ATTR(dim, Int, -1)
      .OP_END_FACTORY_REG(GLUGrad)
}

#endif
