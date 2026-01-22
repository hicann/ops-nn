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
 * \file glu_proto.h
 * \brief
 */
#ifndef OPS_ACTIVATION_GLU_GRAPH_PLUGIN_GLU_PROTO_H_
#define OPS_ACTIVATION_GLU_GRAPH_PLUGIN_GLU_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Activation function called Gated Linear Unit, calculate result by input ND tensor x and integer dim.
*
* @par Inputs:
* x: ND input tensor. Must be one of the following types: float16, float32, bfloat16.
*
* @par Attributes:
* dim: the dimension will be chunked into halves, default to be -1, the dimension itself must be even.
*
* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Third-party framework compatibility
* Compatible with caffe correlation custom operator.
*/

REG_OP(GLU)
      .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
      .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
      .ATTR(dim, Int, -1)
      .OP_END_FACTORY_REG(GLU)

} // namespace ge

#endif