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
 * \file ge_glu_grad_v2_proto.h
 * \brief
 */
#ifndef OPS_ACTIVATION_GE_GLU_GRAD_V2_GRAPH_PLUGIN_GE_GLU_GRAD_V2_PROTO_H_
#define OPS_ACTIVATION_GE_GLU_GRAD_V2_GRAPH_PLUGIN_GE_GLU_GRAD_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the gradient of x through GeGluV2.
*
* @par Inputs:
* Three inputs, including:
* @li dy: A Tensor of the same type as "x".
      The shape of dy matches the shape of x in all dimensions except for the split dimension,
      where its length is half of the length of x's split dimension.
* @li x: A Tensor. Must be one of the following types: float16, bfloat16, float32.
      Shape supports at least 1 dimension, and at most 8 dimensions.
      The length of the split dimension in x must be an even number.
* @li gelu: A Tensor of the same type as "x".
      The shape of gelu matches the shape of x in all dimensions except for the split dimension,
      where its length is half of the length of x's split dimension.
*
* @par Outputs:
* dx: A Tensor. Has the same type as "x". Shape should be same as x.
*
* @par Attributes:
* @li dim: An optional Int. The dimension to be split, default is -1.
* @li approximate: An optional Int. Determines which formula to use for the activation computation.
* The gelu grad approximation algorithm to use: 0('none') or 1('tanh'), default is 1('tanh').
* Atlas Inference Series Product only supports 'tanh'(1).
* @li activate_left: An optional Bool.
* Whether the left side of x is used as an input parameter to the activation function,
* default is false, use the right side.
*
* @par Third-party framework compatibility
* Compatible with the Pytorch operator GeGluGradV2.
*
*/
REG_OP(GeGluGradV2)
    .INPUT(dy, "T")
    .INPUT(x, "T")
    .INPUT(gelu, "T")
    .OUTPUT(dx, "T")
    .DATATYPE(T, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(dim, Int, -1)
    .ATTR(approximate, Int, 1)
    .ATTR(activate_left, Bool, false)
    .OP_END_FACTORY_REG(GeGluGradV2)

} // namespace ge

#endif