/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ge_glu_v2_proto.h
 * \brief
 */
#ifndef OPS_ACTIVATION_GE_GLU_V2_GRAPH_PLUGIN_GE_GLU_V2_PROTO_H_
#define OPS_ACTIVATION_GE_GLU_V2_GRAPH_PLUGIN_GE_GLU_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Compute the GeGluV2,
* where the activations function in GLU is Gelu.

* @par Inputs:
* x: A Tensor. Must be one of the following types: bfloat16, float16, float32.
* Shape supports at least 1 dimensions, and at most 8 dimensions.
* The length of the split dimension in x must be an even number.

* @par Outputs:
* Two outputs, including:
* @li y: A Tensor. Must be one of the following types: bfloat16, float16, float32.
* The dtype of y must exactly same with input x.
* The shape of y matches the shape of x in all dimensions except for the split dimension,
* where its length is half of length of x's split dimension.
* @li gelu: A Tensor. Must be one of the following types: bfloat16, float16, float32.
* The dtype of gelu must exactly same with input x.
* The shape of gelu matches the shape of x in all dimensions except for the split dimension,
* where its length is half of length of x's split dimension.

* @par Attributes:
* Three attributes, including:
* @li dim: An optional int. The dimension to be split, default is -1.
* @li approximate: An optional int. Which formula used for the activation computation.
* The gelu approximation algorithm to use: 'none'(0) or 'tanh'(1), default is 'tanh'(1).
* Atlas Inference Series Product only support 'tanh'(1).
* @li activate_left: An optional bool.
* The gelu activate_left algorithm to use:
*     'false'(activate right) or 'true'(activate left), defalut is 'false'(activate right).
*/
REG_OP(GeGluV2)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .OUTPUT(gelu, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(dim, Int, -1)
    .ATTR(approximate, Int, 1)
    .ATTR(activate_left, Bool, false)
    .OP_END_FACTORY_REG(GeGluV2)

} // namespace ge

#endif