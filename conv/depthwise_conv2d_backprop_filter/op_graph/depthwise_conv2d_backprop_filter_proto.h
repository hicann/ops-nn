/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEPTHWISE_CONV2D_BACKPROP_FILTER_PROTO_H
#define DEPTHWISE_CONV2D_BACKPROP_FILTER_PROTO_H

#include "graph/operator_reg.h"
namespace ge {

/**
 * @brief Computes the gradients of depthwise convolution with respect to the filter.
 * @par Inputs:
 * @li input: A required 4D tensor of type float16, float32 or bfloat16.
 * @li filter_size: A required 1D tensor of type int32 or int64. Filter's shape.
 * @li out_backprop: A required 4D tensor of type float16, float32 or bfloat16.
 * @par Attributes:
 * @li strides: Required. A list of 4 integers.
 * @li dilations: Optional. A list of 4 integers. Defaults to [1, 1, 1, 1].
 * @li pads: Required. A list of 4 integers.
 * @li data_format: Optional. A string. Defaults to "NHWC".
 * @par Outputs:
 * @li filter_grad: A required 4D tensor of type float32.
 */
#ifndef OPS_PROTO_DEF_DEPTHWISECONV2DBACKPROPFILTER
#define OPS_PROTO_DEF_DEPTHWISECONV2DBACKPROPFILTER
REG_OP(DepthwiseConv2DBackpropFilter)
    .INPUT(input, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(filter_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(filter_grad, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(DepthwiseConv2DBackpropFilter)
#endif

} // namespace ge
#endif // DEPTHWISE_CONV2D_BACKPROP_FILTER_PROTO_H
