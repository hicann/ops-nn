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
 * \file conv3d_backprop_input_v2_proto.h
 * \brief
 */
#ifndef OPS_CONV3D_BACKPROP_INPUT_V2_PROTO_H_
#define OPS_CONV3D_BACKPROP_INPUT_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Computes the gradients of convolution3D with respect to the input.
*@par Inputs:
 * @li input_size: A 1D tensor of type int32 or int64.
 * The tensor representing the shape of feature map (the input of the convolution),
 * where feature map is a 5D tensor.
 * The order of integers in the tensor is determined by feature map format,
 * and integers represent the length of each dimension of feature map.
 * The axes sequence that can be entered are as follows:
 * [batch, in_depth, in_height, in_width, in_channels] or
 * [batch, in_channels, in_depth, in_height, in_width].
 * @li filter: A 5D tensor.
 * The format of the filter tensor must be one of the followings:
 * [out_channels, in_channels/groups, filter_depth, filter_height, filter_width] or
 * [out_channels, filter_depth, filter_height, filter_width, in_channels] or
 * [filter_depth, filter_height, filter_width, in_channels, out_channels].
 * In short, the filter tensor support format NCDHW, NDHWC and DHWCN.
 * The NCDHW can be one of the following types: float16, bfloat16, float32.
 * The DHWCN and NDHWC can be one of the following types: float16, bfloat16, float32.
 * kernel_height (H) and kernel_width (W) must have dimension in [1, 511].
 * @li out_backprop: A 5D tensor.
 * The format of the out_backprop tensor must be one of the followings:
 * [batch, out_depth, out_height, out_width, out_channels] or
 * [batch, out_channels, out_depth, out_height, out_width].
 * Gradients with respect to the "output" of the convolution.
*@par Attributes:
 * @li strides: Required. A tuple/list of 5 integers. Specifies the stride of the sliding window
 * for each dimension of feature map. The strides have the same axes sequence as feature map:
 * [batch, stride_depth, stride_height, stride_width, channels] or
 * [batch, channels, stride_depth, stride_height, stride_width].
 * The batch(N) and channels(C) dimensions must be 1.
 * The height (H) and width (W) dimensions must be in [1, 63].
 * The depth (D) dimension must be in [1, 255].
 * @li pads: Required. A tuple/list of 6 integers. Specifies the pads factor of
 * feature map in each directions. Supports only pads along the depth(D),
 * height(H) and width(W) dimensions.
 * The pads sequence is as follows: [front, back, top, bottom, left, right].
 * The height (H), width (W) and depth (D) dimensions must be in [0, 255].
 * Modes "SAME" and "VAILD" padding can be achieved with appropriate values of each direction in pads.
 * @li dilations: Optional. A tuple/list of 5 integers. The dilation factor
 * for each dimension of feature map. Defaults to [1, 1, 1, 1, 1]. \n
 * The dilations have the same axes sequence as feature map:
 * [batch, channels, dilation_depth, dilation_height, dilation_width] or
 * [batch, dilation_depth, dilation_height, dilation_width, channels].
 * The batch(N) and channels dimensions must be 1.
 * The width (W), height (H) and depth(D) dimensions must be in [1, 255].
 * @li groups: An optional integer within the effective range of [1, 65535]. Default to 1.
 * Number of blocked connections from in_channels to out_channels.
 * The in_channels and out_channels must be divisible by groups.
 * When the groups value differs, the supported data type may vary, specifically as follows: \n
 * \n
 * | groups  |        dtype           | out_backprop format | filter format |     y format   |
 * |---------|------------------------|---------------------|---------------|----------------|
 * |  =1     |float16/bfloat16/float32|       NDHWC         |      NDHWC    |     NDHWC      |
 * |  =1     |float16/bfloat16/float32|       NDHWC         |      NCDHW    |     NDHWC      |
 * |  =1     |float16/bfloat16/float32|       NDHWC         |      DHWCN    |     NDHWC      |
 * |  >=1    |float16/bfloat16/float32|       NCDHW         |      NCDHW    |     NCDHW      |
 * |  >=1    |float16/bfloat16/float32|       NCDHW         |      NDHWC    |     NCDHW      |
 * |  >=1    |float16/bfloat16/float32|       NCDHW         |      DHWCN    |     NCDHW      |
 * \n
 * @li data_format: An optional string. The value must be one of ["NDHWC", "NCDHW"]. Defaults to "NDHWC".
 * The correspondence is as follows: batch(N), depth(D), height(H), width(W), channels(C).
 * Specify the data format of the out_backprop and y.
 * @li enable_hf32: Optional. An optional bool parameter. Used to enable hf32 computation.
 * If true, enable hf32 computation, otherwise, disable hf32 computation. Defaults to false. \n
*@par Outputs:
 * y: A tensor. It has the same format as out_backprop.
 * The type is float16, bfloat16, float32.
 * The gradients of feature map.
 * @attention Constraints:
 * In Ascend 950PR/Ascend 950DT: The behavior of gradient computation in the padding region depends on the input shape.
 * Depending on the operator optimization strategy, the padding gradients may be directly set to 0.\n
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_input
*/
REG_OP(Conv3DBackpropInputV2)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(Conv3DBackpropInputV2)

} // namespace ge

#endif // OPS_CONV3D_BACKPROP_INPUT_V2_PROTO_H_
