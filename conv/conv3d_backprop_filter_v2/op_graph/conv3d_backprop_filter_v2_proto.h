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
 * \file conv3d_backprop_filter_v2_proto.h
 * \brief
 */
#ifndef OPS_CONV_CONV3D_BACKPROP_FILTER_PROTO_H_
#define OPS_CONV_CONV3D_BACKPROP_FILTER_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes the gradients of convolution3D with respect to the filter
*@par Inputs:
 * @li x: A 5D tensor. Must be one of the following types: float16, float32, bfloat16.
 * The format of the x tensor must be one of the followings:
 * [batch, in_depth, in_height, in_width, in_channels]
 * or [batch, in_channels, in_depth, in_height, in_width]. \n
 * @li filter_size: A 1D tensor of type int32 or int64. The tensor representing the
 * shape of filter, where filter is a 5D tensor.
 * The order of integers in the tensor is determined by filter format,
 * and integers represent the length of each dimension of filter.
 * The axes sequence that can be entered are as follows:
 * [out_channels, in_channels, filter_depth, filter_height, filter_width]
 * or [out_channels, filter_depth, filter_height, filter_width, in_channels].
 * height (H) and width (W) dimensions must be in [1,511]. \n
 * @li out_backprop: A 5D tensor. Must have the same type and format as x.
 * The format of the out_backprop tensor must be one of the followings:
 * [batch, out_depth, out_height, out_width, out_channels] or
 * [batch, out_channels, out_depth, out_height, out_width].
 * Gradients with respect to the "output" of the convolution. \n
*@par Attributes:
 * @li strides: Required. A tuple/list of 5 integers. Specifies the stride
 * of the sliding window for each dimension of feature map.
 * The strides have the same axes sequence as feature map:
 * [batch, stride_depth, stride_height, stride_width, channels] or
 * [batch, channels, stride_depth, stride_height, stride_width].
 * The batch(N) and channels(C) dimensions must be 1.
 * The height (H) and width (W) dimensions must be in [1, 63].
 * The depth (D) dimension must be in [1, 255]. \n
 * @li pads: Required. A tuple/list of 6 integers. Specifies the pads factor of
 * feature map in each directions. Supports only pads along the depth(D),
 * height(H) and width(W) dimensions.
 * The pads sequence is as follows: [front, back, top, bottom, left, right].
 * The height (H), width (W) and depth (D) dimensions must be in [0, 255].
 * Modes "SAME" and "VAILD" padding can be achieved with appropriate values of each direction in pads. \n
 * @li dilations: Optional. A tuple/list of 5 integers. The dilation factor
 * for each dimension of input. Defaults to [1, 1, 1, 1, 1]
 * The dilations have the same axes sequence feature map:
 * [batch, dilation_depth, dilation_height, dilation_width, channels] or
 * [batch, channels, dilation_depth, dilation_height, dilation_width].
 * The batch(N) and channels dimensions must be 1.
 * The width (W), height (H) and depth(D) dimensions must be in [1,255]. \n
 * @li groups: An optional integer within the effective range of [1, 65535]. Default to 1.
 * Number of blocked connections from in_channels to out_channels.
 * Currently in_channels and out_channels must be divisible by groups
 * and x data type must be one of the following types: float16, bfloat16, float32. \n
 * @li data_format: An optional string. The value must be one of ["NDHWC", "NCDHW"]. Defaults to "NDHWC".
 * The correspondence is as follows: batch(N), depth(D), height(H), width(W), channels(C).
 * Specify the data format of the x and out_backprop. \n

*@par Outputs:
 * y: A Tensor that has the type float32. The format is NCDHW or NDHWC or DHWCN.

*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_filter
*/
REG_OP(Conv3DBackpropFilterV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(filter_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(Conv3DBackpropFilterV2)
} // namespace ge

#endif // OPS_CONV_CONV3D_BACKPROP_FILTER_PROTO_H_