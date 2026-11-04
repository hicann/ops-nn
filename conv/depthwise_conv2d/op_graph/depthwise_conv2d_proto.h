/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file depthwise_conv2d_proto.h
 * \brief
 */

#ifndef DEPTHWISE_CONV2D_PROTO_H
#define DEPTHWISE_CONV2D_PROTO_H

#include "graph/operator_reg.h"
namespace ge {

/**
 * @brief Computes a 2D depthwise convolution given 4D "x" and "filter" tensors.
 * Each input channel is convolved with its own filter channel, channels are not mixed,
 * which is equivalent to a Conv2D with groups = channel number.
 * @par Inputs:
 * @li x: A required 4D tensor of input image. With the format "NHWC" which shape is
 * [n, h, w, in_channels] or the format "NCHW" which shape is [n, in_channels, h, w].
 * @li filter: A required 4D tensor of convolution kernel. Only static shape is supported.
 * With the format "HWCN" which shape is [kernel_h, kernel_w, in_channels, channel_multiplier]
 * or the format "NCHW" which shape is [in_channels, channel_multiplier, kernel_h, kernel_w].
 * @li bias: An optional 1D tensor of additive biases to the outputs.
 * The data is stored in the order of: [out_channels].
 * @li offset_w: An optional quantitative offset tensor. Reserved.
 *\n
 * The following are the supported data types and data formats:
 *\n
| Tensor    | x        | filter   | bias     |    y     |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | float32  | float32  | float32  | float32  |\n
|           | hifloat8 | hifloat8 | float32  | hifloat8 |\n
|           | int8     | int8     | int32    | int32    |\n
| Format    | NCHW     | NCHW     | ND       | NCHW     |\n
|           | NHWC     | HWCN     | ND       | NHWC     |\n
 *\n
 * @par Attributes:
 * @li strides: Required. A list of 4 integers. The stride of the sliding window
 * for each dimension of input. Defaults to [1, 1, 1, 1] is not used, it is required.
 * @li dilations: Optional. A list of 4 integers. Defaults to [1, 1, 1, 1].
 * @li pads: Required. A list of 4 integers. The number of pixels to add to each
 * (pad_top, pad_bottom, pad_left, pad_right) side of the input.
 * @li data_format: Optional. It is a string represents input's data format,
 * only support "NCHW" and "NHWC". Defaults to "NHWC".
 * @li offset_x: Optional. An integer of type int32. Defaults to 0. Reserved.
 * @par Outputs:
 * y: A 4D tensor of output feature map.
 * With the format "NHWC" which shape is [n, out_height, out_width, out_channels]
 * or the format "NCHW" which shape is [n, out_channels, out_height, out_width].
 *\n
 *     out_channels = in_channels * channel_multiplier
 *\n
 *     out_height = (h + pad_top + pad_bottom -
 *                   (dilation_h * (kernel_h - 1) + 1))
 *                  / stride_h + 1
 *\n
 *     out_width = (w + pad_left + pad_right -
 *                  (dilation_w * (kernel_w - 1) + 1))
 *                 / stride_w + 1
 */

#ifndef OPS_PROTO_DEF_DEPTHWISECONV2D
#define OPS_PROTO_DEF_DEPTHWISECONV2D
REG_OP(DepthwiseConv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8, DT_FLOAT}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(DepthwiseConv2D)
#endif // OPS_PROTO_DEF_DEPTHWISECONV2D

} // namespace ge
#endif // DEPTHWISE_CONV2D_PROTO_H
