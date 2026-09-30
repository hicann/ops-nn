/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv2d_proto.h
 * \brief
 */

#ifndef CONV2D_PROTO_H
#define CONV2D_PROTO_H

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief Computes a 2D convolution given 4D "x", "filter" and "bias" tensors.
* Like this, output = CONV(x, filter) + bias.
* @par Inputs:
* @li x: A required 4D tensor of input image. With the format "NHWC" which shape is
* [n, h, w, in_channels] or the format "NCHW" which shape is [n, in_channels, h, w].
* @li filter: A required 4D tensor of convolution kernel.
* With the format "HWCN" which shape is [kernel_h, kernel_w, in_channels / groups, out_channels]
* or the format "NCHW" which shape is [out_channels, in_channels / groups, kernel_h, kernel_w].
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels].
* @li offset_w: An optional quantitative offset tensor. Reserved.
*\n
* The following are the supported data types and data formats (except IPV350 and Ascend 950 AI Processor):
*\n
| Tensor    | x        | filter   | bias     | y        |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | float16  | float16  | float16  | float32  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | bfloat16 | bfloat16 | bfloat16 | float32  |\n
|           | float32  | float32  | float32  | float32  |\n
|           | int8     | int8     | int32    | int32    |\n
| Format    | NCHW     | NCHW     | ND       | NCHW     |\n
|           | NHWC     | HWCN     | ND       | NHWC     |\n
|           | NCHW     | HWCN     | ND       | NCHW     |\n
*\n
* The following are the supported data types and data formats for IPV350:
*\n
| Tensor    | x       | filter  | bias    | y       |\n
| :-------: | :-----: | :-----: | :-----: | :-----: |\n
| Data Type | int16   | int8    | int32   | int32   |\n
|           | int8    | int8    | int32   | int32   |\n
| Format    | NCHW    | NCHW    | ND      | NCHW    |\n
|           | NHWC    | HWCN    | ND      | NHWC    |\n
*\n
* The following are the supported data types and data formats for Ascend 950 AI Processor:
*\n
| Tensor    | x        | filter   | bias     | y        |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | float32  | float32  | float32  | float32  |\n
|           | hifloat8 | hifloat8 | float32  | hifloat8 |\n
| Format    | NCHW     | NCHW     | ND       | NCHW     |\n
|           | NHWC     | HWCN     | ND       | NHWC     |\n
*\n
* @par Attributes:
* @li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NHWC", its shape is [1, stride_h, stride_w, 1],
* when the format is "NCHW", its shape is [1, 1, stride_h, stride_w].
* @li pads: Required. A list of 4 integers. The number of pixels to add to each
* (pad_top, pad_bottom, pad_left, pad_right) side of the input.
* @li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NHWC", its shape is [1, dilation_h, dilation_w, 1],
* when the format is "NCHW", its shape is [1, 1, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* @li data_format: Optional. It is a string represents input's data format.
* Defaults to "NHWC". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Ensure that the output is within the
* effective range. Defaults to 0. Reserved.
* @par Outputs:
* y: A 4D tensor of output feature map.
* With the format "NHWC" which shape is [n, out_height, out_width, out_channels]
* or the format "NCHW" which shape is [n, out_channels, out_height, out_width].
*\n
*     out_height = (h + pad_top + pad_bottom -
*                   (dilation_h * (kernel_h - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (w + pad_left + pad_right -
*                  (dilation_w * (kernel_w - 1) + 1))
*                 / stride_w + 1
*\n
* @attention Constraints:
* @li The following value range restrictions must be met:
*\n
| Name             | Field      | Scope       |\n
| :--------------: | :--------: | :---------: |\n
| x size           | h          | [1, 100000] |\n
|                  | w          | [1, 4096]   |\n
| filter size      | kernel_h   | [1, 511]    |\n
|                  | kernel_w   | [1, 511]    |\n
| strides          | stride_h   | [1, 63]     |\n
|                  | stride_w   | [1, 63]     |\n
| pads             | pad_top    | [0, 255]    |\n
|                  | pad_bottom | [0, 255]    |\n
|                  | pad_left   | [0, 255]    |\n
|                  | pad_right  | [0, 255]    |\n
| dilations        | dilation_h | [1, 255]    |\n
|                  | dilation_w | [1, 255]    |\n
| offset_x         | -          | [-128, 127] |\n
*\n
* @li The w dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
* @li If any dimension of x/filter/bias/offset_w/y shape exceeds max
* int32(2147483647), the product of each dimension of x/filter/bias/offset_w/y
* shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed. \n
* In Ascend 950 AI Processor: If any dimension of x/filter/bias/offset_w/y shape exceeds max
* 1000000, the product of each dimension of x/filter/bias/offset_w/y
* shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
*\n
* @li When the specifications of the Conv2D exceeds the constraints mentioned above,
* a timeout AI Core error may be reported.
*\n
* @par Quantization supported or not
* Yes
*\n
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator "conv2d".
* @li Compatible with the Caffe operator 2D "Convolution".
* @li Compatible with the ONNX operator 2D "Conv".
* @li Compatible with the PyTorch operator "Conv2D".
*/

#ifndef OPS_PROTO_DEF_CONV2D
#define OPS_PROTO_DEF_CONV2D
REG_OP(Conv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2D)
#endif // OPS_PROTO_DEF_CONV2D

} // namespace ge
#endif // CONV2D_PROTO_H
