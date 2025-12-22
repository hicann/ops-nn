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
 * \file conv2d_v2_proto.h
 * \brief
 */

#ifndef CONV2D_V2_PROTO_H
#define CONV2D_V2_PROTO_H

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief Conv2Dv2 computes a 2D convolution given 4D "x", "filter" and "bias" tensors.
* Like this, output = CONV(x, filter) + bias.
* @par Inputs:
* @li x: A required 4D tensor of input image. With the format "NHWC" which shape is
* [n, h, w, in_channels] or the format "NCHW" which shape is [n, in_channels, h, w].
* @li filter: A required 4D tensor of convolution kernel.
* With the format "HWCN" which shape is [kernel_h, kernel_w, in_channels / groups, out_channels]
* or the format "NCHW" which shape is [out_channels, in_channels / groups, kernel_h, kernel_w].
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels].
* @li offset_w: An optional quantitative offset tensor. A tensor of type int8. Reserved.
*\n
* The following are the supported data types and data formats:
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
* @li pads: Optional. A list of 4 integers. The number of pixels to add to each
* (pad_top, pad_bottom, pad_left, pad_right) side of the input. Defaults to [0, 0, 0, 0].
* @li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NHWC", its shape is [1, dilation_h, dilation_w, 1],
* when the format is "NCHW", its shape is [1, 1, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* @li data_format: Optional. It is a string represents input's data format. Defaults to "NCHW". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Ensure that the output is within the
* effective range. Defaults to 0. Reserved.
* @li pad_mode: Optional. An string parameter, indicating the mode of
* pad. It must be "SPECIFIC" or "SAME" or "VALID" or "SAME_UPPER" or "SAME_LOWER".
* Defaults to "SPECIFIC".
* @li enable_hf32: Optional. An optional bool parameter. Used to enable hf32 computation.
* If true, enable hf32 computation, otherwise, disable hf32 computation. Defaults to false.
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
| Name             | Field                | Scope        |\n
| :--------------: | :------------------: | :----------: |\n
| x size           | n                    | [1, 1000000] |\n
|                  | in_channels          | [1, 1000000] |\n
|                  | h                    | [1, 100000]  |\n
|                  | w                    | [1, 4096]    |\n
| filter size      | out_channels         | [1, 1000000] |\n
|                  | in_channels / groups | [1, 1000000] |\n
|                  | kernel_h             | [1, 511]     |\n
|                  | kernel_w             | [1, 511]     |\n
| bias size        | out_channels         | [1, 1000000] |\n
| offset_w size    | out_channels         | [1, 1000000] |\n
| strides          | stride_h             | [1, 63]      |\n
|                  | stride_w             | [1, 63]      |\n
| pads             | pad_top              | [0, 255]     |\n
|                  | pad_bottom           | [0, 255]     |\n
|                  | pad_left             | [0, 255]     |\n
|                  | pad_right            | [0, 255]     |\n
| dilations        | dilation_h           | [1, 255]     |\n
|                  | dilation_w           | [1, 255]     |\n
| groups           | -                    | [1, 65535]   |\n
| data_format      | -                    | ["NHWC", "NCHW"] |\n
| offset_x         | -                    | [-128, 127]  |\n
| pad_mode         | -                    | ["SPECIFIC", "SAME", "VALID", "SAME_UPPER", "SAME_LOWER"] |\n
| enable_hf32      | -                    | [true, false] |\n
*\n
* @li The w dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
* @li If any dimension of x/filter/bias/offset_w/y shape exceeds max
* 1000000, the product of each dimension of x/filter/bias/offset_w/y
* shape exceeds max int32(2147483647) or the value of
* strides/pads/dilations/groups/data_format/offset_x/pad_mode/enable_hf32
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
*\n
*/
REG_OP(Conv2DV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCHW")
    .ATTR(offset_x, Int, 0)
    .ATTR(pad_mode, String, "SPECIFIC")
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(Conv2DV2)

}  // namespace ge
#endif  // CONV2D_V2_PROTO_H
