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
 * \file quant_conv2d_proto.h
 * \brief
 */

#ifndef QUANT_CONV2D_PROTO_H
#define QUANT_CONV2D_PROTO_H

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Computes a 2D convolution given 4D "x", "filter" and "bias" tensors
* and then executes a per-channel dequant operation with "scale" tensors.
* Like this, output = (CONV(x, filter) + bias) * scale.
* @par Inputs:
* @li x: A required 4D tensor of input image. With the format "NCHW" which shape is [n, in_channels, h, w].
* @li filter: A required 4D tensor of convolution kernel.
* With the format "NCHW" which shape is [out_channels, in_channels / groups, kernel_h, kernel_w].
* @li scale: A required 1D tensor of scaling factors. The data is stored in the order of: [out_channels].
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels].
* @li offset: An optional quantitative offset tensor. Reserved.
*\n
* The following are the supported data types and data formats (for Ascend 950 AI Processor):
*\n
| Tensor    | x           | filter      | scale        | bias    | offset  | y                                   |\n
| :-------: | :---------: | :---------: | :----------: | :------:| :------:| :---------------------------------: |\n
| Data Type | int8        | int8        | uint64/int64 | int32   | float32 | float16                             |\n
|           | float8_e4m3 | float8_e4m3 | uint64/int64 | float32 | float32 | float32/float16/bfloat16/float8_e4m3|\n
|           | hifloat8    | hifloat8    | uint64/int64 | float32 | float32 | float32/float16/bfloat16/hifloat8   |\n
| Format    | NCHW        | NCHW        | ND           | ND      | ND      | NCHW                                |\n
*\n
* @par Attributes:
* @li dtype: Required. A integer of type int8. It means output's dtype.
* @li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NCHW", its shape is [1, 1, stride_h, stride_w].
* @li pads: Required. A list of 4 integers. The number of pixels to add to each
* (pad_top, pad_bottom, pad_left, pad_right) side of the input.
* @li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NCHW", its shape is [1, 1, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* @li data_format: Optional. It is a string represents input's data format.
* Defaults to "NHWC". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Ensure that the output is within the
* effective range. Defaults to 0. Reserved.
* @li round_mode: Optional. Defaults to "rint". It is rounding mode of calculation.
* If output's dtype is hifloat8, round_mode can be set to 'round'. Otherwise, it can be set to 'rint'.
* @par Outputs:
* y: A 4D tensor of output feature map.
* When the format "NCHW" which shape is [n, out_channels, out_height, out_width].
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
* @li In Ascend 950 AI Processor: If any dimension of x/filter/bias/scale/offset/y shape exceeds max
* 1000000, the product of each dimension of x/filter/bias/scale/offset/y
* shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
*\n
* @par Quantization supported or not
* Yes
*\n
*/
#ifndef OPS_PROTO_DEF_QUANTCONV2D
#define OPS_PROTO_DEF_QUANTCONV2D
REG_OP(QuantConv2D)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(scale, TensorType({DT_UINT64, DT_INT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32, DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .REQUIRED_ATTR(dtype, Int)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCHW")
    .ATTR(offset_x, Int, 0)
    .ATTR(round_mode, String, "rint")
    .OP_END_FACTORY_REG(QuantConv2D)
#endif // OPS_PROTO_DEF_QUANTCONV2D
} // namespace ge
#endif // QUANT_CONV2D_PROTO_H
