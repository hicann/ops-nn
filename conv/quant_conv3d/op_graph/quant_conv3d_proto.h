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
 * \file quant_conv3d_proto.h
 * \brief
 */

#ifndef QUANT_CONV3D_PROTO_H
#define QUANT_CONV3D_PROTO_H

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief Computes a 3D convolution with 5D "x", "filter" and "bias" tensors,
* and then executes a per-channel dequant operation with "scale" tensors.
* Like this, output = (CONV(x, filter) + bias) * scale.
* @par Inputs:
* @li x: A required 5D tensor of input image. With the format "NDHWC" which shape is
* [n, d, h, w, in_channels] or the format "NCDHW" which shape is [n, in_channels, d, h, w].
* @li filter: A required 5D tensor of convolution kernel.
* With the format "DHWCN" which shape is [kernel_d, kernel_h, kernel_w, in_channels / groups, out_channels]
* or the format "NCDHW" which shape is [out_channels, in_channels / groups, kernel_d, kernel_h, kernel_w].
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
| Format    | NCDHW       | NCDHW       | ND           | ND      | ND      | NCDHW                               |\n
*\n
* @par Attributes:
* @li dtype: Required. A integer of type int8. It means output's dtype.
* @li strides: Required. A list of 5 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NDHWC", its shape is [1, stride_d, stride_h, stride_w, 1],
* when the format is "NCDHW", its shape is [1, 1, stride_d, stride_h, stride_w].
* @li pads: Required. A list of 6 integers. The number of pixels to add to each
* (pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right) side of the input.
* @li dilations: Optional. A list of 5 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NDHWC", its shape is [1, dilation_d, dilation_h, dilation_w, 1],
* when the format is "NCDHW", its shape is [1, 1, dilation_d, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* @li data_format: Optional. It is a string represents input's data format.
* Defaults to "NCDHW". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Ensure that the output is within the
* effective range. Defaults to 0. Reserved.
* @li round_mode: Optional. Defaults to "rint". It is rounding mode of calculation.
* If output's data type is hifloat8, round_mode can be set to 'round'. Otherwise, it can be set to 'rint'.
* @li pad_mode: Optional. An optional string parameter, indicating the mode of pad.
* It must be "SPECIFIC" or "SAME" or "VALID". Defaults to "SPECIFIC".
*\n
* @par Outputs:
* y: A 5D tensor of output feature map.
* With the format "NDHWC", the data is stored in the order of: [n, out_depth, out_height, out_width, out_channels]
* or the format "NCDHW", the data is stored in the order of: [n, out_channels, out_depth, out_height, out_width].
*\n
*     out_depth  = (d + pad_head + pad_tail -
*                   (dilation_d * (kernel_d - 1) + 1))
*                  / stride_d + 1
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
|                  | d                    | [1, 1000000] |\n
|                  | h                    | [1, 100000]  |\n
|                  | w                    | [1, 4096]    |\n
| filter size      | out_channels         | [1, 1000000] |\n
|                  | in_channels / groups | [1, 1000000] |\n
|                  | kernel_d             | [1, 1000000] |\n
|                  | kernel_h             | [1, 511]     |\n
|                  | kernel_w             | [1, 511]     |\n
| scale size       | out_channels         | [1, 1000000] |\n
| bias size        | out_channels         | [1, 1000000] |\n
| offset size      | out_channels         | [1, 1000000] |\n
| strides          | stride_d             | [1, 1000000] |\n
|                  | stride_h             | [1, 63]      |\n
|                  | stride_w             | [1, 63]      |\n
| pads             | pad_head             | [0, 1000000] |\n
|                  | pad_tail             | [0, 1000000] |\n
|                  | pad_top              | [0, 255]     |\n
|                  | pad_bottom           | [0, 255]     |\n
|                  | pad_left             | [0, 255]     |\n
|                  | pad_right            | [0, 255]     |\n
| dilations        | dilation_d           | [1, 1000000] |\n
|                  | dilation_h           | [1, 255]     |\n
|                  | dilation_w           | [1, 255]     |\n
| groups           | -                    | [1, 65535]   |\n
| data_format      | -                    | ["NDHWC", "NCDHW"] |\n
| offset_x         | -                    | [-128, 127]  |\n
| pad_mode         | -                    | ["SPECIFIC", "SAME", "VALID"] |\n
| enable_hf32      | -                    | [true, false] |\n
*\n
* @li The W dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
* @li If any dimension of x/filter/scale/bias/offset/y shape exceeds max int32 minus one (2147483646),
* the product of each dimension of x/filter/scale/bias/offset/y shape exceeds max int32 minus one (2147483646) or
* the value of strides/pads/dilations/groups/data_format/offset_x/pad_mode/enable_hf32 exceeds the range
* in the above table, the correctness of the operator cannot be guaranteed.
*\n
* @par Quantization supported or not
* Yes
*/
REG_OP(QuantConv3D)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(scale, TensorType({DT_UINT64, DT_INT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32, DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .REQUIRED_ATTR(dtype, Int)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(pads, ListInt, {0, 0, 0, 0, 0, 0})
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(offset_x, Int, 0)
    .ATTR(round_mode, String, "rint")
    .ATTR(pad_mode, String, "SPECIFIC")
    .OP_END_FACTORY_REG(QuantConv3D)

}  // namespace ge
#endif  // QUANT_CONV3D_PROTO_H
