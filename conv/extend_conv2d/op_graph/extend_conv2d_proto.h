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
 * \file extend_conv2d_proto.h
 * \brief
 */

#ifndef EXTEND_CONV2D_PROTO_H
#define EXTEND_CONV2D_PROTO_H

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief ExtendConv2D computes a 2D convolution + fixpipe fused operator.
* @par Inputs:
* @li x: A required 4D tensor of input image. With the format "NCHW" which shape is
* [n, in_channels, h, w].
* @li filter: A required 4D tensor of convolution kernel.
* With the format "NCHW" which shape is [out_channels, in_channels / groups, kernel_h, kernel_w].
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels].
* @li offset: An optional quantitative offset tensor. Reserved.
* @li scale0: An optional 1D tensor. Quantization/dequantization/weighting parameter corresponding to
* the first output, which is of the channelwise type.
* @li relu_weight0: An optional 1D tensor. Indicates the relu slope parameter corresponding to the
* first output, which is of the scalar type or channelwise type.
* @li clip_value0: An optional 1D tensor. Truncated value of clip relu corresponding to the first output.
* The value is of the scalar type.
* @li scale1: An optional 1D tensor. Quantization/dequantization/weighting parameter corresponding to
* the second output, which is of the scalar type or channelwise type.
* @li relu_weight1: An optional 1D tensor. Quantization/dequantization/weighting parameter corresponding to
* the second output, which is of the scalar type or channelwise type.
* @li clip_value1: An optional 1D tensor. Truncated value of clip relu corresponding to the seccond output.
* The value is of the scalar type.
*\n
* The following are the supported data types and data formats:
*\n
| Tensor    | x        | filter   | bias    | scale0/1     | relu_weight0/1 | clip_value0/1 | y0/1    |\n
| :-------: | :------: | :------: | :-----: | :----------: | :------------: | :-----------: | :-----: |\n
| Data Type | int8     | int8     | int32   | uint64/int64 | float32        | int8          | float16/int8 |\n
|           | hifloat8 | hifloat8 | float32 | uint64/int64 | float32        | hifloat8      | float32/float16/bfloat16/hifloat8 |\n
|           | float8   | float8   | float32 | uint64/int64 | float32        | float8        | float32/float16/bfloat16/float8_e4m3 |\n
| Format    | NCHW     | NCHW     | ND      | ND           | ND             | ND            | NCHW    |\n

| Tensor    | x        | filter   | bias    | scale0/1     | relu_weight0/1 | clip_value0/1 | y0/1    |\n
| :-------: | :------: | :------: | :-----: | :----------: | :------------: | :-----------: | :-----: |\n
| Data Type | int8     | int8     | int32   | uint64/int64 | float32        | int8          | float16/int8 |\n
| Format    | NHWC     | HWCN     | ND      | ND           | ND             | ND            | NHWC    |\n
*\n
* @par Attributes:
* @li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NCHW", its shape is [1, 1, stride_h, stride_w].
* @li pads: Optional. A list of 4 integers. The number of pixels to add to each
* (pad_top, pad_bottom, pad_left, pad_right). Defaults to [0, 0, 0, 0].
* @li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NCHW", its shape is [1, 1, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* @li data_format: Optional. It is a string represents input's data format.
* Defaults to "NCHW". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Ensure that the output is within the
* effective range. Defaults to 0. Reserved.
* @li round_mode: Rounding mode enable quantization.
* If output's dtype is hifloat8, round_mode can be set to 'round'. Otherwise, it can be set to 'rint'.
* Defaults to "rint".
* @li pad_mode: Optional. An optional string parameter, indicating the mode of pad.
* It must be "SPECIFIC" or "SAME" or "VALID" or "SAME_UPPER" or "SAME_LOWER". Defaults to "SPECIFIC".
* @li enable_hf32: Optional. An bool for extendConv2D. If True, enable hf32 calculation. Defaults to false.
* If False, disable hf32 calculation. Must be false.
* @li enable_relu0: Optional. Indicates whether relu is enabled for the first output. Defaults to false.
* If False, relu is disable for the first output. Must be false.
* @li enable_relu1: Optional. Indicates whether relu is enabled for the second output. Defaults to false.
* If False, relu is disable for the second output. Must be false.
* @li dual_output: Optional. Indicates whether dual outputs are used. Defaults to false.
* When dual_output is false, extendConv2D only has output y0. When dual_output is true,
* extendConv2D has output y0 and y1. Must be false.
* @li dtype0: Optional. A integer of type int8. It means the dtype of output y0.
* Support list is [-1(Default), 0(DT_FLOAT), 1(DT_FLOAT16), 2(DT_INT8), 27(DT_BF16),
* 34(DT_HIFLOAT8), 36(DT_FLOAT8_E4M3FN)]. Defaults to -1, means the dtype is the same as x.
* @li dtype1: Optional. A integer of type int8. It means the dtype of output y1.
* Support list is [-1(Default), 0(DT_FLOAT), 1(DT_FLOAT16), 2(DT_INT8), 27(DT_BF16),
* 34(DT_HIFLOAT8), 36(DT_FLOAT8_E4M3FN)]. Defaults to -1, means the dtype is the same as x. Must be -1.
*\n
* @par Outputs:
* @li y0: The first output of ExtendConv2D. A 4D Tensor of output feature map. Has the same type as "x".
* With the format "NCHW", the data is stored in the order of: [n, out_channels, out_height, out_width].
* @li y1: The second output of ExtendConv2D. A 4D Tensor of output feature map. Has the same type as "x".
* With the format "NCHW", the data is stored in the order of: [n, out_channels, out_height, out_width].
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
| Name              | Field                | Scope        |\n
| :---------------: | :------------------: | :----------: |\n
| x size            | n                    | [1, 1000000] |\n
|                   | in_channels          | [1, 1000000] |\n
|                   | h                    | [1, 100000]  |\n
|                   | w                    | [1, 4096]    |\n
| filter size       | out_channels         | [1, 1000000] |\n
|                   | in_channels / groups | [1, 1000000] |\n
|                   | kernel_h             | [1, 511]     |\n
|                   | kernel_w             | [1, 511]     |\n
| bias size         | out_channels         | [1, 1000000] |\n
| offset_w size     | out_channels or 1    | [1, 1000000] |\n
| scale0 size       | out_channels or 1    | [1, 1000000] |\n
| relu_weight0 size | out_channels or 1    | [1, 1000000] |\n
| clip_value0 size  | out_channels or 1    | [1, 1000000] |\n
| scale1 size       | out_channels or 1    | [1, 1000000] |\n
| relu_weight1 size | out_channels or 1    | [1, 1000000] |\n
| clip_value1 size  | out_channels or 1    | [1, 1000000] |\n
| strides           | stride_h             | [1, 63]      |\n
|                   | stride_w             | [1, 63]      |\n
| pads              | pad_top              | [0, 255]     |\n
|                   | pad_bottom           | [0, 255]     |\n
|                   | pad_left             | [0, 255]     |\n
|                   | pad_right            | [0, 255]     |\n
| dilations         | dilation_h           | [1, 255]     |\n
|                   | dilation_w           | [1, 255]     |\n
| groups            | -                    | [1, 65535]   |\n
| data_format       | -                    | ["NCHW"] |\n
| offset_x          | -                    | [-128, 127]  |\n
| round_mode        | -                    | ["rint", "round"] |\n
| pad_mode          | -                    | ["SPECIFIC", "SAME", "VALID", "SAME_UPPER", "SAME_LOWER"] |\n
| enable_hf32       | -                    | [true, false] |\n
| enable_relu0      | -                    | [true, false] |\n
| enable_relu1      | -                    | [true, false] |\n
| dual_output       | -                    | [true, false] |\n
*\n
* @li In Ascend 950 AI Processor: If any dimension of
* x/filter/bias/offset_w/scale0/relu_weight0/clip_value0/scale1/relu_weight1/clip_value1/y shape exceeds max
* 1000000, the product of each dimension of x/filter/bias/scale/offset/y
* shape exceeds max int32(2147483647) or the value of strides/pads/dilations/offset_x
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
*\n
*/
REG_OP(ExtendConv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(scale0, TensorType({DT_INT64, DT_UINT64}))
    .OPTIONAL_INPUT(relu_weight0, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(clip_value0, TensorType({DT_FLOAT16, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(scale1, TensorType({DT_INT64, DT_UINT64}))
    .OPTIONAL_INPUT(relu_weight1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(clip_value1, TensorType({DT_FLOAT16, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(y0, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCHW")
    .ATTR(offset_x, Int, 0)
    .ATTR(round_mode, String, "rint")
    .ATTR(pad_mode, String, "SPECIFIC")
    .ATTR(enable_hf32, Bool, false)
    .ATTR(enable_relu0, Bool, false)
    .ATTR(enable_relu1, Bool, false)
    .ATTR(dual_output, Bool, false)
    .ATTR(dtype0, Int, -1)
    .ATTR(dtype1, Int, -1)
    .OP_END_FACTORY_REG(ExtendConv2D)

}  // namespace ge
#endif  // EXTEND_CONV2D_PROTO_H
