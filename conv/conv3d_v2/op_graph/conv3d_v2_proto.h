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
 * \file conv3dv2_proto.h
 * \brief
 */

#ifndef CONV3DV2_PROTO_H
#define CONV3DV2_PROTO_H

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief Computes a 3D convolution with 5D "x", "filter" and "bias" tensors.
* Like this, output = CONV(x, filter) + bias. \n
* If case with 'int8' dtype appears in Atlas A3 Training Series Product/Atlas A3 Inference Series Product or
* Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component,
* like this: output = CONV(x, filter) * scale + bias.
* @par Inputs:
* @li x: A required 5D tensor of input image. \n
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type bfloat16, float16, float32, int8 and format "NCDHW" is supported. \n
* In Ascend 950 AI Processor, a tensor of type bfloat16, float16, float32 or hifloat8 and format "NCDHW" or
* "NDHWC" can be supported.
* @li filter: A required 5D tensor of convolution kernel. \n
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type bfloat16, float16, float32, int8 and format "NCDHW" is supported.
* Kernel_h and kernel_w should be both less than 512. \n
* In Ascend 950 AI Processor, a tensor with data type bfloat16, float16, float32 or hifloat8 and format "NCDHW" or
* "DHWCN" can be supported. Kernel_h and kernel_w should be both less than 256.
* @li bias: An optional 1D tensor of additive biases to the outputs.
* The data is stored in the order of: [out_channels]. \n
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type float16, float32 and format "ND" is supported. \n
* In Ascend 950 AI Processor, a tensor with data type bfloat16, float16 or float32 and format "ND" can be supported.
* @li scale: A optional 1D tensor of scaling factors.
* The data is stored in the order of: [out_channels]. \n
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type float32 and format "ND" is supported. \n
* In Ascend 950 AI Processor, this parameter is not supported.
* @li offset: An optional 1D tensor of bias.
* The data is stored in the order of: [out_channels].
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type float32 and format "ND" is supported. \n
* In Ascend 950 AI Processor, this parameter is not supported.
* @li offset_w: An optional quantitative offset tensor. A tensor of type int8. Reserved.
*\n
* @li The following are the supported data types and data formats for
* (Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product):
\n
| Tensor    | x        | filter   | bias     | scale   | offset  |    y     |\n
| :-------: | :------: | :------: | :------: | :-----: | :-----: | :------: |\n
| Data Type | float16  | float16  | float16  | float32 | float32 | float16  |\n
|           | bfloat16 | bfloat16 | float32  | float32 | float32 | bfloat16 |\n
|           | float32  | float32  | float32  | float32 | float32 | float32  |\n
|           | int8     | int8     | float32  | float32 | float32 | bfloat16 |\n
| Format    | NCDHW    | NCDHW    | ND       | ND      | ND      | NCDHW    |\n
\n
* The following are the supported data types and data formats for Ascend 950 AI Processor:
\n
| Tensor    | x        | filter   | bias     |    y     |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | float32  | float32  | float32  | float32  |\n
|           | hifloat8 | hifloat8 | float32  | hifloat8 |\n
| Format    | NCDHW    | NCDHW    | ND       | NCDHW    |\n
|           | NDHWC    | DHWCN    | ND       | NDHWC    |\n
\n
* @par Attributes:
* @li strides: Required. A list of 5 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The n and in_channels dimensions must be set to 1.
* When the format is "NDHWC", its shape is [1, stride_d, stride_h, stride_w, 1],
* when the format is "NCDHW", its shape is [1, 1, stride_d, stride_h, stride_w].
* @li pads: Optional. A list of 6 integers. The number of pixels to add to each
* (pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right) side of the input. Defaults to [0, 0, 0, 0, 0, 0].
* @li dilations: Optional. A list of 5 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The n and in_channels dimensions must be set to 1.
* When the format is "NDHWC", its shape is [1, dilation_d, dilation_h, dilation_w, 1],
* when the format is "NCDHW", its shape is [1, 1, dilation_d, dilation_h, dilation_w]. Defaults to [1, 1, 1, 1].
* @li groups: Optional. An integer of type int32. The number of groups
* in group convolution. In_channels and out_channels must both be divisible by "groups". Defaults to 1.
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product, groups can only be equal to 1.
* @li data_format: Optional. It is a string represents input's data format. Defaults to "NCDHW". Reserved.
* @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
* and is used for filling in pad values. Defaults to 0. It can only be supported in
* Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
* @li pad_mode: Optional. An optional string parameter, indicating the mode of pad.
* It must be "SPECIFIC" or "SAME" or "VALID". Defaults to "SPECIFIC".
* @li enable_hf32: Optional. An optional bool parameter. Used to enable hf32 computation.
* If true, enable hf32 computation, otherwise, disable hf32 computation. Defaults to false.
* @par Outputs:
* y: A 5D tensor of output. \n
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product,
* a tensor with data type bfloat16,float32,float16 and format "NCDHW" is supported,
* which the data is stored in [n, out_channels, out_depth, out_height, out_width]. \n
* In Ascend 950 AI Processor, a tensor with data type bfloat16, float16, float32 or hifloat8 and
* format "NCDHW" or "NDHWC" can be supported. which the data is stored in
* [n, out_channels, out_depth, out_height, out_width] or [n, out_depth, out_height, out_width, out_channels].
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
|                  | kernel_h             | [1, 255]     |\n
|                  | kernel_w             | [1, 255]     |\n
| bias size        | out_channels         | [1, 1000000] |\n
| scale size       | out_channels         | [1, 1000000] |\n
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
* @li The w dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
* @li If any dimension of x/filter/bias/scale/offset/y shape exceeds max 1000000,
* the product of each dimension of x/filter/bias/scale/offset/y
* shape exceeds max int32 minus one (2147483646) or the value of
* strides/pads/dilations/groups/data_format/offset_x/pad_mode/enable_hf32
* exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
\n
*/

REG_OP(Conv3DV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(pads, ListInt, {0, 0, 0, 0, 0, 0})
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(offset_x, Int, 0)
    .ATTR(pad_mode, String, "SPECIFIC")
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(Conv3DV2)

}  // namespace ge
#endif  // CONV3DV2_PROTO_H
