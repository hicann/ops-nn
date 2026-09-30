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
 * \file conv3d_proto.h
 * \brief
 */

#ifndef CONV3D_PROTO_H
#define CONV3D_PROTO_H

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Computes a 3D convolution with 5D "x", "filter" and "bias" tensors.
* Like this, output = CONV(x, filter) + bias.
* @par Inputs:
    * @li x: A required 5D tensor of input image.
            The format of x is NCDHW or NDHWC.
            The data is stored in the order of: [n, in_channels, d, h, w] or [n, d, h, w, in_channels]. \n
            Any dimension of x shape must be in [1, 2147483646] except Ascend 950 AI Processor. \n
            In Ascend 950 AI Processor, any dimension of x shape must be in [1, 1000000].
    * @li filter: A required 5D tensor of convolution kernel.
                Must have the same type as "x".
                The format support NCDHW or DHWCN.
                The data is stored in the order of:[out_channels, in_channels, kernel_d, kernel_h, kernel_w] or
                [kernel_d, kernel_h, kernel_w, in_channels, out_channels]. \n
                The value of kernel_h * kernel_w * kernel_k0 must be in [0, 65535],
                kernel_k0 is determined by the data type, indicating the number of elements aligned to 32B. \n
                The kernel_h and kernel_w dimensions must be in [1, 511]. \n
                The other values of filter_size must be in [1, 2147483646]. \n
                When format is DHWCN and type is float32,
                filter should be a constants except Ascend 950 AI Processor. \n
                In Ascend 950 AI Processor, the kernel_h and kernel_w dimensions must be in [1, 255],
                And the other values of filter_size must be in [1, 1000000].
    * @li bias: An optional 1D tensor of additive biases to the outputs.
                The data is stored in the order of: [out_channels].
                "out_channels" must equals to the "out_channels" of output y. \n
                In Ascend 950 AI Processor, the out_channels dimension must be in [1, 1000000]
    * @li offset_w: An optional quantitative offset tensor. Reserved.
*\n
*\n
* The following are the supported data types and data formats for Ascend 950 AI Processor:
*\n
| Tensor    | x        | filter   | bias     |   y      |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | bfloat16 | bfloat16 | bfloat16 | bfloat16 |\n
|           | float32  | float32  | float32  | float32  |\n
|           | hifloat8 | hifloat8 | float32  | hifloat8 |\n
|           | int8     | int8     | int32    | int32    |\n
| Format    | NCDHW    | NCDHW    | ND       | NCDHW    |\n
|           | NDHWC    | DHWCN    | ND       | NDHWC    |\n
*\n
* The following are the supported data types and data formats for other products:
*\n
| Tensor    | x        | filter   | bias     | y        |\n
| :-------: | :------: | :------: | :------: | :------: |\n
| Data Type | float16  | float16  | float16  | float16  |\n
|           | bfloat16 | bfloat16 | float32  | bfloat16 |\n
| Format    | NCDHW    | NCDHW    | ND       | NCDHW    |\n
|           | NDHWC    | DHWCN    | ND       | NDHWC    |\n
|           | NCDHW    | DHWCN    | ND       | NCDHW    |\n
*\n
* @par Attributes:
    * @li strides: Required. A list of 5 integers. Specifies the stride of the
                sliding window for each dimension of "x". The dimension order is determined by the data format of
"x". The n and in_channels dimensions must be 1. \n When the format is "NDHWC", its shape is [1, stride_d, stride_h,
stride_w, 1], when the format is "NCDHW", its shape is [1, 1, stride_d, stride_h, stride_w]. \n The stride_h and
stride_w dimensions must be in [1, 63]. The stride_d must be in [1, 2147483646] except Ascend 950 AI Processor.
\n In Ascend 950 AI Processor the stride_d must be in [1, 1000000].
    * @li pads: Required. A list of 1, 3 or 6 integers. Supports only padding along the d, h and w dimensions.
                Length 6: explicit pads in sequence of pad_head, pad_tail, pad_top, pad_bottom, pad_left and
                pad_right; length 3: [pad_d, pad_h, pad_w], broadcast to both sides of each dimension;
                length 1: the same value for all dimensions. \n
                The pad_top, pad_bottom, pad_left and pad_right must be in [0, 255].
                The pad_head and pad_tail must be in [0, 2147483646] except Ascend 950 AI Processor. \n
                In Ascend 950 AI Processor the pad_head and pad_tail must be in [1, 1000000].
    * @li dilations: Optional. A list of 5 integers. Specifies the dilation
                    factor for each dimension of "x". The dimension order is determined by the data format of "x".
\n When the format is "NDHWC", its shape is [1, dilation_d, dilation_h, dilation_w, 1], when the format is "NCDHW",
its shape is [1, 1, dilation_d, dilation_h, dilation_w]. \n Default value is [1, 1, 1, 1, 1]. \n The dilation_h and
dilation_w dimensions must be in [1, 255]. The dilation_d dimensions must be in [0, 2147483646] except Ascend 950 AI
Processor. \n In Ascend 950 AI Processor the dilation_d dimensions must be in [1, 1000000].
    * @li groups: Optional. An integer of type int32. The number of groups
                in group convolution. In_channels and out_channels must both be divisible by "groups".
                The value of groups must be in [1, 65535]. Default value is 1.
    * @li data_format: Optional. It represents data format of the input x and output y, and is a string
                    dtype with "NCDHW" and "NDHWC". Defaults to "NDHWC".
    * @li offset_x: Optional. An integer of type int32. It means offset in quantization algorithm
                    and is used for filling in pad values. Ensure that the output is within the
                    effective range. Defaults to 0.
* @par Outputs:
* y: A 5D tensor of output feature map. Has the same type as "x". With the format "NCDHW" or "NDHWC",
        the data is stored in the order of: [n, out_channels, out_depth, out_height, out_width] or
        [n, out_depth, out_height, out_width, out_channels]. \n
*\n
*     out_depth = (d + pad_head + pad_tail -
*                  (dilation_d * (kernel_d - 1) + 1))
*                 / stride_d + 1
*\n
*     out_height = (h + pad_top + pad_bottom -
*                   (dilation_h * (kernel_h - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (w + pad_left + pad_right -
*                  (dilation_w * (kernel_w - 1) + 1))
*                 / stride_w + 1
*\n
        Any dimension of y shape must be in [1, 2147483646] except Ascend 950 AI Processor. \n
        In Ascend 950 AI Processor, any dimension of y shape must be in [1, 1000000].
* @attention Constraints:
    * @li The input x size after padding should be greater than the filter size.
    * @li The w dimension of the input x supports cases exceeding 4096, but it may
    * cause compilation errors.
    * @li If any dimension of x/filter/bias/y shape exceeds max int32 minus one (2147483646),
    * the product of each dimension of x/filter/bias/y shape exceeds max int32 minus one (2147483646) or
    * the value of strides/pads/dilations/offset_x exceeds the range which is described in Attributes,
    * the correctness of the operator cannot be guaranteed. \n
    * In Ascend 950 AI Processor: If any dimension of x/filter/bias/y shape exceeds max
    * 1000000, the product of each dimension of x/filter/bias/y
    * shape exceeds max int32(2147483647) or the value of stride/padding/dilation/offset_x
    * exceeds the range in the above table, the correctness of the operator cannot be guaranteed.
    * @li If the Conv3D enters the Direct Memory Access(DMA) copy process, a timeout AI Core error may be reported.
    * You are advised to reduce the Conv3D specifications and try again.
    * You can view the warning log to check whether the DMA copy process is entered.
    * For example: 'The Conv3D has entered the DMA processing process. A timeout AI Core error may be reported.
    * If a timeout AI Core error is reported, reduce the Conv3D specifications and try again' \n
* @par Third-party framework compatibility
    * @li Compatible with the TensorFlow operator "conv3d".
    * @li Compatible with the Caffe operator "Convolution".
    * @li Compatible with the ONNX operator 3D "Conv".
    * @li Compatible with the PyTorch operator "Conv3D".
*/

#ifndef OPS_PROTO_DEF_CONV3D
#define OPS_PROTO_DEF_CONV3D
REG_OP(Conv3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv3D)
#endif // OPS_PROTO_DEF_CONV3D
} // namespace ge
#endif // CONV3D_PROTO_H
