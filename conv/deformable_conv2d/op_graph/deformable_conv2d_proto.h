/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_PROTO_DEF_DEFORMABLECONV2D
#define OPS_PROTO_DEF_DEFORMABLECONV2D

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Computes a 2D deformable convolution given 4D "x", "filter" and
 *  "offsets" tensors.
 * @par Inputs:
 * @li x: A 4D tensor of input image. With the format "NCHW", the data is
 * stored in the order of: [batch, in_channels, in_height, in_width].
 * @li filter: A 4D tensor of learnable filters. Must have the same type as
 * "x". With the format "NCHW" , the data is stored in the order of:
 * [out_channels, in_channels / groups, filter_height, filter_width].
 * @li offsets: A 4D tensor of x-y coordinates offset and mask. With the format
 * "NCHW", the data is stored in the order of: [batch, deformable_groups *
 * filter_height * filter_width * 3, out_height, out_width].
 * @li bias: An optional 1D tensor of additive biases to the filter outputs.
 *  The data is stored in the order of: [out_channels].
 * \n
 * \n
 *  The following are the supported data types and data formats:
 * \n
 * \n
 |  Tensor    | x       | filter  | offsets | bias    | y       |\n
    |  :-------: | :-----: | :-----: | :-----: | :-----: | :-----: |\n
    |  Data Type | float16 | float16 | float16 | float16 | float16 |\n
    |            | float32 | float32 | float32 | float32 | float32 |\n
    |  Format    | NCHW    | NCHW    | NCHW    | ND      | NCHW    |\n
    |            | NHWC    | HWCN    | NHWC    |         | NHWC    |\n
    * \n
    *  For float32 type, the actual convolution calculation part on the chip is
    *  based on float16.
    * \n
    *
    * @par Attributes:
    * @li strides: Required. A list of 4 integers. The stride of the sliding
    * window for each dimension of input. The dimension order is interpreted
    * according to the data format of "x". The N and C dimensions must be
    * set to 1.
    * @li pads: Required. A list of 4 integers. The number of pixels to add to
    * each (top, bottom, left, right) side of the input.
    * @li dilations: Optional. A list of 4 integers. The dilation factor for each
    * dimension of input. The dimension order is interpreted according to the
    * data format of "x". The N and C dimensions must be set to 1. Defaults to
    * [1, 1, 1, 1].
    * @li groups: Optional. An integer of type int32. The number of blocked
    * connections from input channels to output channels. In_channels and
    * out_channels must both be divisible by "groups". Defaults to 1.
    * @li data_format: Reserved.
    * @li deformable_groups: Optional. An integer of type int32. The number of
    * deformable group partitions. In_channels must be divisible by
    * "deformable_groups". Defaults to 1.
    * @li modulated: Optional. Specify version of DeformableConv2D, true means v2,
    * false means v1, currently only support v2.
    * \n
    * \n
    *  The following value range restrictions must be met:
    * \n
    * \n
    |  Name             | Field    | Scope                       |\n
    |  :--------------: | :------: | :-------------------------: |\n
    |  Input Image Size | H        | [1, 100000 / filter_height] |\n
    |                   | W        | [1, 4096 / filter_width]    |\n
    |  Filter Size      | H        | [1, 63]                     |\n
    |                   | W        | [1, 63]                     |\n
    |  Strides          | H        | [1, 63]                     |\n
    |                   | W        | [1, 63]                     |\n
    |  Pads             | Top      | [0, 255]                    |\n
    |                   | Bottom   | [0, 255]                    |\n
    |                   | Left     | [0, 255]                    |\n
    |                   | Right    | [0, 255]                    |\n
    |  Dilations        | H        | [1, 255]                    |\n
    |                   | W        | [1, 255]                    |\n
    * \n
    *
    * @par Outputs:
    *  y:  A 4D Tensor of output feature map. Has the same type as "x". With the
    *  format "NCHW", the data is stored in the order of: [batch, out_channels,
    *  out_height, out_width].
    * \n
    *      out_height = (in_height + pad_top + pad_bottom -
    *                    (dilation_h * (filter_height - 1) + 1))
    *                   / stride_h + 1
    * \n
    *      out_width = (in_width + pad_left + pad_right -
    *                   (dilation_w * (filter_width - 1) + 1))
    *                  / stride_w + 1
    * \n
    *
    * @par Quantization supported or not
    * @li No
    *
    * @par Third-party framework compatibility
    */

REG_OP(DeformableConv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(offsets, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(deformable_groups, Int, 1)
    .ATTR(modulated, Bool, true)
    .OP_END_FACTORY_REG(DeformableConv2D)
} // namespace ge
#endif // OPS_PROTO_DEF_DEFORMABLECONV2D
