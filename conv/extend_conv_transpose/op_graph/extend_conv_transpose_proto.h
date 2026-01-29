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
 * \file extend_conv_transpose_proto.h
 * \brief
 */
#ifndef OPS_EXTEND_CONV_TRANSPOSE_PROTO_H_
#define OPS_EXTEND_CONV_TRANSPOSE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Computes the transpose of convolution 3d with respect to the input.

*@par Inputs:
 * @li input_size: A tensor of type int32 or int64. An integer vector
 * representing the shape of input.
 * @li x: A tensor of the following types, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8. Only support DT_FLOAT16 and DT_INT8 right now.
 * The format is NCDHW.
 * @li filter: A 5D tensor.
 * The format is NCDHW，and NDHWC.
 * The height (H), width (W) dimension must be in [1, 511].
 * @li bias: Optional. An optional 1D tensor of type float16, float32 or int32.
 * The data is stored in the order of: [out_channels].
 * @li scale: An optional 1D tensor. Quantization/dequantization/weighting parameter corresponding to
* the first output, which is of the channelwise type. \n

*@par Attributes:
 * @li strides: Required. A tuple/list of 5 integers. Specifies the stride of the sliding window
 * for each dimension of "x". The strides have the same axes sequence as "x":
 * [batch, stride_depth, stride_height, stride_width, channels] or
 * [batch, channels, stride_depth, stride_height, stride_width].
 * The N and C dimensions must be 1.
 * The height (H) and width (W) dimensions must be in [1, 63].
 * The depth (D) dimension must be in [1, 255].
 * @li pads: Required. A tuple/list of 6 integers.
 * All dimensions must be in [0, 255].
 * @li dilations: Optional. A tuple/list of 5 integers,
 * The dilation factor for each dimension of input. Defaults to [1, 1, 1, 1, 1]. \n
 * The batch(N) and channels dimensions must be 1.
 * The width (W), height (H) and depth(D) dimensions must be in [1, 255].
 * The dilations have the same axes sequence "x":
 * [batch, channels, dilation_depth, dilation_height, dilation_width] or
 * [batch, dilation_depth, dilation_height, dilation_width, channels].
 * @li groups: An optional integer within the effective range of [1, 65535]. Default to 1.
 * Number of blocked connections from in_channels to out_channels.
 * The in_channels and out_channels must be divisible by groups.
 * When the groups value differs, the supported data type may vary, specifically as follows: \n
 * \n
 * | groups |        dtype           |   x format  | filter format |    y format    |
 * |--------|------------------------|-------------|---------------|----------------|
 * |  =1    |int8                    |    NCDHW    |      NCDHW    |     NCDHW      |
 * |  =1    |int8                    |    NCDHW    |      NDHWC    |     NCDHW      |
 * |  =1    |float16                 |    NCDHW    |      NCDHW    |     NCDHW      |
 * |  =1    |float16                 |    NCDHW    |      NDHWC    |     NCDHW      |
 * |  >1    |int8                    |    NCDHW    |      NCDHW    |     NCDHW      |
 * |  >1    |float16                 |    NCDHW    |      NCDHW    |     NCDHW      |
 * |  >1    |float16                 |    NCDHW    |      NDHWC    |     NCDHW      |
 * \n
 * @li data_format:  An optional string. The value must be one of ["NDHWC", "NCDHW"]. Defaults to "NDHWC".
 * The correspondence is as follows: batch(N), depth(D), height(H), width(W), channels(C).
 * Specify the data format of the x and y.
 * @li output_padding: Optional. The size will be added in the output shape.
 * Defaults to [0, 0, 0, 0, 0]. The N and C dimensions must be 0.
 * @li offset_x: Optional. Defaults to 0. Reserved. 
 * @li fusion_mode: Optional. Only support 0 or 1, Indicates whether relu is enabled for the output. Defaults to 0.
* If 0, relu is disable for the output.
* @li y_quant_mode: Optional. Defaults to 0. Reserved. \n

*@par Outputs:
 * y: It has the same type and format as "x".
*/
REG_OP(ExtendConvTranspose)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT32}))
    .OPTIONAL_INPUT(scale, TensorType({DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT8}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .ATTR(fusion_mode, Int, 0)
    .ATTR(y_quant_mode, Int, 0)
    .OP_END_FACTORY_REG(ExtendConvTranspose)
} // namespace ge

#endif // OPS_EXTEND_CONV_TRANSPOSE_PROTO_H_
