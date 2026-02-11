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
 * \file deformable_offsets_proto.h
 * \brief
 */
#ifndef OPS_CONV_DEFORMABLE_OFFSETS_PLUGIN_DEFORMABLE_OFFSETS_PROTO_H_
#define OPS_CONV_DEFORMABLE_OFFSETS_PLUGIN_DEFORMABLE_OFFSETS_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes the deformed convolution output with the expected input
* @par Inputs:
* Two inputs:
* @li x: A 4D tensor of input image. A tensor of type float16, float32, bfloat16. The format support NHWC.
* Shape support 4D.
* @li offsets: A tensor of type float16, float32, bfloat16. Deformation offset parameter.
* The format support NHWC. Shape support 4D. Has the same format and dtype as "x".

*@par Attributes:
* @li strides: A tuple/list of 4 integers. The stride of the sliding window for
* height and width for H/W dimension. Required and no default value.
* @li pads: A tuple/list of 4 integers. Padding added to H/W dimension
* of the input. Required and no default value.
* @li ksize: A tuple/list of 2 integers. Kernel size. Required and no default value.
* @li dilations: A tuple/list of 4 integers. The dilation factor for each dimension
* of input. Defaults to [1, 1, 1, 1]
* @li data_format: An optional string from: "NCHW", "NHWC". The default value "NCHW" is not supported.
* Specify the data format of the input x. The format of the attribute
* @li deformable_groups: An optional int specify the c-axis grouping number of input x. Defaults to "1".
* @li modulated: An optional bool specify version of DeformableConv2D, true means v2, false means v1. Defaults to
"true".
* Only support true now.

*@par Outputs:
* y: Deformed convolution output. A tensor of type float16, float32, bfloat16. The format support NHWC.
* Shape support 4D. Has the same format and dtype as input "x".
*/
REG_OP(DeformableOffsets)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(offsets, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .REQUIRED_ATTR(ksize, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(data_format, String, "NHWC")
    .ATTR(deformable_groups, Int, 1)
    .ATTR(modulated, Bool, true)
    .OP_END_FACTORY_REG(DeformableOffsets)
} // namespace ge

#endif // OPS_CONV_DEFORMABLE_OFFSETS_PLUGIN_DEFORMABLE_OFFSETS_PROTO_H_
