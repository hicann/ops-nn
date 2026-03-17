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
 * \file deformable_offsets_grad_proto.h
 * \brief
 */
#ifndef OPS_IMAGE_DEFORMABLE_OFFSETS_GRAD_PLUGIN_DEFORMABLE_OFFSETS_GRAD_PROTO_H_
#define OPS_IMAGE_DEFORMABLE_OFFSETS_GRAD_PLUGIN_DEFORMABLE_OFFSETS_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Computes the gradients of DeformableOffsets with respect to input and offsets
 * @par Inputs:
 * Three inputs:
 * @li grad: A Tensor of type float16,float32, bfloat16. gradients with respect to DeformableOffsets output
 * @li x: A Tensor of type float16,float32,bfloat16.
 * @li offsets: A Tensor of type float16,float32,bfloat16.Deformation offset parameter.
 *@par Attributes:
 * @li strides: A tuple/list of 4 integers.The stride of the sliding window for
 * height and width for H/W dimension.
 * @li pads: A tuple/list of 4 integers.Padding added to H/W dimension
 * of the input.
 * @li ksize: A tuple/list of 2 integers.kernel size.
 * @li dilations: A tuple/list of 4 integers, The dilation factor for each dimension
 * of input.  Defaults to [1, 1, 1, 1]
 * @li data_format: An optional string from: "NCHW", "NHWC". Defaults to "NCHW". Specify the data format of the input x.
 * @li deformable_groups: Specify the c-axis grouping number of input x.
 * @li modulated: Specify version of DeformableConv2D, true means v2, false means v1.
 *@par Outputs:
 * @li grad_x: A Tensor of type float16, float32, bfloat16. Gradients with respect to input_x
 * @li grad_offsets: A Tensor of type float16, float32, bfloat16. Gradients with respect to input_offsets
 * \n
 *      out_height = (in_height + pad_top + pad_bottom -
 *                    (dilation_h * (ksize_height - 1) + 1))
 *                   / stride_h + 1
 * \n
 *      out_width = (in_width + pad_left + pad_right -
 *                   (dilation_w * (ksize_width - 1) + 1))
 *                  / stride_w + 1
 * \n
 * @attention Constraints:
 * Multiplying the H/W of offsets by the H/W of ksize equals the H/W of grad.
 */
REG_OP(DeformableOffsetsGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(offsets, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad_offsets, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .REQUIRED_ATTR(ksize, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(data_format, String, "NCHW")
    .ATTR(deformable_groups, Int, 1)
    .ATTR(modulated, Bool, true)
    .OP_END_FACTORY_REG(DeformableOffsetsGrad)
} // namespace ge

#endif // OPS_IMAGE_DEFORMABLE_OFFSETS_GRAD_PLUGIN_DEFORMABLE_OFFSETS_GRAD_PROTO_H_
