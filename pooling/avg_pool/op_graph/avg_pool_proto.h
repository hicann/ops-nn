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
 * \file avg_pool_proto.h
 * \brief
 */
#ifndef OPS_POOLING_AVG_POOL_PROTO_H_
#define OPS_POOLING_AVG_POOL_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"



namespace ge {
/**
* @brief Performs average pooling on the input.

* @par Inputs:
* x: A tensor of shape [N, C, H, W] or [N, H, W, C] which supports data type float16, float32, bfloat16, double. \n

* @par Attributes:
* @li ksize: A required list of 4 ints, specifying the size of the sliding window,
* The ksize of the N and C dimensions are 1.
* @li strides: A required list of 4 ints, specifying the stride of the
* sliding window. The strides of the N and C dimensions are 1.
* @li padding: A required string, specifying the padding algorithm,
 * either "VALID" or "SAME". With "SAME" means that the outputs will have the
 * same spatial dimensions as its inputs. With "VALID" means no padding.
* @li data_format: An optional string, specifying the data format of "ksize"
* and "strides", either "NCHW", or "NHWC" (default). \n

* @par Outputs:
* y: The average pooled output tensor. Has the same type and format
* as input "x". \n

* @attention Constraints:
* @li This operator applies only to a TensorFlow network.
* @li Only single input and single output are supported.
* @li For Atlas Training Series Product, Atlas A2 Training Series Product/Atlas 800I A2 Inference Product,
* Atlas A3 Training Series Product: "ksize_H" and "ksize_W" are positive integers within the range [1, 255].
* ksize_H * ksize_W < 256. \n
* For Ascend 950 AI Processor: The ksize of the H and W dimensions should be greater than 0.
* @li For Atlas Training Series Product, Atlas A2 Training Series Product/Atlas 800I A2 Inference Product,
* Atlas A3 Training Series Product: the values of "strides_h" and "strides_w" are positive integers within
* the range [1, 63]. \n
* For Ascend 950 AI Processor: The stride of the H and W dimensions should be greater than 0.
* @li When the C axis is greater than 1, if points with the same H and W dimensions in x contain one INF input
* on the C axis, the output of the INF input covered by the sliding window on this C axis is INF, and the
* outputs of other C axis without INF input covered by the sliding window are Nan. If points with the same
* H and W dimensions in x contain more than one INF input on the C axis, the outputs of all INF input data
* covered by the sliding window on the C axis are Nan. this constraints not for Ascend 950 AI Processor.
* @li The output "y" shape at the N and C dimensions should be equal with input "x" shape at same dimensions. The output
* shape at the H and W dimensions is calculated by below formula: \n
* @code{.c}
    when "padding" is "SAME":
             out_height = (in_height + stride_h - 1) / stride_h
             out_width = (in_width + stride_w - 1) / stride_w
    when "padding" is "VALID":
             out_height = (in_height + stride_h - ksize_h) / stride_h
             out_width = (in_width + stride_w - ksize_w) / stride_w
  It not support out_height < 0 or out_width < 0.
* @endcode
 * @par Third-party framework compatibility
* Compatible with the TensorFlow operator AvgPool.
*/

REG_OP(AvgPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16, DT_DOUBLE}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(AvgPool)

} // namespace ge

#endif // OPS_POOLING_AVG_POOL_PROTO_H_
