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
 * \file max_pool_grad_with_argmax_proto.h
 * \brief
 */
 
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MAX_POOLING_GRAD_WITH_ARGMAX_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MAX_POOLING_GRAD_WITH_ARGMAX_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Performs the backpropagation of MaxPoolGradWithArgmax.

* @par Inputs:
* Three inputs, including:
* @li x: A tensor of dtype bfloat16, float16, float32, the shape is `[batch, channels, height_in, width_in]` or
  `[batch, height_in, width_in, channels]` , the format is `NCHW` or `NHWC`.
* @li grad: A tensor has the same dtype and format as input "x", the shape is `[batch, channels, height_out, width_out]` or
  `[batch, height_out, width_out, channels]`.
* @li argmax: A tensor has the same shape and format as input "grad", the dtype is int32 or int64.

* @par Attributes:
* @li ksize: A required list of int64 values,
* specifying the size of the window for each dimension of the input tensor. No default value.
* @li strides: A required list of int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor. No default value.
* @li padding: A string specifying the padding algorithm for the input feature map. No default value.
* @li include_batch_in_index: A boolean attribute indicating whether the batch dimension is included when computing argmax indices.
*     If false (default), the argmax index is computed within each batch independently, and the index only reflects the spatial and channel position.
*     If true, the batch dimension is included in the flattened index computation, so the argmax index spans across the batch dimension.
*     Default value: false. 
* @li data_format: A string specifying the data layout of the input and output tensors. Default value: NHWC. 
*     - "NHWC": Data is stored in the order [batch, height, width, channels].
*     - "NCHW": Data is stored in the order [batch, channels, height, width].

* @par Outputs:
* y: A tensor. Has the same dtype, shape and format as input "x".

* @attention Constraints:
* @li "ksize" is a list that has length 4: 
* If data_format is "NCHW", the ksize[0] = 1 or ksize[1] = 1,
* If data_format is "NHWC", the ksize[0] = 1 or ksize[3] = 1.
* @li "strides" is a list that has length 4: 
* If data_format is "NCHW", the strides[0] = 1 or strides[1] = 1,
* If data_format is "NHWC", the strides[0] = 1 or strides[3] = 1.
* @li padding: a string must be either "SAME" or "VALID".
* @li "include_batch_in_index": This operator currently only supports include_batch_in_index = false.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolGradWithArgmax.
*/
REG_OP(MaxPoolGradWithArgmax)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(argmax, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(include_batch_in_index, Bool, false)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolGradWithArgmax)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H