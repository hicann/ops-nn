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
 * \file max_pool_grad_with_argmax_v3_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_MAX_POOLING_GRAD_WITH_ARGMAX_V3_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MAX_POOLING_GRAD_WITH_ARGMAX_V3_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Performs the backpropagation of MaxPoolGradWithArgmaxV3.

* @par Inputs:
* Three inputs, including:
* @li x: A tensor of dtype bfloat16, float16, float32, the shape is `[batch, channels, height_in, width_in]` or
  `[batch, height_in, width_in, channels]` , the format is `NCHW` or `NHWC`.
* @li grad: A tensor has the same dtype and format as input "x", the shape is `[batch, channels, height_out, width_out]`
or
  `[batch, height_out, width_out, channels]`.
* @li argmax: A tensor has the same shape and format as input "grad", the dtype is int32 or int64.

* @par Attributes:
* @li ksize: A required list of int64 values,
* specifying the size of the window for each dimension of the input tensor. No default value.
* @li strides: A required list of int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor. No default value.
* @li pads: A required list of int64 values,
* specifying the pad of the input feature map. No default value.

* @par Outputs:
* y: A Tensor. Has the same dtype , shape and format as input "x".

* @attention Constraints:
* @li The MaxPoolGradWithArgmaxV3 operator has the same function, and it is recommended to use the V3 operator.
* @li ksize: a list that has length 2:
* @li strides: a list that has length 2:
* @li pads: a list that has length 2:
* 1 <= pads[0] <= (ksize[0]//2), 1 <= pads[1] <= (ksize[1]//2).
* @li dilation: a list that has length 2. default value is {1,1}.
* @li dtype: A optional int. default value is 3.
* @li ceil_mode: defaults to False.
* @li data_format: defaults to "NCHW".

* @par Third-party framework compatibility
* Compatible with the Pytorch backward operator of max_pool2d_with_indices.
*/
REG_OP(MaxPoolGradWithArgmaxV3)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .INPUT(argmax, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1, 1})
    .ATTR(ceil_mode, Bool, false)
    .ATTR(data_format, String, "NCHW")
    .OP_END_FACTORY_REG(MaxPoolGradWithArgmaxV3)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H
