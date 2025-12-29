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
 * \file max_pool_with_argmax_v3_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Performs max pooling on the input and outputs both max values and indices.

* @par Inputs:
* One input:
* x: A tensor of type bfloat16, float16, float32, the shape is `[batch, channels, height_in, width_in]` or
  `[batch, height_in, width_in, channels]`.

* @par Attributes:
* @li ksize: A required list of int64 values,
* specifying the size of the window for each dimension of the input tensor.
* A list that has length 2.
* @li strides: A required list of int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor.
* A list that has length 2.
* @li pads: A required list of int64 values,
* specifying the pad of the input feature map. No default value.
* A list that has length 2:
* 0 <= pads[0] <= (ksize[0]//2), 0 <= pads[1] <= (ksize[1]//2).
* @li dilation: A list that has length 2, default value is {1,1}.
* @li dtype: An optional int. default value is 3.  (3 is int32, 9 is int64)
* @li ceil_mode: When true, will use ceil instead of floor to compute the output shape, defaults to false.
* @li data_format: The value can be "NCHW" or "NHWC", defaults to "NCHW".

* @par Outputs:
* @li y: A tensor has the same type and format as input "x", the shape is `[batch, channels, height_out, width_out]` or
  `[batch, height_out, width_out, channels]`.
* @li argmax:  A tensor of type is int64 or int32, the shape is `[batch, channels, height_out, width_out]` or
  `[batch, height_out, width_out, channels]`.

* @par Third-party framework compatibility
* Compatible with the PyTorch operator max_pool2d_with_indices.
*/
REG_OP(MaxPoolWithArgmaxV3)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(argmax, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1, 1})
    .ATTR(ceil_mode, Bool, false)
    .ATTR(data_format, String, "NCHW")
    .OP_END_FACTORY_REG(MaxPoolWithArgmaxV3)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_POOLING_OPS_H
