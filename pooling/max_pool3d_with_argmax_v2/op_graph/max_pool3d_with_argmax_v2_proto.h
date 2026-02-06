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
 * \file max_pool3d_with_argmax_v2_proto.h
 * \brief
 */

#ifndef OPS_POOLING_MAX_POOL3D_WITH_ARGMAX_V2_PROTO_H_
#define OPS_POOLING_MAX_POOL3D_WITH_ARGMAX_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs max pooling on the input and outputs both max values and indices.

* @par Inputs:
* One input:
* x: A tensor of type bfloat16, float16, float32, the shape is [batch, channels, depth_in, height_in, width_in] or
  [batch, depth_in, height_in, width_in, channels].

* @par Attributes:
* @li ksize: A required list of int64 values,
* specifying the size of the window for each dimension of the input tensor.
* A list that has length 3.
* @li strides: A required list of int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor.
* A list that has length 3.
* @li pads: A required list of int64 values,
* specifying the pad of the input feature map.
* A list that has length 3:
* 0 <= pads[0] <= (ksize[0]//2), 0 <= pads[1] <= (ksize[1]//2), 0 <= pads[2] <= (ksize[2]//2).
* @li dilation: A list that has length 3, default value is {1,1,1}.
* @li ceil_mode: When true, will use ceil instead of floor to compute the output shape, defaults to false.
* @li data_format: The value can be "NCDHW" or "NDHWC", defaults to "NCDHW".
* @li dtype: An optional int, default value is 3.  (3 is int32, 9 is int64)

* @par Outputs:
* @li y: A tensor has the same type and format as input "x", the shape is [batch, channels, depth_out, height_out, width_out] or
  [batch, depth_out, height_out, width_out, channels].
* @li argmax:  A tensor of type is int64 or int32, the shape is [batch, channels, depth_out, height_out, width_out] or
  [batch, depth_out, height_out, width_out, channels].

* @par Third-party framework compatibility
* Compatible with the PyTorch operator max_pool3d_with_indices.
*/

REG_OP(MaxPool3DWithArgmaxV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_BF16}))
    .OUTPUT(argmax, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilation, ListInt, {1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(dtype, Int, 3)
    .OP_END_FACTORY_REG(MaxPool3DWithArgmaxV2)
} //namespace ge

#endif  //OPS_POOLING_MAX_POOL3D_WITH_ARGMAX_V2_PROTO_H_