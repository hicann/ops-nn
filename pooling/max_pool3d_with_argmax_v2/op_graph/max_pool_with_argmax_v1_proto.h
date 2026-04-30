/**
  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
  * CANN Open Software License Agreement Version 2.0 (the "License").
  * Please refer to the License for details. You may not use this file except in compliance with the License.
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
  * See LICENSE in the root of the software repository for the full text of the License.
  */


#ifndef OPS_POOLING_MAX_POOL_WITH_ARGMAX_V1_PROTO_H_
#define OPS_POOLING_MAX_POOL_WITH_ARGMAX_V1_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs max pooling on the input and outputs both max values and indices.

* @par Inputs:
* One input:
* x: A tensor of type float16,float32, support format: [NC1HWC0, NCHW],
* format NCHW can only support float32. \n

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor.
* @li pads: A required list of int8, int16, int32, or int64 values,
* specifying the pad of the input feature map. \n
* @li dtype: An Attr which type is int8, int16, int32, or int64. Default value is 3.
* @li dilation:A list of int8, int16, int32, or int64 values, specifying the stride in each kernel.
* Default value is: [1,1,1,1].
* @li ceil_mode: An attr which type is bool, default value is false. When True,
* will use ceil instead of floor to compute the output shape.

* @par Outputs:
* @li y: A tensor. Has the same type and format as input "x".
* @li argmax:  A tensor. type:uint16, int32. 
* Format NCHW can only support int32, format NC1HWC0 can only support uint16,
* specifying when the dtype of argmax is int32, argmax must be index; when the dtype of argmax is uint16, argmax must be mask. \n

* @attention Constraints:
* @li The MaxPoolWithArgmaxV2 operator has the same function, and it is recommended to use the V2 operator.
* @li ksize: a list that has length 4:
* ksize[0] = 1, ksize[3] = 1, ksize[1] * ksize[2] < 1033.
* @li strides: a list that has length 4:
* strides[0] = 1, strides[3] = 1, 1 <= strides[1] <= 127, 1 <= strides[2] <= 127, strides[1] * strides[2] < 7932.
* @li pads: a list that has length 4:
* pads[0] = 1, pads[3] = 1, 0 <= pads[1] <= (ksize[1]//2), 0 <= pads[2] <= (ksize[2]//2).

* @par Third-party framework compatibility
* Compatible with the PyTorch operator max_pool2d_with_indices.
*/
REG_OP(MaxPoolWithArgmaxV1)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(argmax, TensorType({DT_UINT16, DT_INT32}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dtype, Int, 3)
    .ATTR(dilation, ListInt, {1, 1, 1, 1})
    .ATTR(ceil_mode, Bool, false)
    .OP_END_FACTORY_REG(MaxPoolWithArgmaxV1)

} // namespace ge
#endif