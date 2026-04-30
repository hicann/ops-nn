/**
  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
  * CANN Open Software License Agreement Version 2.0 (the "License").
  * Please refer to the License for details. You may not use this file except in compliance with the License.
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
  * See LICENSE in the root of the software repository for the full text of the License.
  */

#ifndef OP_PROTO_MAX_POOL_H_
#define OP_PROTO_MAX_POOL_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs max pooling on the input .

* @par Inputs:
* One input:
* x: A 4-D Tensor. Supported type:float16, float32, double, int8, int16,
* int32, int64, uint8, uint16, qint8. Supported format: NHWC, NCHW.

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of
* the input tensor.
* @li padding: A required string. Supported modes: SAME, VALID. \n
* when padding is "SAME": pads 0 to ensure output shape equal to ceil(input shape / stride) ,
* (output shape equal to input shape when stride=1). \n
* when padding is "VALID": no padding. The kernel slides only over valid regions, resulting in smaller output .
* @li data_format: An optional string. Supported format: NHWC, NCHW. Defaults to "NHWC" . \n

* @par Outputs:
* y: A 4-D Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 4. The ksize of the H and W dimensions should be greater than 0.
* The ksize of the N and C dimensions should be 1. e.g. For "data_format" is "NCHW", ksize[0] = 1 and ksize[1] = 1.
* For "data_format" is "NHWC", ksize[0] = 1 and ksize[3] = 1. \n
* For Non-Ascend950PR/Ascend950DT: The produce of the ksize in H and W dimensions
* should be less than or equal to 255. e.g. For "data_format" is "NCHW", ksize[2] * ksize[3] <= 255. \n
* @li "strides" is a list that has length 4. The stride of the N and C dimensions should be 1. \n
* For Non-Ascend950PR/Ascend950DT: The stride of the H and W dimensions should be greater than 0 and
* smaller than 64. \n
* For Ascend950PR/Ascend950DT: The stride of the H and W dimensions should be greater than 0.
* @li The ouput "y" shape at the N and C dimensions should be equal with input "x" shape at same dimensions. The output
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
* Compatible with the TensorFlow operator MaxPool.
*/
REG_OP(MaxPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                          DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                          DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                           DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPool)

} // namespace ge
#endif