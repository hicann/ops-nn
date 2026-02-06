/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_PROTO_MAX_POOL_V2_H_
#define OP_PROTO_MAX_POOL_V2_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs max_pool_ext2 on the input .

* @par Inputs:
* Three inputs:
* @li x: A Tensor of type float16.
* @li strides: A required type of int32 values,
* specifying the stride of the sliding window for each dimension of the input tensor. No default value.
* @li ksize: A required type of int32 values,
* specifying the size of the window for each dimension of the input tensor. No default value. \n

* @par Attributes:
* @li padding: A required string. No default value. is either "SAME" or "VALID" . \n
* @li data_format: An optional string. Defaults to "NHWC". Supported modes: "NHWC","NCHW". \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1, ksize[1] * ksize[2] <= 255.
* @li "stride" is a list that has length 4: strides[0] = 1 or strides[3] = 1, strides[1] <= 63, strides[0] >= 1,
* strides[2] <= 63, strides[2] >= 1.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPoolV2.
*/
REG_OP(MaxPoolV2)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(ksize, TensorType({DT_INT32}))
    .INPUT(strides, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPoolV2)

}  // namespace ge

#endif